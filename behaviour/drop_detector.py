"""
Drop Detector
Identifies product drop events via temporal kinematics:
1. Object in motion or held
2. Rapid downward acceleration (vy > threshold)
3. Sudden impact deceleration at floor level (vy -> 0)
4. Object remains stationary post-impact
"""

from typing import List, Optional
import uuid
from detection.tracker import TrackedObject
from detection.object_classes import WarehouseEntity
from behaviour.base import BehaviourEvent, BehaviourType
from risk.risk_engine import RiskEngine

class DropDetector:
    def __init__(self, vy_threshold: float = 120.0, drop_min_displacement: float = 40.0):
        self.vy_threshold = vy_threshold
        self.drop_min_displacement = drop_min_displacement
        self.triggered_tracks = set()

    def process(
        self,
        tracked_objects: List[TrackedObject],
        frame_idx: int,
        timestamp: float
    ) -> List[BehaviourEvent]:
        events: List[BehaviourEvent] = []

        for trk in tracked_objects:
            if trk.entity_type == WarehouseEntity.OPERATOR:
                continue
            
            # Prevent multiple duplicate alerts on the exact same continuous event
            if trk.track_id in self.triggered_tracks:
                continue

            history = trk.history
            if len(history) < 10:
                continue

            # Check kinematics over recent history
            window = min(len(history), 15)
            vys = [h.get("vy", 0.0) for h in history[-window:]]
            max_vy = max(vys)
            
            # 2. Check vertical displacement (y increased from higher to lower)
            y_start = history[-window]["center"][1]
            y_end = history[-1]["center"][1]
            delta_y = y_end - y_start

            # 3. Check current post-impact velocity (is it decelerating or stationary?)
            curr_speed = trk.vx**2 + trk.vy**2

            if max_vy > self.vy_threshold and delta_y > self.drop_min_displacement and curr_speed < 80.0:
                self.triggered_tracks.add(trk.track_id)
                
                params = {
                    "downward_velocity": max_vy,
                    "drop_height_px": delta_y,
                    "stationary_after_impact": curr_speed < 40.0
                }
                risk_level, risk_score, recommendation = RiskEngine.evaluate_risk(
                    BehaviourType.PRODUCT_DROP, params
                )

                event = BehaviourEvent(
                    event_id=f"drop_{trk.track_id}_{frame_idx}_{uuid.uuid4().hex[:6]}",
                    behaviour_type=BehaviourType.PRODUCT_DROP,
                    timestamp_sec=timestamp,
                    frame_idx=frame_idx,
                    object_track_id=trk.track_id,
                    confidence=min(0.96, 0.75 + (max_vy / 500.0)),
                    risk_level=risk_level,
                    risk_score=risk_score,
                    evidence_description=f"Product #{trk.track_id} experienced rapid downward acceleration (peak {max_vy:.1f} px/s, vertical drop ~{delta_y:.0f}px) and impacted the floor surface.",
                    root_cause="Operator dropped carton during manual loading/unloading transfer instead of lowering it in a controlled motion.",
                    recommended_action=recommendation,
                    bounding_box=trk.box,
                    metadata=params
                )
                events.append(event)

        return events
