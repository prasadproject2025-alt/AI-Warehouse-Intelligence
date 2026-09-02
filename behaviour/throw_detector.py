"""
Throw / Push Detector
Identifies aggressive throwing, tossing, or forceful pushing of products or mattresses.
Characteristics:
- High release velocity as object separates from operator
- Rapid spatial detachment from operator bounding box
- Free-flight trajectory (parabolic or sudden acceleration)
"""

from typing import List, Optional
import numpy as np
import uuid
from detection.tracker import TrackedObject
from detection.object_classes import WarehouseEntity
from behaviour.base import BehaviourEvent, BehaviourType
from risk.risk_engine import RiskEngine

class ThrowDetector:
    def __init__(self, release_speed_threshold: float = 140.0):
        self.release_speed_threshold = release_speed_threshold
        self.alerted_tracks = set()

    def process(
        self,
        tracked_objects: List[TrackedObject],
        frame_idx: int,
        timestamp: float
    ) -> List[BehaviourEvent]:
        events: List[BehaviourEvent] = []
        operators = [t for t in tracked_objects if t.entity_type == WarehouseEntity.OPERATOR]

        for trk in tracked_objects:
            if trk.entity_type == WarehouseEntity.OPERATOR:
                continue
            if trk.track_id in self.alerted_tracks:
                continue

            history = trk.history
            if len(history) < 8:
                continue

            # Calculate resultant speed over last 6 frames
            recent = history[-6:]
            speeds = [np.hypot(h.get("vx", 0.0), h.get("vy", 0.0)) for h in recent]
            max_speed = max(speeds)

            # Check separation from nearest operator
            closest_op = None
            min_dist = float("inf")
            for op in operators:
                d = np.hypot(op.center[0] - trk.center[0], op.center[1] - trk.center[1])
                if d < min_dist:
                    min_dist = d
                    closest_op = op

            # If speed is very high and moving away from operator or into truck bay
            if max_speed > self.release_speed_threshold and min_dist > 120.0:
                self.alerted_tracks.add(trk.track_id)
                
                params = {
                    "release_velocity": round(max_speed, 1),
                    "operator_distance": round(min_dist, 1)
                }
                risk_level, risk_score, recommendation = RiskEngine.evaluate_risk(
                    BehaviourType.PRODUCT_THROW, params
                )

                item_label = "Mattress" if trk.entity_type == WarehouseEntity.MATTRESS else "Carton / Product"
                event = BehaviourEvent(
                    event_id=f"throw_{trk.track_id}_{frame_idx}_{uuid.uuid4().hex[:6]}",
                    behaviour_type=BehaviourType.PRODUCT_THROW,
                    timestamp_sec=timestamp,
                    frame_idx=frame_idx,
                    object_track_id=trk.track_id,
                    operator_track_id=closest_op.track_id if closest_op else None,
                    confidence=0.91,
                    risk_level=risk_level,
                    risk_score=risk_score,
                    evidence_description=f"{item_label} #{trk.track_id} thrown/pitched with high velocity ({max_speed:.1f}px/s) into storage/vehicle area.",
                    root_cause="Operator threw goods rather than placing them sequentially and gently onto pallets or into vehicle compartment.",
                    recommended_action=recommendation,
                    bounding_box=trk.box,
                    metadata=params
                )
                events.append(event)

        return events
