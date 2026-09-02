"""
Wet Floor Hazard Detector
Detects handling, dragging, or staging of cartons over wet or moisture-hazard floor zones.
"""

from typing import List, Optional
import uuid
from detection.tracker import TrackedObject
from detection.object_classes import WarehouseEntity
from behaviour.base import BehaviourEvent, BehaviourType
from risk.risk_engine import RiskEngine

class WetFloorDetector:
    def __init__(self, check_interval: int = 15):
        self.check_interval = check_interval
        self.alerted_tracks = set()

    def process(
        self,
        tracked_objects: List[TrackedObject],
        frame_idx: int,
        timestamp: float,
        is_wet_environment: bool = False
    ) -> List[BehaviourEvent]:
        events: List[BehaviourEvent] = []
        if frame_idx % self.check_interval != 0:
            return events

        # If wet environment flag is active (or detected from dock reflections / floor moisture zone)
        for trk in tracked_objects:
            if trk.entity_type == WarehouseEntity.OPERATOR:
                continue
            if trk.track_id in self.alerted_tracks:
                continue

            # Ground contact check (lower 40% of frame is floor)
            is_on_floor = trk.box[3] > 400.0 # Standard 720p floor baseline
            is_moving_on_floor = is_on_floor and (abs(trk.vx) > 15.0 or trk.distance_travelled > 40.0)

            if is_wet_environment and is_moving_on_floor:
                self.alerted_tracks.add(trk.track_id)
                
                params = {
                    "product_track_id": trk.track_id,
                    "floor_y": round(trk.box[3], 1),
                    "on_wet_floor": True
                }
                risk_level, risk_score, recommendation = RiskEngine.evaluate_risk(
                    BehaviourType.WET_FLOOR_HAZARD, params
                )

                event = BehaviourEvent(
                    event_id=f"wet_{trk.track_id}_{frame_idx}_{uuid.uuid4().hex[:6]}",
                    behaviour_type=BehaviourType.WET_FLOOR_HAZARD,
                    timestamp_sec=timestamp,
                    frame_idx=frame_idx,
                    object_track_id=trk.track_id,
                    confidence=0.90,
                    risk_level=risk_level,
                    risk_score=risk_score,
                    evidence_description=f"Product #{trk.track_id} moved across wet/slick warehouse dock floor without elevation pallet.",
                    root_cause="Material handling conducted directly over wet concrete, causing water absorption through corrugated base and potential slip-and-fall.",
                    recommended_action=recommendation,
                    bounding_box=trk.box,
                    metadata=params
                )
                events.append(event)

        return events
