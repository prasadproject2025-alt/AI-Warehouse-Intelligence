"""
Orientation Violation Detector
Detects vertical products (such as cupboards, wardrobes, tall KD packets, or appliances)
stored or staged in a horizontal orientation.
Characteristics:
- Product categorized as upright/vertical good (cupboard, tall carton, etc.)
- Current bounding box aspect ratio indicates horizontal placement (w >> h)
"""

from typing import List, Optional
import uuid
from detection.tracker import TrackedObject
from detection.object_classes import WarehouseEntity
from behaviour.base import BehaviourEvent, BehaviourType
from risk.risk_engine import RiskEngine

class OrientationDetector:
    def __init__(self, check_interval: int = 20):
        self.check_interval = check_interval
        self.alerted_tracks = set()

    def process(
        self,
        tracked_objects: List[TrackedObject],
        frame_idx: int,
        timestamp: float
    ) -> List[BehaviourEvent]:
        events: List[BehaviourEvent] = []
        if frame_idx % self.check_interval != 0:
            return events

        for trk in tracked_objects:
            if trk.track_id in self.alerted_tracks:
                continue

            # Check if this object is a cupboard or tall carton stored horizontally
            # If cupboard has width > height * 1.3, it is laid flat on its side or back
            is_cupboard = trk.entity_type == WarehouseEntity.CUPBOARD
            is_horizontal_orientation = trk.width > (trk.height * 1.25)

            if is_cupboard and is_horizontal_orientation:
                self.alerted_tracks.add(trk.track_id)
                
                params = {
                    "aspect_ratio": round(trk.width / max(1.0, trk.height), 2),
                    "width": round(trk.width, 1),
                    "height": round(trk.height, 1)
                }
                risk_level, risk_score, recommendation = RiskEngine.evaluate_risk(
                    BehaviourType.ORIENTATION_VIOLATION, params
                )

                event = BehaviourEvent(
                    event_id=f"orient_{trk.track_id}_{frame_idx}_{uuid.uuid4().hex[:6]}",
                    behaviour_type=BehaviourType.ORIENTATION_VIOLATION,
                    timestamp_sec=timestamp,
                    frame_idx=frame_idx,
                    object_track_id=trk.track_id,
                    confidence=0.92,
                    risk_level=risk_level,
                    risk_score=risk_score,
                    evidence_description=f"Orientation violation: Vertical cupboard #{trk.track_id} positioned horizontally (width {trk.width:.0f}px > height {trk.height:.0f}px).",
                    root_cause="Product staged horizontally in violation of 'This Side Up' orientation marking, risking panel deflection, hinge strain, and internal component damage.",
                    recommended_action=recommendation,
                    bounding_box=trk.box,
                    metadata=params
                )
                events.append(event)

        return events
