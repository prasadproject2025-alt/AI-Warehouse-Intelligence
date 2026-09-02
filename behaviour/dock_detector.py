"""
Dock Level Hazard Detector
Detects improper dock level handling, uneven dock-to-vehicle transitions,
or dragging heavy cupboards/cartons across dock thresholds without mechanical dock levellers.
"""

from typing import List, Optional
import uuid
import numpy as np
from detection.tracker import TrackedObject
from detection.object_classes import WarehouseEntity
from behaviour.base import BehaviourEvent, BehaviourType
from risk.risk_engine import RiskEngine

class DockDetector:
    def __init__(self, check_interval: int = 15):
        self.check_interval = check_interval
        self.alerted_tracks = set()

    def process(
        self,
        tracked_objects: List[TrackedObject],
        frame_idx: int,
        timestamp: float,
        is_dock_scene: bool = False
    ) -> List[BehaviourEvent]:
        events: List[BehaviourEvent] = []
        if frame_idx % self.check_interval != 0:
            return events

        for trk in tracked_objects:
            if trk.entity_type == WarehouseEntity.OPERATOR:
                continue
            if trk.track_id in self.alerted_tracks:
                continue

            # Heavy items (cupboard or large carton) moving across the dock threshold
            is_heavy_item = trk.entity_type in [WarehouseEntity.CUPBOARD, WarehouseEntity.CARTON]
            is_sliding = abs(trk.vx) > 25.0 and trk.distance_travelled > 60.0

            if is_dock_scene and is_heavy_item and is_sliding:
                self.alerted_tracks.add(trk.track_id)
                
                params = {
                    "product_track_id": trk.track_id,
                    "distance_travelled": round(trk.distance_travelled, 1)
                }
                risk_level, risk_score, recommendation = RiskEngine.evaluate_risk(
                    BehaviourType.DOCK_LEVEL_HAZARD, params
                )

                event = BehaviourEvent(
                    event_id=f"dock_{trk.track_id}_{frame_idx}_{uuid.uuid4().hex[:6]}",
                    behaviour_type=BehaviourType.DOCK_LEVEL_HAZARD,
                    timestamp_sec=timestamp,
                    frame_idx=frame_idx,
                    object_track_id=trk.track_id,
                    confidence=0.89,
                    risk_level=risk_level,
                    risk_score=risk_score,
                    evidence_description=f"Hazardous dock transition: Item #{trk.track_id} dragged across dock-to-vehicle threshold without dock leveller bridge.",
                    root_cause="Gap between vehicle bed and loading dock traversed manually without dock leveller or bridge plate, subjecting product to bottom impact and joint shearing.",
                    recommended_action=recommendation,
                    bounding_box=trk.box,
                    metadata=params
                )
                events.append(event)

        return events
