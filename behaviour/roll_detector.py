"""
Roll Detector
Detects rolling or tumbling of cartons or mattresses along warehouse floors.
Characteristics:
- Object aspect ratio (width/height) oscillates cyclically
- Object translates across ground plane
- Contact remains continuous with floor without lifting equipment
"""

from typing import List, Optional
import numpy as np
import uuid
from detection.tracker import TrackedObject
from detection.object_classes import WarehouseEntity
from behaviour.base import BehaviourEvent, BehaviourType
from risk.risk_engine import RiskEngine

class RollDetector:
    def __init__(self, min_roll_frames: int = 20):
        self.min_roll_frames = min_roll_frames
        self.alerted_tracks = set()

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
            if trk.track_id in self.alerted_tracks:
                continue

            history = trk.history
            if len(history) < self.min_roll_frames:
                continue

            # Look at aspect ratio oscillations over the window
            recent = history[-self.min_roll_frames:]
            aspect_ratios = [h.get("aspect_ratio", 1.0) for h in recent]
            
            # Count aspect ratio flips across 1.0 threshold (width vs height dominance)
            flips = 0
            for k in range(1, len(aspect_ratios)):
                prev_ratio = aspect_ratios[k - 1]
                curr_ratio = aspect_ratios[k]
                if (prev_ratio > 1.15 and curr_ratio < 0.85) or (prev_ratio < 0.85 and curr_ratio > 1.15):
                    flips += 1

            # Check translation distance
            dx = abs(recent[-1]["center"][0] - recent[0]["center"][0])
            dy = abs(recent[-1]["center"][1] - recent[0]["center"][1])

            # Tumbling end-over-end produces alternating dimensions with floor translation
            if (flips >= 2 or (flips >= 1 and dx > 60.0)):
                self.alerted_tracks.add(trk.track_id)
                
                params = {
                    "aspect_ratio_flips": flips,
                    "roll_distance_px": round(dx, 1)
                }
                risk_level, risk_score, recommendation = RiskEngine.evaluate_risk(
                    BehaviourType.ROLLING_PRODUCT, params
                )

                event = BehaviourEvent(
                    event_id=f"roll_{trk.track_id}_{frame_idx}_{uuid.uuid4().hex[:6]}",
                    behaviour_type=BehaviourType.ROLLING_PRODUCT,
                    timestamp_sec=timestamp,
                    frame_idx=frame_idx,
                    object_track_id=trk.track_id,
                    confidence=0.86,
                    risk_level=risk_level,
                    risk_score=risk_score,
                    evidence_description=f"Product #{trk.track_id} rolled / tumbled end-over-end along floor ({flips} geometric inversion cycles, {dx:.0f}px translation).",
                    root_cause="Operator tumbled/rolled product on concrete floor instead of utilizing a hand truck or lifting onto a pallet.",
                    recommended_action=recommendation,
                    bounding_box=trk.box,
                    metadata=params
                )
                events.append(event)

        return events
