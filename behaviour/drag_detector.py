"""
Drag Detector
Detects products being dragged along the floor plane without mechanical trolleys or proper lifting.
Characteristics:
- Object bottom rests near or on floor zone
- Significant sustained horizontal velocity (vx) without vertical lifting (vy ~ 0)
- Sustained displacement across ground plane
- Operator in close proximity pulling or pushing
"""

from typing import List, Optional
import numpy as np
import uuid
from detection.tracker import TrackedObject
from detection.object_classes import WarehouseEntity
from behaviour.base import BehaviourEvent, BehaviourType
from risk.risk_engine import RiskEngine

class DragDetector:
    def __init__(self, min_drag_frames: int = 15, min_horiz_speed: float = 35.0):
        self.min_drag_frames = min_drag_frames
        self.min_horiz_speed = min_horiz_speed
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
            if len(history) < self.min_drag_frames:
                continue

            # Look at horizontal displacement over the recent window
            recent = history[-self.min_drag_frames:]
            vxs = [abs(h.get("vx", 0.0)) for h in recent]
            vys = [abs(h.get("vy", 0.0)) for h in recent]
            
            mean_vx = np.mean(vxs)
            mean_vy = np.mean(vys)
            
            # Floor proximity: object bottom is in lower 60% of vertical scene
            bottom_y = trk.box[3]
            
            # Horizontal motion dominates vertical motion heavily (dragging along plane)
            is_horizontal_slide = (mean_vx > self.min_horiz_speed) and (mean_vy < mean_vx * 0.45)
            
            # Check if there is an operator nearby pulling/dragging
            has_operator_near = False
            nearby_op_id = None
            for op in operators:
                # Distance between centers
                dist = np.hypot(op.center[0] - trk.center[0], op.center[1] - trk.center[1])
                if dist < 220.0:
                    has_operator_near = True
                    nearby_op_id = op.track_id
                    break

            # Total dragged distance
            drag_dist = abs(recent[-1]["center"][0] - recent[0]["center"][0])

            if is_horizontal_slide and has_operator_near and drag_dist > 50.0:
                self.alerted_tracks.add(trk.track_id)
                
                params = {
                    "drag_distance_px": drag_dist,
                    "mean_speed": round(mean_vx, 1),
                    "on_wet_floor": False
                }
                risk_level, risk_score, recommendation = RiskEngine.evaluate_risk(
                    BehaviourType.PRODUCT_DRAG, params
                )

                event = BehaviourEvent(
                    event_id=f"drag_{trk.track_id}_{frame_idx}_{uuid.uuid4().hex[:6]}",
                    behaviour_type=BehaviourType.PRODUCT_DRAG,
                    timestamp_sec=timestamp,
                    frame_idx=frame_idx,
                    object_track_id=trk.track_id,
                    operator_track_id=nearby_op_id,
                    confidence=0.88,
                    risk_level=risk_level,
                    risk_score=risk_score,
                    evidence_description=f"Product #{trk.track_id} dragged horizontally across warehouse floor for {drag_dist:.0f}px (velocity {mean_vx:.1f}px/s) without material handling equipment.",
                    root_cause="Operator dragged product directly over concrete floor instead of utilizing a trolley, hand pallet truck, or two-person team lift.",
                    recommended_action=recommendation,
                    bounding_box=trk.box,
                    metadata=params
                )
                events.append(event)

        return events
