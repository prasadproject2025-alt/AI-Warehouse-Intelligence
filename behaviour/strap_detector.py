"""
Strap Pulling Detector
Detects operators pulling or lifting cartons using packaging straps rather than base lifting points.
Characteristics:
- Operator grip / interaction points are concentrated strictly on upper perimeter of carton
- No lifting support or hand contact observed under carton bottom half
- Carton is tilted or pulled under tensile stress on strapping bands
"""

from typing import List, Optional
import uuid
import numpy as np
from detection.tracker import TrackedObject
from detection.object_classes import WarehouseEntity
from behaviour.base import BehaviourEvent, BehaviourType
from risk.risk_engine import RiskEngine

class StrapDetector:
    def __init__(self, check_interval: int = 15):
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

        operators = [t for t in tracked_objects if t.entity_type == WarehouseEntity.OPERATOR]
        products = [t for t in tracked_objects if t.entity_type == WarehouseEntity.CARTON]

        for op in operators:
            for prod in products:
                pair_key = (op.track_id, prod.track_id)
                if pair_key in self.alerted_tracks:
                    continue

                # Check proximity between operator hands/torso (upper/mid body) and product top edge
                op_hands_y = op.box[1] + (op.height * 0.5)
                prod_top_y = prod.box[1]

                # Distance between operator center and carton top edge
                dist_y = abs(op_hands_y - prod_top_y)
                dist_x = abs(op.center[0] - prod.center[0])

                # When pulling with strap: operator stands at arm's length (dist_x ~ 80-180px)
                # and hands are aligned right at the top strap level, with carton moving
                is_moving = abs(prod.vx) > 20.0 or abs(prod.vy) > 15.0
                if dist_y < 50.0 and 60.0 < dist_x < 220.0 and is_moving:
                    self.alerted_tracks.add(pair_key)
                    
                    params = {
                        "operator_track_id": op.track_id,
                        "product_track_id": prod.track_id,
                        "grip_distance_x": round(dist_x, 1)
                    }
                    risk_level, risk_score, recommendation = RiskEngine.evaluate_risk(
                        BehaviourType.STRAP_PULLING, params
                    )

                    event = BehaviourEvent(
                        event_id=f"strap_{op.track_id}_{prod.track_id}_{frame_idx}_{uuid.uuid4().hex[:6]}",
                        behaviour_type=BehaviourType.STRAP_PULLING,
                        timestamp_sec=timestamp,
                        frame_idx=frame_idx,
                        object_track_id=prod.track_id,
                        operator_track_id=op.track_id,
                        confidence=0.87,
                        risk_level=risk_level,
                        risk_score=risk_score,
                        evidence_description=f"Operator #{op.track_id} pulling/holding Carton #{prod.track_id} using packaging straps instead of base support.",
                        root_cause="Operator utilized packaging strapping band as a lifting handle, causing local tensile tearing of corrugated box walls.",
                        recommended_action=recommendation,
                        bounding_box=prod.box,
                        metadata=params
                    )
                    events.append(event)

        return events
