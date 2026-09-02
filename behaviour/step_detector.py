"""
Step on Cartons Detector
Detects operators stepping, standing, or walking on top of cartons or packages.
Characteristics:
- Operator lower bounding box (feet/legs) overlaps with product top surface
- Operator vertical weight vector rests directly above product
"""

from typing import List, Optional
import uuid
from detection.tracker import TrackedObject
from detection.object_classes import WarehouseEntity
from behaviour.base import BehaviourEvent, BehaviourType
from risk.risk_engine import RiskEngine

class StepDetector:
    def __init__(self, check_interval: int = 10):
        self.check_interval = check_interval
        self.alerted_pairs = set()

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
        products = [t for t in tracked_objects if t.entity_type != WarehouseEntity.OPERATOR]

        for op in operators:
            # Feet region: bottom 20% of operator box
            feet_y1 = op.box[3] - (op.height * 0.22)
            feet_y2 = op.box[3]
            feet_box = [op.box[0], feet_y1, op.box[2], feet_y2]

            for prod in products:
                pair_key = (op.track_id, prod.track_id)
                if pair_key in self.alerted_pairs:
                    continue

                # Check if feet box overlaps with the top half of the product box
                prod_top_half = [prod.box[0], prod.box[1], prod.box[2], prod.box[1] + (prod.height * 0.55)]
                
                # Check horizontal overlap
                x_overlap = max(0.0, min(feet_box[2], prod_top_half[2]) - max(feet_box[0], prod_top_half[0]))
                y_overlap = max(0.0, min(feet_box[3], prod_top_half[3]) - max(feet_box[1], prod_top_half[1]))
                
                if x_overlap > 30.0 and y_overlap > 15.0:
                    self.alerted_pairs.add(pair_key)
                    
                    params = {
                        "operator_track_id": op.track_id,
                        "product_track_id": prod.track_id,
                        "overlap_area": round(x_overlap * y_overlap, 1)
                    }
                    risk_level, risk_score, recommendation = RiskEngine.evaluate_risk(
                        BehaviourType.STEPPING_ON_CARTON, params
                    )

                    event = BehaviourEvent(
                        event_id=f"step_{op.track_id}_{prod.track_id}_{frame_idx}_{uuid.uuid4().hex[:6]}",
                        behaviour_type=BehaviourType.STEPPING_ON_CARTON,
                        timestamp_sec=timestamp,
                        frame_idx=frame_idx,
                        object_track_id=prod.track_id,
                        operator_track_id=op.track_id,
                        confidence=0.94,
                        risk_level=risk_level,
                        risk_score=risk_score,
                        evidence_description=f"Critical handling violation: Operator #{op.track_id} observed standing / stepping directly onto Carton #{prod.track_id}.",
                        root_cause="Operator stepped on product cartons to reach elevated storage or navigate cluttered staging bay rather than using clear walkways.",
                        recommended_action=recommendation,
                        bounding_box=op.box,
                        metadata=params
                    )
                    events.append(event)

        return events
