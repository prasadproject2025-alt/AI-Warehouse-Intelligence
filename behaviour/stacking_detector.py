"""
Stacking Detector
Detects improper stacking practices:
- Heavy/larger packets placed on top of smaller/lighter packets
- Inadequate bottom support causing crushing risk or unstable overhang
"""

from typing import List, Optional
import uuid
from detection.tracker import TrackedObject
from detection.object_classes import WarehouseEntity
from behaviour.base import BehaviourEvent, BehaviourType
from risk.risk_engine import RiskEngine

class StackingDetector:
    def __init__(self, check_interval: int = 15):
        self.check_interval = check_interval
        self.alerted_pairs = set()

    def process(
        self,
        tracked_objects: List[TrackedObject],
        frame_idx: int,
        timestamp: float
    ) -> List[BehaviourEvent]:
        events: List[BehaviourEvent] = []
        
        # Only evaluate every N frames to avoid redundant compute
        if frame_idx % self.check_interval != 0:
            return events

        products = [t for t in tracked_objects if t.entity_type != WarehouseEntity.OPERATOR]

        # Compare pairs of stationary or resting products
        for i in range(len(products)):
            for j in range(len(products)):
                if i == j:
                    continue
                top_obj = products[i]
                bot_obj = products[j]

                pair_key = (min(top_obj.track_id, bot_obj.track_id), max(top_obj.track_id, bot_obj.track_id))
                if pair_key in self.alerted_pairs:
                    continue

                # Check vertical stacking: top_obj bottom is near bot_obj top
                # top_obj y2 is close to bot_obj y1
                top_bottom_y = top_obj.box[3]
                bot_top_y = bot_obj.box[1]
                
                # Check horizontal overlap
                x_overlap = max(0.0, min(top_obj.box[2], bot_obj.box[2]) - max(top_obj.box[0], bot_obj.box[0]))
                min_width = min(top_obj.width, bot_obj.width)
                
                if x_overlap > 0.5 * min_width and abs(top_bottom_y - bot_top_y) < 45.0:
                    # Top object is resting on bottom object
                    # Violation: Top object is significantly larger/wider than bottom object
                    # or top object is classified as heavy furniture/cupboard on lighter carton
                    is_size_inverted = top_obj.width > (bot_obj.width * 1.25)
                    is_heavy_on_light = (
                        top_obj.entity_type in [WarehouseEntity.CUPBOARD, WarehouseEntity.EQUIPMENT] and
                        bot_obj.entity_type == WarehouseEntity.CARTON
                    )

                    if is_size_inverted or is_heavy_on_light:
                        self.alerted_pairs.add(pair_key)
                        
                        params = {
                            "heavy_on_light": is_heavy_on_light,
                            "top_width": round(top_obj.width, 1),
                            "bottom_width": round(bot_obj.width, 1),
                            "top_track_id": top_obj.track_id,
                            "bottom_track_id": bot_obj.track_id
                        }
                        risk_level, risk_score, recommendation = RiskEngine.evaluate_risk(
                            BehaviourType.IMPROPER_STACKING, params
                        )

                        desc = (
                            f"Improper stacking detected: Heavy/larger item #{top_obj.track_id} "
                            f"(width {top_obj.width:.0f}px) stacked directly on top of smaller/lighter packet "
                            f"#{bot_obj.track_id} (width {bot_obj.width:.0f}px)."
                        )

                        event = BehaviourEvent(
                            event_id=f"stack_{pair_key[0]}_{pair_key[1]}_{frame_idx}_{uuid.uuid4().hex[:6]}",
                            behaviour_type=BehaviourType.IMPROPER_STACKING,
                            timestamp_sec=timestamp,
                            frame_idx=frame_idx,
                            object_track_id=top_obj.track_id,
                            confidence=0.92,
                            risk_level=risk_level,
                            risk_score=risk_score,
                            evidence_description=desc,
                            root_cause="Heavier or larger dimension product placed atop lighter packaging, risking carton wall collapse and load toppling.",
                            recommended_action=recommendation,
                            bounding_box=top_obj.box,
                            metadata=params
                        )
                        events.append(event)

        return events
