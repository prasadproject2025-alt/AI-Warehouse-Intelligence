"""
Base classes and schemas for the Behaviour Intelligence Engine.
"""

from typing import Dict, Any, Optional, List
from enum import Enum
from pydantic import BaseModel

class BehaviourType(str, Enum):
    PRODUCT_DROP = "product_drop"
    PRODUCT_DRAG = "product_drag"
    PRODUCT_THROW = "product_throw"
    ROLLING_PRODUCT = "rolling_product"
    IMPROPER_STACKING = "improper_stacking"
    STEPPING_ON_CARTON = "stepping_on_carton"
    STRAP_PULLING = "strap_pulling"
    WET_FLOOR_HAZARD = "wet_floor_hazard"
    ORIENTATION_VIOLATION = "orientation_violation"
    DOCK_LEVEL_HAZARD = "dock_level_hazard"

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class BehaviourEvent(BaseModel):
    event_id: str
    behaviour_type: BehaviourType
    timestamp_sec: float
    frame_idx: int
    object_track_id: Optional[int] = None
    operator_track_id: Optional[int] = None
    confidence: float
    risk_level: RiskLevel
    risk_score: float # 0 to 100
    evidence_description: str
    root_cause: str
    recommended_action: str
    bounding_box: List[float] # [x1, y1, x2, y2]
    metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "behaviour_type": self.behaviour_type.value,
            "timestamp_sec": round(self.timestamp_sec, 2),
            "frame_idx": self.frame_idx,
            "object_track_id": self.object_track_id,
            "operator_track_id": self.operator_track_id,
            "confidence": round(self.confidence, 3),
            "risk_level": self.risk_level.value,
            "risk_score": round(self.risk_score, 1),
            "evidence_description": self.evidence_description,
            "root_cause": self.root_cause,
            "recommended_action": self.recommended_action,
            "bounding_box": [round(b, 1) for b in self.bounding_box],
            "metadata": self.metadata
        }
