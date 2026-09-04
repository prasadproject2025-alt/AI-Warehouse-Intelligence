"""
Base classes and schemas for the Behaviour Intelligence Engine.

Two things here are deliberate and load-bearing:

* ``ImplementationStatus`` - each detector declares honestly what it can do
  today. The dashboard renders this verbatim instead of claiming everything is
  "Active". A detector that needs footage or a trained model says so.
* ``EvidenceTier`` - the responsible-AI distinction the challenge asks for:
  observed behaviour -> potential risk -> confirmed damage. Nothing in this
  system can emit CONFIRMED_DAMAGE from video alone; that tier is reachable
  only through human confirmation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BehaviourType(str, Enum):
    PRODUCT_DROP = "product_drop"
    PRODUCT_DRAG = "product_drag"
    PRODUCT_THROW = "product_throw"
    ROLLING_PRODUCT = "rolling_product"
    IMPROPER_STACKING = "improper_stacking"
    STEPPING_ON_CARTON = "stepping_on_carton"
    UNSUPPORTED_HANDLING = "unsupported_handling"
    WET_FLOOR_HAZARD = "wet_floor_hazard"
    ORIENTATION_VIOLATION = "orientation_violation"
    DOCK_LEVEL_HAZARD = "dock_level_hazard"
    OUTSIDE_DESIGNATED_AREA = "outside_designated_area"
    UNSAFE_LOADING_SEQUENCE = "unsafe_loading_sequence"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class EvidenceTier(str, Enum):
    """How far the system is entitled to go in describing what it saw."""

    OBSERVED_BEHAVIOUR = "OBSERVED_BEHAVIOUR"
    POTENTIAL_RISK = "POTENTIAL_RISK"
    CONFIRMED_DAMAGE = "CONFIRMED_DAMAGE"  # only ever set by a human reviewer


class ImplementationStatus(str, Enum):
    IMPLEMENTED = "IMPLEMENTED"
    PARTIAL = "PARTIALLY_IMPLEMENTED"
    REQUIRES_DATA = "REQUIRES_ADDITIONAL_FOOTAGE"
    REQUIRES_TRAINING = "REQUIRES_MODEL_TRAINING"
    REQUIRES_CONFIG = "REQUIRES_ZONE_CONFIGURATION"


class BehaviourEvent(BaseModel):
    event_id: str
    behaviour_type: BehaviourType
    timestamp_sec: float
    frame_idx: int
    object_track_id: Optional[int] = None
    operator_track_id: Optional[int] = None
    confidence: float
    risk_level: RiskLevel
    risk_score: float  # 0-100
    evidence_description: str
    root_cause: str
    recommended_action: str
    bounding_box: List[float]  # [x1, y1, x2, y2]
    risk_factors: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_stages: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_tier: EvidenceTier = EvidenceTier.OBSERVED_BEHAVIOUR
    duration_sec: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

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
            "risk_factors": self.risk_factors,
            "evidence_stages": self.evidence_stages,
            "evidence_tier": self.evidence_tier.value,
            "duration_sec": round(self.duration_sec, 2),
            "metadata": self.metadata,
        }


class BaseBehaviourDetector:
    """
    Common scaffolding for behaviour detectors.

    Subclasses declare their capability honestly via the class attributes and
    implement ``process``. The cooldown here is per (behaviour, track) so a
    single continuous action produces one incident rather than a burst.
    """

    behaviour_type: BehaviourType
    status: ImplementationStatus = ImplementationStatus.IMPLEMENTED
    #: Short description of what the detector needs to work well.
    requirements: str = ""
    #: What it cannot do today. Rendered in the coverage table.
    limitations: str = ""

    def __init__(self, cooldown_sec: float = 4.0) -> None:
        self.cooldown_sec = cooldown_sec
        self._last_fired: Dict[Any, float] = {}

    def _cooled_down(self, key: Any, timestamp: float) -> bool:
        last = self._last_fired.get(key)
        return last is None or (timestamp - last) >= self.cooldown_sec

    def _mark_fired(self, key: Any, timestamp: float) -> None:
        self._last_fired[key] = timestamp

    @classmethod
    def describe(cls) -> Dict[str, str]:
        return {
            "behaviour_type": cls.behaviour_type.value,
            "status": cls.status.value,
            "requirements": cls.requirements,
            "limitations": cls.limitations,
        }


def stages_from_track(track, last_n: int = 6) -> List[Dict[str, Any]]:
    """Serialise a track's recent motion-state transitions as event evidence."""
    return [
        {"stage": state, "at_sec": round(ts, 2)}
        for state, ts in track.state_history[-last_n:]
    ]
