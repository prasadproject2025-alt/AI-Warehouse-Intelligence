"""
Scene-context behaviour detectors: wet floor and dock-level transition.

Both behaviours depend on a scene condition that this pilot's footage cannot
establish from pixels alone:

* **Wet floor.** A specular-reflection classifier (bright, low-saturation floor
  pixels) was evaluated against the pilot videos and did **not** separate the
  wet-floor clip from the dry ones - the wet clip actually scored lower than
  two dry clips. Rather than ship a detector that would fabricate hazards, the
  floor condition is a declared scene input (supervisor report, floor sensor or
  the ingest form), and the detector then does the part it can do reliably:
  decide whether goods were actually moved through the affected area.

* **Dock transition.** Whether a bay is a vehicle dock is site knowledge, so it
  is declared per camera. The detector then verifies the observable part: a
  heavy item sliding across the transition with no leveller or trolley present.

Previously both of these were switched on by matching the *filename*, which
meant the result depended on what a file happened to be called.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List

from behaviour.base import (
    BaseBehaviourDetector,
    BehaviourEvent,
    BehaviourType,
    ImplementationStatus,
    stages_from_track,
)
from detection.tracker import MotionState, TrackedObject
from risk.risk_engine import RiskEngine


class WetFloorDetector(BaseBehaviourDetector):
    behaviour_type = BehaviourType.WET_FLOOR_HAZARD
    status = ImplementationStatus.PARTIAL
    requirements = (
        "Floor condition must be declared at ingest (or supplied by a floor sensor). "
        "Automatic wet-floor sensing needs footage where reflections are distinguishable."
    )
    limitations = (
        "The wet condition itself is not detected from video in this build - a specular "
        "classifier was tested on the pilot clips and could not separate wet from dry."
    )

    MIN_EXPOSURE_SEC = 1.0

    def __init__(self, cooldown_sec: float = 8.0) -> None:
        super().__init__(cooldown_sec)
        self._exposed_since: Dict[int, float] = {}

    def process(
        self,
        tracks: List[TrackedObject],
        frame_idx: int,
        timestamp: float,
        ctx: Dict[str, Any],
    ) -> List[BehaviourEvent]:
        if not ctx.get("wet_floor_active"):
            return []

        events: List[BehaviourEvent] = []
        floor = ctx["floor_line_norm"]

        for trk in tracks:
            if not trk.is_product or trk.hits < 5:
                continue
            in_floor_band = trk.bottom_y_norm > (floor - 0.08)
            moving = trk.state in (MotionState.SLIDING, MotionState.CARRIED)
            if not (in_floor_band and moving):
                self._exposed_since.pop(trk.track_id, None)
                continue

            since = self._exposed_since.setdefault(trk.track_id, timestamp)
            exposure = timestamp - since
            if exposure < self.MIN_EXPOSURE_SEC or not self._cooled_down(trk.track_id, timestamp):
                continue

            params = {
                "wet_zone_coverage": float(ctx.get("wet_zone_coverage", 0.3)),
                "product_in_zone": True,
                "exposure_sec": round(exposure, 2),
                "condition_source": ctx.get("floor_condition_source", "declared_at_ingest"),
                "detection_confidence": trk.confidence,
            }
            assessment = RiskEngine.evaluate(
                self.behaviour_type,
                params,
                product_type=trk.entity_type.value,
                recurrence_count=ctx["recurrence"].get(self.behaviour_type.value, 0),
                bay=ctx.get("bay"),
            )
            self._mark_fired(trk.track_id, timestamp)
            events.append(
                BehaviourEvent(
                    event_id=f"wet_{trk.track_id}_{frame_idx}_{uuid.uuid4().hex[:6]}",
                    behaviour_type=self.behaviour_type,
                    timestamp_sec=timestamp,
                    frame_idx=frame_idx,
                    object_track_id=trk.track_id,
                    operator_track_id=trk.operator_contact_id,
                    confidence=float(min(0.82, 0.45 + trk.confidence * 0.35)),
                    risk_level=assessment.level,
                    risk_score=assessment.score,
                    evidence_description=(
                        f"{trk.entity_type.value.title()} #{trk.track_id} was moved along the floor "
                        f"for {exposure:.1f} s while the floor condition for this bay is recorded as "
                        f"WET (source: {params['condition_source']})."
                    ),
                    root_cause=(
                        "Material movement continued over a floor reported as wet, risking moisture "
                        "ingress through the package base and loss of footing."
                    ),
                    recommended_action=assessment.recommendation,
                    bounding_box=trk.box,
                    risk_factors=assessment.factors_as_dicts(),
                    evidence_stages=stages_from_track(trk),
                    evidence_tier=assessment.evidence_tier,
                    duration_sec=exposure,
                    metadata=params,
                )
            )
        return events


class DockDetector(BaseBehaviourDetector):
    behaviour_type = BehaviourType.DOCK_LEVEL_HAZARD
    status = ImplementationStatus.PARTIAL
    requirements = "Camera must be declared as covering a vehicle dock transition."
    limitations = (
        "The dock gap height is not measured. The detector confirms an unaided heavy "
        "transfer across the declared transition, not the size of the level difference."
    )

    MIN_SLIDE_SEC = 1.0
    MIN_SIZE_NORM = 0.20

    def __init__(self, cooldown_sec: float = 8.0) -> None:
        super().__init__(cooldown_sec)
        self._slide_since: Dict[int, float] = {}

    def process(
        self,
        tracks: List[TrackedObject],
        frame_idx: int,
        timestamp: float,
        ctx: Dict[str, Any],
    ) -> List[BehaviourEvent]:
        if not ctx.get("dock_transfer_active"):
            return []

        events: List[BehaviourEvent] = []
        equipment = bool(ctx.get("handling_equipment_present"))

        for trk in tracks:
            if not trk.is_product or trk.hits < 5:
                continue
            size_norm = max(trk.width, trk.height) / trk.frame_height
            if size_norm < self.MIN_SIZE_NORM or trk.state is not MotionState.SLIDING:
                self._slide_since.pop(trk.track_id, None)
                continue

            since = self._slide_since.setdefault(trk.track_id, timestamp)
            sliding = timestamp - since
            if sliding < self.MIN_SLIDE_SEC or not self._cooled_down(trk.track_id, timestamp):
                continue

            params = {
                "vehicle_present": bool(ctx.get("vehicle_detected")),
                "no_leveller_detected": not equipment,
                "item_size_norm": round(size_norm, 3),
                "slide_seconds": round(sliding, 2),
                "detection_confidence": trk.confidence,
            }
            assessment = RiskEngine.evaluate(
                self.behaviour_type,
                params,
                product_type=trk.entity_type.value,
                recurrence_count=ctx["recurrence"].get(self.behaviour_type.value, 0),
                bay=ctx.get("bay"),
            )
            self._mark_fired(trk.track_id, timestamp)
            events.append(
                BehaviourEvent(
                    event_id=f"dock_{trk.track_id}_{frame_idx}_{uuid.uuid4().hex[:6]}",
                    behaviour_type=self.behaviour_type,
                    timestamp_sec=timestamp,
                    frame_idx=frame_idx,
                    object_track_id=trk.track_id,
                    operator_track_id=trk.operator_contact_id,
                    confidence=float(min(0.82, 0.45 + trk.confidence * 0.35)),
                    risk_level=assessment.level,
                    risk_score=assessment.score,
                    evidence_description=(
                        f"{trk.entity_type.value.title()} #{trk.track_id} "
                        f"({size_norm:.2f} frame-heights across) was slid across the declared dock "
                        f"transition for {sliding:.1f} s"
                        + ("." if equipment else " with no leveller, bridge plate or trolley detected.")
                    ),
                    root_cause=(
                        "Heavy package transferred across the vehicle/dock threshold by dragging, "
                        "exposing the base to step impact at the level change."
                    ),
                    recommended_action=assessment.recommendation,
                    bounding_box=trk.box,
                    risk_factors=assessment.factors_as_dicts(),
                    evidence_stages=stages_from_track(trk),
                    evidence_tier=assessment.evidence_tier,
                    duration_sec=sliding,
                    metadata=params,
                )
            )
        return events
