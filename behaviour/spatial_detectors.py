"""
Spatial and configuration behaviour detectors.

These reason about the geometric relationship between operators, products and
the configured warehouse zones, held over time so that a momentary bounding-box
overlap cannot raise an incident on its own.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from behaviour.base import (
    BaseBehaviourDetector,
    BehaviourEvent,
    BehaviourType,
    ImplementationStatus,
    stages_from_track,
)
from detection.object_classes import (
    HANDLING_EQUIPMENT_ENTITIES,
    WarehouseEntity,
)
from detection.tracker import MotionState, TrackedObject
from risk.risk_engine import RiskEngine


def _x_overlap(a: TrackedObject, b: TrackedObject) -> float:
    return max(0.0, min(a.box[2], b.box[2]) - max(a.box[0], b.box[0]))


def _point_in_polygon(x: float, y: float, poly: List[List[float]]) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-9) + x1
            if x < xin:
                inside = not inside
    return inside


class StackingDetector(BaseBehaviourDetector):
    """
    Improper stacking: a wider or heavier package resting on a narrower or
    lighter one. Requires the configuration to persist while both items are at
    rest, so a carton passing in front of another does not trigger it.
    """

    behaviour_type = BehaviourType.IMPROPER_STACKING
    status = ImplementationStatus.IMPLEMENTED
    requirements = "Both packages detected simultaneously and stationary for >=1.5 s."
    limitations = (
        "Weight is inferred from apparent size and product class, not measured; "
        "a small dense package on a large light one is not distinguishable from video."
    )

    #: How long the pair may go unobserved before the stack is treated as gone.
    PAIR_GRACE = 0.7  # seconds

    OVERHANG_RATIO = 1.25
    MIN_STABLE_SEC = 1.5

    def __init__(self, cooldown_sec: float = 6.0) -> None:
        super().__init__(cooldown_sec)
        self._pair_since: Dict[Tuple[int, int], float] = {}
        self._pair_last_seen: Dict[Tuple[int, int], float] = {}

    def process(
        self,
        tracks: List[TrackedObject],
        frame_idx: int,
        timestamp: float,
        ctx: Dict[str, Any],
    ) -> List[BehaviourEvent]:
        events: List[BehaviourEvent] = []
        products = [t for t in tracks if t.is_product and t.hits >= 4]
        seen_pairs = set()

        for top in products:
            for bot in products:
                if top.track_id == bot.track_id:
                    continue
                # 'top' must rest on 'bot': its base meets the other's top edge.
                gap = abs(top.box[3] - bot.box[1])
                if gap > 0.05 * top.frame_height:
                    continue
                overlap = _x_overlap(top, bot)
                if overlap < 0.5 * min(top.width, bot.width):
                    continue
                # Both must be settled, not mid-transfer.
                if top.speed > 0.05 or bot.speed > 0.05:
                    continue

                key = (top.track_id, bot.track_id)
                seen_pairs.add(key)
                since = self._pair_since.setdefault(key, timestamp)
                stable = timestamp - since
                if stable < self.MIN_STABLE_SEC:
                    continue
                if not self._cooled_down(key, timestamp):
                    continue

                ratio = top.width / max(1.0, bot.width)
                heavy_on_light = (
                    top.entity_type is WarehouseEntity.CUPBOARD
                    and bot.entity_type is WarehouseEntity.CARTON
                )
                if ratio < self.OVERHANG_RATIO and not heavy_on_light:
                    continue

                params = {
                    "width_ratio": round(ratio, 2),
                    "heavy_on_light": heavy_on_light,
                    "stable_seconds": round(stable, 2),
                    "top_track_id": top.track_id,
                    "bottom_track_id": bot.track_id,
                    "detection_confidence": min(top.confidence, bot.confidence),
                }
                assessment = RiskEngine.evaluate(
                    self.behaviour_type,
                    params,
                    product_type=top.entity_type.value,
                    recurrence_count=ctx["recurrence"].get(self.behaviour_type.value, 0),
                    bay=ctx.get("bay"),
                )
                self._mark_fired(key, timestamp)
                events.append(
                    BehaviourEvent(
                        event_id=f"stack_{top.track_id}_{bot.track_id}_{frame_idx}_{uuid.uuid4().hex[:6]}",
                        behaviour_type=self.behaviour_type,
                        timestamp_sec=timestamp,
                        frame_idx=frame_idx,
                        object_track_id=top.track_id,
                        confidence=float(min(0.88, 0.5 + min(top.confidence, bot.confidence) * 0.4)),
                        risk_level=assessment.level,
                        risk_score=assessment.score,
                        evidence_description=(
                            f"{top.entity_type.value.title()} #{top.track_id} has rested on "
                            f"{bot.entity_type.value} #{bot.track_id} for {stable:.1f} s at "
                            f"{ratio:.2f}x its base width, leaving the upper package unsupported at the edges."
                        ),
                        root_cause=(
                            "Stacking order places a larger or heavier package above a smaller or "
                            "lighter one, loading the lower carton walls beyond their design limit."
                        ),
                        recommended_action=assessment.recommendation,
                        bounding_box=top.box,
                        risk_factors=assessment.factors_as_dicts(),
                        evidence_stages=stages_from_track(top),
                        evidence_tier=assessment.evidence_tier,
                        duration_sec=stable,
                        metadata=params,
                    )
                )

        # As in SteppingDetector: a stack does not dismantle itself because one
        # frame missed a detection. Without a grace window the 1.5 s stability
        # requirement was reset by every dropout and could never be reached.
        for key in seen_pairs:
            self._pair_last_seen[key] = timestamp
        for key in list(self._pair_since):
            last = self._pair_last_seen.get(key, timestamp)
            if timestamp - last > self.PAIR_GRACE:
                self._pair_since.pop(key, None)
                self._pair_last_seen.pop(key, None)
        return events


class SteppingDetector(BaseBehaviourDetector):
    """
    Operator standing or stepping on packaging.

    The naive test (operator box overlaps carton box) fires constantly in 2D
    because people routinely stand *behind* stacks. A single global floor line
    is no better: on a receding ground plane a distant worker's feet are
    legitimately high in the frame, which flagged every background worker.

    This detector therefore compares each operator's feet against the ground
    plane *at their own depth*, inferred from their apparent stature, and
    requires the excess elevation to exceed the noise of that fit and persist.
    """

    behaviour_type = BehaviourType.STEPPING_ON_CARTON
    status = ImplementationStatus.IMPLEMENTED
    requirements = (
        "A fitted ground plane (>=25 operator observations spanning a range of depths) "
        "and a detected package beneath the operator."
    )
    limitations = (
        "Assumes one continuous ground plane. On a ramp or split-level dock the fit is "
        "ambiguous, and the detector stays silent rather than guessing."
    )

    #: Elevation must exceed this multiple of the ground-plane fit residual.
    ELEVATION_SIGMA = 2.5
    MIN_ELEVATION = 0.045  # absolute floor, in frame-heights
    #: Elapsed-time floor. Kept small because sparse sampling, not duration,
    #: is the limiting factor; MIN_OBSERVATIONS carries the confirmation.
    MIN_DWELL = 0.15       # seconds
    #: Independent frames that must confirm the contact before it is reported.
    MIN_OBSERVATIONS = 3
    #: How long a contact may go unobserved before it is treated as ended.
    #: Sized above the detector's typical dropout so a genuine stand is not
    #: chopped into fragments, but well below a realistic step-off-and-return.
    CONTACT_GRACE = 0.7    # seconds
    #: Largest elevation a package step can plausibly produce, in frame-heights.
    #: If the adaptive threshold exceeds this the ground-plane fit is unusable.
    MAX_ELEVATION = 0.16

    def __init__(self, cooldown_sec: float = 6.0) -> None:
        super().__init__(cooldown_sec)
        # Keyed by product track id; see the note at the dwell timer.
        #: Set when the ground-plane fit is too noisy to make a determination.
        self.unable_to_judge: Optional[str] = None
        self._contact_since: Dict[int, float] = {}
        self._contact_hits: Dict[int, int] = {}
        self._contact_frame: Dict[int, int] = {}
        self._contact_last_seen: Dict[int, float] = {}

    def process(
        self,
        tracks: List[TrackedObject],
        frame_idx: int,
        timestamp: float,
        ctx: Dict[str, Any],
    ) -> List[BehaviourEvent]:
        events: List[BehaviourEvent] = []
        ground_plane = ctx.get("ground_plane")
        if ground_plane is None:
            # No usable ground-plane fit yet: cannot separate "elevated" from
            # "further away", so report nothing rather than guess.
            return events
        residual = ctx.get("ground_plane_residual", 0.05)
        threshold = max(self.MIN_ELEVATION, self.ELEVATION_SIGMA * residual)
        if threshold > self.MAX_ELEVATION:
            # The fit is too noisy to judge. Standing on a package raises an
            # operator by well under this much, so demanding more would mean the
            # detector could never fire while still appearing active. Report the
            # limitation instead of failing silently.
            self.unable_to_judge = (
                f"ground-plane fit too noisy (residual {residual:.3f} requires "
                f"{threshold:.3f} elevation, above the {self.MAX_ELEVATION:.3f} "
                f"a package step can physically produce)"
            )
            return events
        self.unable_to_judge = None

        operators = [t for t in tracks if t.entity_type is WarehouseEntity.OPERATOR and t.hits >= 4]
        products = [t for t in tracks if t.is_product and t.hits >= 4]
        seen = set()

        for op in operators:
            expected_floor = ground_plane(op.height / op.frame_height)
            if expected_floor is None:
                continue
            feet_norm = op.box[3] / op.frame_height
            # Positive means the feet sit above the floor at this operator's depth.
            elevation = expected_floor - feet_norm
            if elevation < threshold:
                continue  # standing on the floor at their own depth, as expected

            for prod in products:
                # Feet horizontally inside the package footprint.
                feet_cx = op.center[0]
                if not (prod.box[0] <= feet_cx <= prod.box[2]):
                    continue
                # Feet at the package's top surface, not far above or below it.
                if abs(op.box[3] - prod.box[1]) > 0.10 * op.frame_height:
                    continue

                # Keyed on the package, not the (operator, package) pair.
                # The question this timer answers is "how long has this package
                # been stood on", and operator identity is not stable enough to
                # answer it: on the pilot footage a single 0.8 s step was split
                # across two operator track ids into 0.2 s and 0.4 s fragments,
                # neither of which met the dwell requirement. The per-operator
                # elevation test above still runs every frame, so this does not
                # weaken the false-positive guard.
                key = prod.track_id
                seen.add(key)
                since = self._contact_since.setdefault(key, timestamp)
                dwell = timestamp - since
                # Count distinct frames, not operator/package pairings: two
                # operators near the same package must not satisfy the
                # confirmation requirement inside a single frame.
                if self._contact_frame.get(key) != frame_idx:
                    self._contact_frame[key] = frame_idx
                    self._contact_hits[key] = self._contact_hits.get(key, 0) + 1

                # Confirmation is by count of independent observations, not by
                # elapsed time alone. A step onto a package is brief and the
                # product is only detected in ~59% of frames, so a 0.6 s wall
                # clock requirement was unreachable even when the operator was
                # plainly elevated. Requiring several confirming observations
                # rejects single-frame coincidence, which is what the dwell
                # gate was actually for, while the depth-aware elevation test
                # above carries the real false-positive burden.
                if self._contact_hits[key] < self.MIN_OBSERVATIONS:
                    continue
                if dwell < self.MIN_DWELL or not self._cooled_down(key, timestamp):
                    continue

                params = {
                    "elevation_above_floor_norm": round(elevation, 3),
                    "elevation_threshold_norm": round(threshold, 3),
                    "dwell_sec": round(dwell, 2),
                    "operator_track_id": op.track_id,
                    "product_track_id": prod.track_id,
                    "detection_confidence": min(op.confidence, prod.confidence),
                }
                assessment = RiskEngine.evaluate(
                    self.behaviour_type,
                    params,
                    product_type=prod.entity_type.value,
                    recurrence_count=ctx["recurrence"].get(self.behaviour_type.value, 0),
                    bay=ctx.get("bay"),
                )
                self._mark_fired(key, timestamp)
                events.append(
                    BehaviourEvent(
                        event_id=f"step_{op.track_id}_{prod.track_id}_{frame_idx}_{uuid.uuid4().hex[:6]}",
                        behaviour_type=self.behaviour_type,
                        timestamp_sec=timestamp,
                        frame_idx=frame_idx,
                        object_track_id=prod.track_id,
                        operator_track_id=op.track_id,
                        confidence=float(min(0.88, 0.5 + min(op.confidence, prod.confidence) * 0.4)),
                        risk_level=assessment.level,
                        risk_score=assessment.score,
                        evidence_description=(
                            f"Operator #{op.track_id} stood {elevation:.2f} frame-heights above the "
                            f"ground plane at their own depth (threshold {threshold:.2f}), directly on "
                            f"{prod.entity_type.value} #{prod.track_id}, for {dwell:.1f} s."
                        ),
                        root_cause=(
                            "Packaging used as a step or working platform to reach height or cross "
                            "an obstructed staging area."
                        ),
                        recommended_action=assessment.recommendation,
                        bounding_box=op.box,
                        risk_factors=assessment.factors_as_dicts(),
                        evidence_stages=stages_from_track(op),
                        evidence_tier=assessment.evidence_tier,
                        duration_sec=dwell,
                        metadata=params,
                    )
                )

        # Detection dropout is not the same as the operator stepping off.
        # Forgetting the contact the instant a pair is missing reset the dwell
        # timer on every dropped frame, so with intermittent product detection
        # the dwell requirement could never be satisfied. Contact is retained
        # across short gaps and only discarded once the pair has genuinely been
        # absent for longer than the grace window.
        for key in seen:
            self._contact_last_seen[key] = timestamp
        for key in list(self._contact_since):
            last = self._contact_last_seen.get(key, timestamp)
            if timestamp - last > self.CONTACT_GRACE:
                self._contact_since.pop(key, None)
                self._contact_last_seen.pop(key, None)
                self._contact_hits.pop(key, None)
                self._contact_frame.pop(key, None)
        return events


class OrientationDetector(BaseBehaviourDetector):
    """
    Upright-marked product laid flat.

    Only fires when the *same tracked item* has been observed upright and is
    then observed flat. Flagging every wide box as an orientation violation
    (the previous behaviour) is meaningless, because most packages are wider
    than they are tall from a ceiling camera.
    """

    behaviour_type = BehaviourType.ORIENTATION_VIOLATION
    status = ImplementationStatus.PARTIAL
    requirements = "The item must be tracked through the upright-to-flat transition."
    limitations = (
        "Handling arrows and 'this way up' labels are not read. An item already flat "
        "when it enters frame cannot be judged, and is not reported."
    )

    UPRIGHT_ASPECT = 0.80
    FLAT_ASPECT = 1.30
    MIN_FLAT_SEC = 1.5

    def __init__(self, cooldown_sec: float = 8.0) -> None:
        super().__init__(cooldown_sec)
        self._flat_since: Dict[int, float] = {}

    def process(
        self,
        tracks: List[TrackedObject],
        frame_idx: int,
        timestamp: float,
        ctx: Dict[str, Any],
    ) -> List[BehaviourEvent]:
        events: List[BehaviourEvent] = []

        for trk in tracks:
            if not trk.is_product or trk.hits < 8:
                continue
            # The item must have genuinely been upright at some point.
            if trk.min_aspect_seen > self.UPRIGHT_ASPECT:
                continue
            if trk.aspect_ratio < self.FLAT_ASPECT:
                self._flat_since.pop(trk.track_id, None)
                continue

            since = self._flat_since.setdefault(trk.track_id, timestamp)
            flat_sec = timestamp - since
            if flat_sec < self.MIN_FLAT_SEC or not self._cooled_down(trk.track_id, timestamp):
                continue

            params = {
                "observed_transition": True,
                "flat_seconds": round(flat_sec, 2),
                "upright_aspect": round(trk.min_aspect_seen, 2),
                "current_aspect": round(trk.aspect_ratio, 2),
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
                    event_id=f"orient_{trk.track_id}_{frame_idx}_{uuid.uuid4().hex[:6]}",
                    behaviour_type=self.behaviour_type,
                    timestamp_sec=timestamp,
                    frame_idx=frame_idx,
                    object_track_id=trk.track_id,
                    confidence=float(min(0.85, 0.45 + trk.confidence * 0.4)),
                    risk_level=assessment.level,
                    risk_score=assessment.score,
                    evidence_description=(
                        f"{trk.entity_type.value.title()} #{trk.track_id} was observed upright "
                        f"(aspect {trk.min_aspect_seen:.2f}) and has now been held flat "
                        f"(aspect {trk.aspect_ratio:.2f}) for {flat_sec:.1f} s."
                    ),
                    root_cause=(
                        "Package laid on its side or back, contrary to the upright orientation its "
                        "internal structure and packaging were designed for."
                    ),
                    recommended_action=assessment.recommendation,
                    bounding_box=trk.box,
                    risk_factors=assessment.factors_as_dicts(),
                    evidence_stages=stages_from_track(trk),
                    evidence_tier=assessment.evidence_tier,
                    duration_sec=flat_sec,
                    metadata=params,
                )
            )
        return events


class UnsupportedHandlingDetector(BaseBehaviourDetector):
    """
    Large or heavy product handled manually with no material-handling equipment
    anywhere in the working area. This is the challenge's "product handled
    without required equipment" scenario, expressed in terms the system can
    actually observe.
    """

    behaviour_type = BehaviourType.UNSUPPORTED_HANDLING
    status = ImplementationStatus.IMPLEMENTED
    requirements = "Trolleys/pallet trucks must be visible in frame when present."
    limitations = (
        "Absence of equipment in frame is not proof it was unavailable; the finding is "
        "raised as an opportunity to check equipment provisioning, not as a violation."
    )

    MIN_SIZE_NORM = 0.22
    MIN_CARRY_SEC = 1.5

    def __init__(self, cooldown_sec: float = 8.0) -> None:
        super().__init__(cooldown_sec)
        self._carry_since: Dict[int, float] = {}

    def process(
        self,
        tracks: List[TrackedObject],
        frame_idx: int,
        timestamp: float,
        ctx: Dict[str, Any],
    ) -> List[BehaviourEvent]:
        events: List[BehaviourEvent] = []
        if ctx.get("handling_equipment_present"):
            self._carry_since.clear()
            return events

        for trk in tracks:
            if not trk.is_product or trk.hits < 6:
                continue
            size_norm = max(trk.width, trk.height) / trk.frame_height
            moving_with_operator = (
                trk.operator_contact_id is not None
                and trk.state in (MotionState.CARRIED, MotionState.SLIDING)
            )
            if size_norm < self.MIN_SIZE_NORM or not moving_with_operator:
                self._carry_since.pop(trk.track_id, None)
                continue

            since = self._carry_since.setdefault(trk.track_id, timestamp)
            carried = timestamp - since
            if carried < self.MIN_CARRY_SEC or not self._cooled_down(trk.track_id, timestamp):
                continue

            params = {
                "handling_equipment_present": False,
                "item_size_norm": round(size_norm, 3),
                "carry_seconds": round(carried, 2),
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
                    event_id=f"unsup_{trk.track_id}_{frame_idx}_{uuid.uuid4().hex[:6]}",
                    behaviour_type=self.behaviour_type,
                    timestamp_sec=timestamp,
                    frame_idx=frame_idx,
                    object_track_id=trk.track_id,
                    operator_track_id=trk.operator_contact_id,
                    confidence=float(min(0.80, 0.45 + trk.confidence * 0.35)),
                    risk_level=assessment.level,
                    risk_score=assessment.score,
                    evidence_description=(
                        f"{trk.entity_type.value.title()} #{trk.track_id} "
                        f"({size_norm:.2f} frame-heights across) was moved manually for "
                        f"{carried:.1f} s with no trolley, pallet truck or forklift detected in the area."
                    ),
                    root_cause=(
                        "Large package moved by hand where mechanical handling aid would be "
                        "the specified method."
                    ),
                    recommended_action=assessment.recommendation,
                    bounding_box=trk.box,
                    risk_factors=assessment.factors_as_dicts(),
                    evidence_stages=stages_from_track(trk),
                    evidence_tier=assessment.evidence_tier,
                    duration_sec=carried,
                    metadata=params,
                )
            )
        return events


class DesignatedAreaDetector(BaseBehaviourDetector):
    """
    Product left outside the designated staging area.

    Requires a staging polygon to be configured for the camera. With no zone
    configured the detector reports its status and emits nothing - it does not
    guess where the staging area is.
    """

    behaviour_type = BehaviourType.OUTSIDE_DESIGNATED_AREA
    status = ImplementationStatus.REQUIRES_CONFIG
    requirements = (
        "A staging-zone polygon must be supplied per camera (normalised coordinates) "
        "via the video ingest form or the zones API."
    )
    limitations = "Without a configured zone the system cannot know where goods belong."

    MIN_DWELL = 3.0

    def __init__(self, cooldown_sec: float = 10.0) -> None:
        super().__init__(cooldown_sec)
        self._outside_since: Dict[int, float] = {}

    def process(
        self,
        tracks: List[TrackedObject],
        frame_idx: int,
        timestamp: float,
        ctx: Dict[str, Any],
    ) -> List[BehaviourEvent]:
        zone: Optional[List[List[float]]] = ctx.get("staging_zone")
        if not zone or len(zone) < 3:
            return []

        events: List[BehaviourEvent] = []
        for trk in tracks:
            if not trk.is_product or trk.hits < 6:
                continue
            if trk.state not in (MotionState.SETTLED, MotionState.STATIONARY):
                self._outside_since.pop(trk.track_id, None)
                continue

            fx = trk.center[0] / trk.frame_width
            fy = trk.box[3] / trk.frame_height  # base contact point
            if _point_in_polygon(fx, fy, zone):
                self._outside_since.pop(trk.track_id, None)
                continue

            since = self._outside_since.setdefault(trk.track_id, timestamp)
            dwell = timestamp - since
            if dwell < self.MIN_DWELL or not self._cooled_down(trk.track_id, timestamp):
                continue

            params = {
                "dwell_sec": round(dwell, 2),
                "position_norm": [round(fx, 3), round(fy, 3)],
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
                    event_id=f"zone_{trk.track_id}_{frame_idx}_{uuid.uuid4().hex[:6]}",
                    behaviour_type=self.behaviour_type,
                    timestamp_sec=timestamp,
                    frame_idx=frame_idx,
                    object_track_id=trk.track_id,
                    confidence=float(min(0.86, 0.5 + trk.confidence * 0.4)),
                    risk_level=assessment.level,
                    risk_score=assessment.score,
                    evidence_description=(
                        f"{trk.entity_type.value.title()} #{trk.track_id} has been stationary outside "
                        f"the configured staging zone for {dwell:.1f} s."
                    ),
                    root_cause=(
                        "Goods staged outside the marked area, obstructing movement paths and "
                        "exposing the package to impact from passing traffic."
                    ),
                    recommended_action=assessment.recommendation,
                    bounding_box=trk.box,
                    risk_factors=assessment.factors_as_dicts(),
                    evidence_stages=stages_from_track(trk),
                    evidence_tier=assessment.evidence_tier,
                    duration_sec=dwell,
                    metadata=params,
                )
            )
        return events


class LoadingSequenceDetector(BaseBehaviourDetector):
    """
    Unsafe loading/unloading sequence: several packages in uncontrolled motion
    at the same transfer point simultaneously, which is how sliding and toppling
    incidents begin.
    """

    behaviour_type = BehaviourType.UNSAFE_LOADING_SEQUENCE
    status = ImplementationStatus.PARTIAL
    requirements = "At least three products tracked concurrently in the transfer area."
    limitations = (
        "Detects concurrency and congestion, not adherence to a specific documented "
        "loading plan; verifying a plan needs the plan as an input."
    )

    MIN_CONCURRENT = 3
    MIN_SUSTAIN = 1.5

    def __init__(self, cooldown_sec: float = 10.0) -> None:
        super().__init__(cooldown_sec)
        self._busy_since: Optional[float] = None

    def process(
        self,
        tracks: List[TrackedObject],
        frame_idx: int,
        timestamp: float,
        ctx: Dict[str, Any],
    ) -> List[BehaviourEvent]:
        moving = [
            t
            for t in tracks
            if t.is_product
            and t.hits >= 4
            and t.state in (MotionState.CARRIED, MotionState.SLIDING, MotionState.FALLING)
        ]
        if len(moving) < self.MIN_CONCURRENT:
            self._busy_since = None
            return []

        if self._busy_since is None:
            self._busy_since = timestamp
        sustained = timestamp - self._busy_since
        if sustained < self.MIN_SUSTAIN or not self._cooled_down("scene", timestamp):
            return []

        anchor = max(moving, key=lambda t: t.area if hasattr(t, "area") else t.width * t.height)
        params = {
            "concurrent_items": len(moving),
            "sustained_sec": round(sustained, 2),
            "blocked_walkway": len(moving) >= 4,
            "detection_confidence": float(np.mean([t.confidence for t in moving])),
        }
        assessment = RiskEngine.evaluate(
            self.behaviour_type,
            params,
            product_type=anchor.entity_type.value,
            recurrence_count=ctx["recurrence"].get(self.behaviour_type.value, 0),
            bay=ctx.get("bay"),
        )
        self._mark_fired("scene", timestamp)
        return [
            BehaviourEvent(
                event_id=f"seq_{frame_idx}_{uuid.uuid4().hex[:6]}",
                behaviour_type=self.behaviour_type,
                timestamp_sec=timestamp,
                frame_idx=frame_idx,
                object_track_id=anchor.track_id,
                confidence=float(min(0.78, 0.45 + params["detection_confidence"] * 0.35)),
                risk_level=assessment.level,
                risk_score=assessment.score,
                evidence_description=(
                    f"{len(moving)} packages were in simultaneous motion at this transfer point for "
                    f"{sustained:.1f} s, with no single-item handover sequence observable."
                ),
                root_cause=(
                    "Several packages handled concurrently rather than in a planned one-at-a-time "
                    "loading sequence, increasing collision and topple risk."
                ),
                recommended_action=assessment.recommendation,
                bounding_box=anchor.box,
                risk_factors=assessment.factors_as_dicts(),
                evidence_stages=stages_from_track(anchor),
                evidence_tier=assessment.evidence_tier,
                duration_sec=sustained,
                metadata=params,
            )
        ]


def handling_equipment_present(tracks: List[TrackedObject]) -> bool:
    return any(t.entity_type in HANDLING_EQUIPMENT_ENTITIES and t.hits >= 3 for t in tracks)
