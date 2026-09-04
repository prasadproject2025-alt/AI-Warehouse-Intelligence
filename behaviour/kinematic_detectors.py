"""
Kinematic behaviour detectors: drop, throw, drag, roll.

Every detector here consumes a *sequence* of tracked states, not a single
frame. A drop, for example, is only emitted when the full chain is observed:

    operator contact -> sustained descent -> abrupt deceleration -> at rest

Any missing link means no event. All thresholds are in frame-height units
(see ``detection.tracker``) so they hold across resolutions and zoom levels.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from behaviour.base import (
    BaseBehaviourDetector,
    BehaviourEvent,
    BehaviourType,
    ImplementationStatus,
    stages_from_track,
)
from detection.object_classes import WarehouseEntity
from detection.tracker import MotionState, TrackedObject
from risk.risk_engine import RiskEngine

#: Assumed stature of a standing adult operator, used to convert normalised
#: distances into an approximate metre scale for the risk narrative.
ADULT_HEIGHT_M = 1.70


class SceneScale:
    """Monocular scale estimate derived from observed operator stature."""

    def __init__(self) -> None:
        self._samples: List[float] = []

    def observe(self, tracks: List[TrackedObject]) -> None:
        for t in tracks:
            if t.entity_type is WarehouseEntity.OPERATOR and t.hits > 3:
                h_norm = t.height / t.frame_height
                if 0.1 < h_norm < 0.95:
                    self._samples.append(h_norm)
        if len(self._samples) > 300:
            self._samples = self._samples[-300:]

    @property
    def available(self) -> bool:
        return len(self._samples) >= 10

    def to_metres(self, norm_distance: float) -> Optional[float]:
        """Convert a frame-height fraction into approximate metres."""
        if not self.available:
            return None
        median_person = float(np.median(self._samples))
        if median_person <= 1e-6:
            return None
        return round(norm_distance / median_person * ADULT_HEIGHT_M, 2)


def _product_family(track: TrackedObject) -> str:
    return track.entity_type.value if track.entity_type else "carton"


class DropDetector(BaseBehaviourDetector):
    """
    Product drop.

    Required temporal chain:
      1. the item was in operator contact or elevated above the floor line,
      2. sustained downward motion over consecutive analysis samples,
      3. an abrupt velocity collapse (floor or stack impact),
      4. the item stays put afterwards.
    """

    behaviour_type = BehaviourType.PRODUCT_DROP
    status = ImplementationStatus.IMPLEMENTED
    requirements = "Product must be detected and tracked through the fall (>=4 samples)."
    limitations = (
        "Drop height is a monocular estimate scaled against operator stature; "
        "it is an approximation, not a measurement."
    )

    FALL_VY = 0.30          # frame-heights/s of sustained descent
    IMPACT_VY = 0.12        # velocity considered 'stopped'
    MIN_DESCENT = 0.08      # net normalised descent

    def process(
        self,
        tracks: List[TrackedObject],
        frame_idx: int,
        timestamp: float,
        ctx: Dict[str, Any],
    ) -> List[BehaviourEvent]:
        events: List[BehaviourEvent] = []
        scale: SceneScale = ctx["scale"]
        floor = ctx["floor_line_norm"]

        for trk in tracks:
            if not trk.is_product or trk.hits < 4:
                continue
            if not self._cooled_down(trk.track_id, timestamp):
                continue

            hist = trk.recent(3.0)
            if len(hist) < 5:
                continue

            fall_end = self._find_fall_then_impact(hist)
            if fall_end is None:
                continue
            start_i, impact_i = fall_end

            descent = hist[impact_i]["center"][1] / trk.frame_height - (
                hist[start_i]["center"][1] / trk.frame_height
            )
            if descent < self.MIN_DESCENT:
                continue

            # Must have been handled, or have started clearly off the floor.
            was_handled = any(
                h.get("operator_contact") is not None for h in hist[: start_i + 1]
            )
            started_elevated = hist[start_i]["bottom_y_norm"] < (floor - 0.10)
            if not (was_handled or started_elevated):
                continue

            # Must be at rest now.
            if trk.speed > self.IMPACT_VY:
                continue

            peak = max(h["vy"] for h in hist[start_i : impact_i + 1])
            drop_m = scale.to_metres(descent)
            params = {
                "drop_height_norm": round(descent, 3),
                "drop_height_m": drop_m,
                "peak_fall_speed": round(peak, 3),
                "impact_detected": True,
                "settled_after_impact": trk.state
                in (MotionState.SETTLED, MotionState.STATIONARY),
                "detection_confidence": trk.confidence,
            }
            assessment = RiskEngine.evaluate(
                self.behaviour_type,
                params,
                product_type=_product_family(trk),
                recurrence_count=ctx["recurrence"].get(self.behaviour_type.value, 0),
                bay=ctx.get("bay"),
            )

            height_txt = (
                f"approximately {drop_m:.1f} m"
                if drop_m
                else f"{descent:.2f} frame-heights"
            )
            self._mark_fired(trk.track_id, timestamp)
            events.append(
                BehaviourEvent(
                    event_id=f"drop_{trk.track_id}_{frame_idx}_{uuid.uuid4().hex[:6]}",
                    behaviour_type=self.behaviour_type,
                    timestamp_sec=timestamp,
                    frame_idx=frame_idx,
                    object_track_id=trk.track_id,
                    operator_track_id=hist[start_i].get("operator_contact"),
                    confidence=float(min(0.93, 0.55 + trk.confidence * 0.4)),
                    risk_level=assessment.level,
                    risk_score=assessment.score,
                    evidence_description=(
                        f"{trk.entity_type.value.title()} #{trk.track_id} descended {height_txt} "
                        f"in {hist[impact_i]['time'] - hist[start_i]['time']:.2f} s "
                        f"(peak {peak:.2f} frame-heights/s), decelerated abruptly on contact, "
                        "and remained where it landed."
                    ),
                    root_cause=(
                        "Package released instead of being lowered under control during manual transfer."
                    ),
                    recommended_action=assessment.recommendation,
                    bounding_box=trk.box,
                    risk_factors=assessment.factors_as_dicts(),
                    evidence_stages=stages_from_track(trk),
                    evidence_tier=assessment.evidence_tier,
                    duration_sec=hist[impact_i]["time"] - hist[start_i]["time"],
                    metadata=params,
                )
            )
        return events

    def _find_fall_then_impact(self, hist: List[Dict[str, Any]]):
        """Locate a sustained descent immediately followed by a velocity collapse."""
        i = 0
        n = len(hist)
        while i < n - 2:
            if hist[i]["vy"] > self.FALL_VY:
                j = i
                while j + 1 < n and hist[j + 1]["vy"] > self.FALL_VY * 0.6:
                    j += 1
                # Need at least two consecutive descending samples.
                if j - i >= 1:
                    for k in range(j, min(n, j + 4)):
                        if hist[k]["vy"] < self.IMPACT_VY:
                            return i, k
                i = j + 1
            else:
                i += 1
        return None


class ThrowDetector(BaseBehaviourDetector):
    """
    Product thrown or pushed.

    Distinguished from a drop by a horizontal launch component and a genuine
    unsupported flight phase with no operator in contact range.
    """

    behaviour_type = BehaviourType.PRODUCT_THROW
    status = ImplementationStatus.PARTIAL
    requirements = "Product must remain tracked from release to landing."
    limitations = (
        "Not separable on the current pilot footage. Peak product speed was measured "
        "across all seven clips: the throwing clips peak at 0.40 frame-heights/s (and "
        "one has no product track at all), while clips containing no throw reach 0.52 "
        "and 1.55. The distributions overlap, so no release-speed threshold divides "
        "them - lowering it would add false positives without recovering the true ones. "
        "The cause is upstream: the track breaks during the fast phase of a throw, so "
        "the release is never observed. Needs a detector that holds the product through "
        "rapid motion."
    )

    RELEASE_SPEED = 0.70
    MIN_HORIZONTAL = 0.10

    def process(
        self,
        tracks: List[TrackedObject],
        frame_idx: int,
        timestamp: float,
        ctx: Dict[str, Any],
    ) -> List[BehaviourEvent]:
        events: List[BehaviourEvent] = []

        for trk in tracks:
            if not trk.is_product or trk.hits < 4:
                continue
            if not self._cooled_down(trk.track_id, timestamp):
                continue

            hist = trk.recent(2.0)
            if len(hist) < 4:
                continue

            speeds = [float(np.hypot(h["vx"], h["vy"])) for h in hist]
            peak_i = int(np.argmax(speeds))
            peak = speeds[peak_i]
            if peak < self.RELEASE_SPEED:
                continue

            # Contact before release, no contact at peak flight.
            had_contact = any(h.get("operator_contact") is not None for h in hist[:peak_i + 1])
            free_at_peak = hist[peak_i].get("operator_contact") is None
            if not (had_contact and free_at_peak):
                continue

            dx = abs(hist[-1]["center"][0] - hist[0]["center"][0]) / trk.frame_height
            if dx < self.MIN_HORIZONTAL:
                continue  # vertical-only -> that is a drop, handled elsewhere

            # Ballistic: downward velocity increasing through the flight.
            tail = hist[peak_i:]
            ballistic = len(tail) >= 3 and (tail[-1]["vy"] - tail[0]["vy"]) > 0.10

            landed_on_product = any(
                o.is_product
                and o.track_id != trk.track_id
                and abs(o.box[1] - trk.box[3]) < 0.06 * trk.frame_height
                and min(o.box[2], trk.box[2]) - max(o.box[0], trk.box[0]) > 0
                for o in tracks
            )

            params = {
                "release_speed": round(peak, 3),
                "horizontal_travel_norm": round(dx, 3),
                "ballistic_phase": bool(ballistic),
                "landed_on_product": bool(landed_on_product),
                "detection_confidence": trk.confidence,
            }
            assessment = RiskEngine.evaluate(
                self.behaviour_type,
                params,
                product_type=_product_family(trk),
                recurrence_count=ctx["recurrence"].get(self.behaviour_type.value, 0),
                bay=ctx.get("bay"),
            )

            self._mark_fired(trk.track_id, timestamp)
            events.append(
                BehaviourEvent(
                    event_id=f"throw_{trk.track_id}_{frame_idx}_{uuid.uuid4().hex[:6]}",
                    behaviour_type=self.behaviour_type,
                    timestamp_sec=timestamp,
                    frame_idx=frame_idx,
                    object_track_id=trk.track_id,
                    operator_track_id=hist[0].get("operator_contact"),
                    confidence=float(min(0.90, 0.50 + trk.confidence * 0.4)),
                    risk_level=assessment.level,
                    risk_score=assessment.score,
                    evidence_description=(
                        f"{trk.entity_type.value.title()} #{trk.track_id} left operator contact at "
                        f"{peak:.2f} frame-heights/s and travelled {dx:.2f} frame-heights horizontally "
                        f"with{'' if ballistic else 'out a confirmed'} unsupported flight phase."
                    ),
                    root_cause=(
                        "Package propelled toward its destination instead of being carried and placed."
                    ),
                    recommended_action=assessment.recommendation,
                    bounding_box=trk.box,
                    risk_factors=assessment.factors_as_dicts(),
                    evidence_stages=stages_from_track(trk),
                    evidence_tier=assessment.evidence_tier,
                    duration_sec=hist[-1]["time"] - hist[peak_i]["time"],
                    metadata=params,
                )
            )
        return events


class DragDetector(BaseBehaviourDetector):
    """
    Product dragged along the floor instead of lifted or trolleyed.

    Requires sustained floor-plane sliding with an operator in contact range
    for most of the window - a single fast frame is not a drag.
    """

    behaviour_type = BehaviourType.PRODUCT_DRAG
    status = ImplementationStatus.IMPLEMENTED
    requirements = "Operator and product both tracked; product base within the floor band."
    limitations = "Cannot separate pushing from pulling without pose estimation."

    MIN_DURATION = 1.0
    MIN_DISTANCE = 0.10

    def process(
        self,
        tracks: List[TrackedObject],
        frame_idx: int,
        timestamp: float,
        ctx: Dict[str, Any],
    ) -> List[BehaviourEvent]:
        events: List[BehaviourEvent] = []

        for trk in tracks:
            if not trk.is_product or trk.hits < 5:
                continue
            if not self._cooled_down(trk.track_id, timestamp):
                continue

            hist = trk.recent(4.0)
            if len(hist) < 5:
                continue

            sliding = [h for h in hist if h["state"] == MotionState.SLIDING.value]
            if len(sliding) < 3:
                continue
            duration = sliding[-1]["time"] - sliding[0]["time"]
            if duration < self.MIN_DURATION:
                continue

            dist = abs(sliding[-1]["center"][0] - sliding[0]["center"][0]) / trk.frame_height
            if dist < self.MIN_DISTANCE:
                continue

            contact_frac = sum(
                1 for h in sliding if h.get("operator_contact") is not None
            ) / len(sliding)
            if contact_frac < 0.5:
                continue  # moving on the floor with nobody near it is not a drag

            params = {
                "drag_distance_norm": round(dist, 3),
                "duration_sec": round(duration, 2),
                "on_wet_floor": bool(ctx.get("wet_floor_active")),
                "handling_equipment_present": ctx.get("handling_equipment_present"),
                "detection_confidence": trk.confidence,
            }
            assessment = RiskEngine.evaluate(
                self.behaviour_type,
                params,
                product_type=_product_family(trk),
                recurrence_count=ctx["recurrence"].get(self.behaviour_type.value, 0),
                bay=ctx.get("bay"),
            )

            self._mark_fired(trk.track_id, timestamp)
            events.append(
                BehaviourEvent(
                    event_id=f"drag_{trk.track_id}_{frame_idx}_{uuid.uuid4().hex[:6]}",
                    behaviour_type=self.behaviour_type,
                    timestamp_sec=timestamp,
                    frame_idx=frame_idx,
                    object_track_id=trk.track_id,
                    operator_track_id=sliding[-1].get("operator_contact"),
                    confidence=float(min(0.90, 0.55 + trk.confidence * 0.35)),
                    risk_level=assessment.level,
                    risk_score=assessment.score,
                    evidence_description=(
                        f"{trk.entity_type.value.title()} #{trk.track_id} slid along the floor for "
                        f"{duration:.1f} s covering {dist:.2f} frame-heights with an operator in "
                        "contact range throughout and no lifting phase observed."
                    ),
                    root_cause=(
                        "Package moved by dragging over the floor surface rather than on a trolley, "
                        "pallet truck or by team lift."
                    ),
                    recommended_action=assessment.recommendation,
                    bounding_box=trk.box,
                    risk_factors=assessment.factors_as_dicts(),
                    evidence_stages=stages_from_track(trk),
                    evidence_tier=assessment.evidence_tier,
                    duration_sec=duration,
                    metadata=params,
                )
            )
        return events


class RollDetector(BaseBehaviourDetector):
    """
    Product rolled or tumbled end-over-end along the floor.

    Detected from cyclical inversion of the bounding-box aspect ratio while the
    item translates with its base on the floor.
    """

    behaviour_type = BehaviourType.ROLLING_PRODUCT
    status = ImplementationStatus.PARTIAL
    requirements = "Product tracked continuously through at least two full inversions."
    limitations = (
        "Aspect-ratio inversion is a proxy for rotation, and on the pilot footage it "
        "produced zero inversions across every clip, including both rolling clips - so "
        "this detector currently recovers nothing. Two reasons: axis-symmetric items "
        "(rolled mattresses, drums) rotate without changing their aspect ratio at all, "
        "and an axis-aligned bounding box around a tumbling carton changes far less "
        "than the carton itself does. A rotated-box or segmentation model is required; "
        "no threshold on this proxy can fix it."
    )

    MIN_CYCLES = 2

    def process(
        self,
        tracks: List[TrackedObject],
        frame_idx: int,
        timestamp: float,
        ctx: Dict[str, Any],
    ) -> List[BehaviourEvent]:
        events: List[BehaviourEvent] = []
        floor = ctx["floor_line_norm"]

        for trk in tracks:
            if not trk.is_product or trk.hits < 8:
                continue
            if not self._cooled_down(trk.track_id, timestamp):
                continue

            hist = trk.recent(3.5)
            if len(hist) < 8:
                continue

            ratios = [h["aspect_ratio"] for h in hist]
            cycles, above = 0, ratios[0] > 1.0
            for r in ratios[1:]:
                if above and r < 0.85:
                    cycles += 1
                    above = False
                elif not above and r > 1.18:
                    cycles += 1
                    above = True
            if cycles < self.MIN_CYCLES:
                continue

            dx = abs(hist[-1]["center"][0] - hist[0]["center"][0]) / trk.frame_height
            if dx < 0.06:
                continue
            on_floor = np.mean([h["bottom_y_norm"] for h in hist]) > (floor - 0.10)
            if not on_floor:
                continue

            params = {
                "inversion_cycles": cycles,
                "roll_distance_norm": round(dx, 3),
                "detection_confidence": trk.confidence,
            }
            assessment = RiskEngine.evaluate(
                self.behaviour_type,
                params,
                product_type=_product_family(trk),
                recurrence_count=ctx["recurrence"].get(self.behaviour_type.value, 0),
                bay=ctx.get("bay"),
            )

            self._mark_fired(trk.track_id, timestamp)
            events.append(
                BehaviourEvent(
                    event_id=f"roll_{trk.track_id}_{frame_idx}_{uuid.uuid4().hex[:6]}",
                    behaviour_type=self.behaviour_type,
                    timestamp_sec=timestamp,
                    frame_idx=frame_idx,
                    object_track_id=trk.track_id,
                    confidence=float(min(0.82, 0.45 + trk.confidence * 0.35)),
                    risk_level=assessment.level,
                    risk_score=assessment.score,
                    evidence_description=(
                        f"{trk.entity_type.value.title()} #{trk.track_id} showed {cycles} end-over-end "
                        f"inversions while translating {dx:.2f} frame-heights along the floor."
                    ),
                    root_cause=(
                        "Package rotated over its edges across the floor instead of being carried "
                        "or moved on handling equipment."
                    ),
                    recommended_action=assessment.recommendation,
                    bounding_box=trk.box,
                    risk_factors=assessment.factors_as_dicts(),
                    evidence_stages=stages_from_track(trk),
                    evidence_tier=assessment.evidence_tier,
                    duration_sec=hist[-1]["time"] - hist[0]["time"],
                    metadata=params,
                )
            )
        return events
