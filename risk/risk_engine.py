"""
Transparent Multi-Factor Risk Scoring Engine.

Risk is never random and never a bare constant: every score is the sum of a
documented base weight plus named, signed contributions derived from measured
physical quantities, scene context and historical recurrence. The full
breakdown is persisted with the incident and rendered in the UI, so a
supervisor can audit exactly why an event was rated the way it was.

Responsible-AI note: the output describes a *potential* damage risk from an
observed behaviour. Confirming damage requires physical inspection, which is
represented by the evidence tier, not by the score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from behaviour.base import BehaviourType, EvidenceTier, RiskLevel


@dataclass
class RiskFactor:
    """One named, signed contribution to the final score."""

    name: str
    points: float
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "points": round(self.points, 1), "detail": self.detail}


@dataclass
class RiskAssessment:
    level: RiskLevel
    score: float
    factors: List[RiskFactor] = field(default_factory=list)
    recommendation: str = ""
    evidence_tier: EvidenceTier = EvidenceTier.OBSERVED_BEHAVIOUR

    def factors_as_dicts(self) -> List[Dict[str, Any]]:
        return [f.to_dict() for f in self.factors]

    def explanation(self) -> str:
        parts = [f"{f.name} ({f.points:+.0f})" for f in self.factors]
        return " | ".join(parts)


class RiskEngine:
    """Deterministic, explainable risk scoring for detected handling behaviours."""

    #: Base severity of the behaviour class itself, before any measurement.
    BASE_SCORES: Dict[BehaviourType, float] = {
        BehaviourType.PRODUCT_DROP: 62.0,
        BehaviourType.PRODUCT_DRAG: 40.0,
        BehaviourType.PRODUCT_THROW: 66.0,
        BehaviourType.ROLLING_PRODUCT: 48.0,
        BehaviourType.IMPROPER_STACKING: 55.0,
        BehaviourType.STEPPING_ON_CARTON: 70.0,
        BehaviourType.UNSUPPORTED_HANDLING: 45.0,
        BehaviourType.WET_FLOOR_HAZARD: 58.0,
        BehaviourType.ORIENTATION_VIOLATION: 42.0,
        BehaviourType.DOCK_LEVEL_HAZARD: 60.0,
        BehaviourType.OUTSIDE_DESIGNATED_AREA: 38.0,
        BehaviourType.UNSAFE_LOADING_SEQUENCE: 52.0,
    }

    #: Fragility multiplier applied by product family.
    PRODUCT_SENSITIVITY: Dict[str, Tuple[float, str]] = {
        "cupboard": (1.15, "knock-down furniture package (panel/hinge sensitive)"),
        "mattress": (0.90, "soft goods (compression tolerant, soiling sensitive)"),
        "carton": (1.00, "standard corrugated carton"),
    }

    THRESHOLDS = {"CRITICAL": 82.0, "HIGH": 64.0, "MEDIUM": 42.0}

    # ------------------------------------------------------------------ score
    @classmethod
    def evaluate(
        cls,
        behaviour_type: BehaviourType,
        physical_params: Dict[str, Any],
        product_type: str = "carton",
        recurrence_count: int = 0,
        bay: Optional[str] = None,
    ) -> RiskAssessment:
        base = cls.BASE_SCORES.get(behaviour_type, 45.0)
        factors: List[RiskFactor] = [
            RiskFactor(
                "Behaviour class baseline",
                base,
                f"{behaviour_type.value.replace('_', ' ')} carries an inherent handling-damage risk",
            )
        ]

        factors.extend(cls._physical_factors(behaviour_type, physical_params))

        # Product fragility.
        mult, sens_detail = cls.PRODUCT_SENSITIVITY.get(product_type, (1.0, "unclassified product"))
        if abs(mult - 1.0) > 1e-6:
            running = sum(f.points for f in factors)
            delta = running * (mult - 1.0)
            factors.append(
                RiskFactor("Product sensitivity", delta, f"Item classified as {sens_detail}")
            )

        # Recurrence: a repeated behaviour is a process problem, not a one-off.
        if recurrence_count >= 10:
            factors.append(
                RiskFactor(
                    "Recurring behaviour",
                    10.0,
                    f"{recurrence_count} prior occurrences recorded - systemic process gap",
                )
            )
        elif recurrence_count >= 4:
            factors.append(
                RiskFactor(
                    "Recurring behaviour",
                    5.0,
                    f"{recurrence_count} prior occurrences recorded - emerging pattern",
                )
            )

        # Confidence discount: weak perception evidence must not yield a
        # confident high-risk rating.
        conf = float(physical_params.get("detection_confidence", 0.8))
        if conf < 0.45:
            factors.append(
                RiskFactor(
                    "Low perception confidence",
                    -12.0,
                    f"Detection confidence {conf:.2f}; flagged for human confirmation",
                )
            )
        elif conf < 0.6:
            factors.append(
                RiskFactor(
                    "Moderate perception confidence",
                    -6.0,
                    f"Detection confidence {conf:.2f}",
                )
            )

        score = max(5.0, min(98.0, sum(f.points for f in factors)))

        if score >= cls.THRESHOLDS["CRITICAL"]:
            level = RiskLevel.CRITICAL
        elif score >= cls.THRESHOLDS["HIGH"]:
            level = RiskLevel.HIGH
        elif score >= cls.THRESHOLDS["MEDIUM"]:
            level = RiskLevel.MEDIUM
        else:
            level = RiskLevel.LOW

        tier = (
            EvidenceTier.POTENTIAL_RISK
            if level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
            else EvidenceTier.OBSERVED_BEHAVIOUR
        )

        return RiskAssessment(
            level=level,
            score=round(score, 1),
            factors=factors,
            recommendation=cls.recommendation(behaviour_type, level),
            evidence_tier=tier,
        )

    # Backwards-compatible tuple API used by older callers and tests.
    @classmethod
    def evaluate_risk(
        cls, behaviour_type: BehaviourType, physical_params: Dict[str, Any]
    ) -> Tuple[RiskLevel, float, str]:
        a = cls.evaluate(behaviour_type, physical_params)
        return a.level, a.score, a.recommendation

    # ------------------------------------------------------------- physical
    @classmethod
    def _physical_factors(
        cls, behaviour: BehaviourType, p: Dict[str, Any]
    ) -> List[RiskFactor]:
        """
        Convert measured quantities into signed points.

        All spatial quantities arrive normalised to frame height, so an
        estimate of drop height is expressed as a fraction of the operator's
        own height where one is available - which is what makes the "~1 m"
        style wording defensible rather than a pixel guess.
        """
        out: List[RiskFactor] = []

        if behaviour is BehaviourType.PRODUCT_DROP:
            metres = p.get("drop_height_m")
            if metres is not None:
                if metres >= 1.0:
                    out.append(
                        RiskFactor(
                            "Drop height",
                            18.0,
                            f"Fall of approximately {metres:.1f} m (estimated against operator stature)",
                        )
                    )
                elif metres >= 0.5:
                    out.append(
                        RiskFactor("Drop height", 9.0, f"Fall of approximately {metres:.1f} m")
                    )
                else:
                    out.append(
                        RiskFactor("Drop height", 2.0, f"Short fall of approximately {metres:.1f} m")
                    )
            peak = float(p.get("peak_fall_speed", 0.0))
            if peak > 0.9:
                out.append(
                    RiskFactor(
                        "Impact velocity",
                        10.0,
                        f"Peak descent {peak:.2f} frame-heights/s before floor contact",
                    )
                )
            if p.get("impact_detected"):
                out.append(
                    RiskFactor(
                        "Abrupt deceleration",
                        8.0,
                        "Velocity collapsed within one analysis interval, consistent with hard floor impact",
                    )
                )
            if p.get("settled_after_impact"):
                out.append(
                    RiskFactor(
                        "Uncontrolled landing",
                        4.0,
                        "Product came to rest where it landed rather than being placed",
                    )
                )

        elif behaviour is BehaviourType.PRODUCT_DRAG:
            dist = float(p.get("drag_distance_norm", 0.0))
            dur = float(p.get("duration_sec", 0.0))
            if dist > 0.35:
                out.append(
                    RiskFactor("Drag distance", 14.0, f"Dragged {dist:.2f} frame-heights across the floor")
                )
            elif dist > 0.15:
                out.append(RiskFactor("Drag distance", 7.0, f"Dragged {dist:.2f} frame-heights"))
            if dur > 3.0:
                out.append(
                    RiskFactor("Sustained duration", 6.0, f"Abrasive floor contact sustained for {dur:.1f} s")
                )
            if p.get("on_wet_floor"):
                out.append(
                    RiskFactor("Wet floor contact", 16.0, "Base dragged through a detected wet-floor zone")
                )
            if p.get("handling_equipment_present") is False:
                out.append(
                    RiskFactor(
                        "No handling equipment",
                        8.0,
                        "No trolley or pallet truck visible in the working area during the move",
                    )
                )

        elif behaviour is BehaviourType.PRODUCT_THROW:
            speed = float(p.get("release_speed", 0.0))
            if speed > 1.4:
                out.append(RiskFactor("Release velocity", 16.0, f"Released at {speed:.2f} frame-heights/s"))
            elif speed > 0.9:
                out.append(RiskFactor("Release velocity", 8.0, f"Released at {speed:.2f} frame-heights/s"))
            if p.get("ballistic_phase"):
                out.append(
                    RiskFactor(
                        "Unsupported flight",
                        10.0,
                        "Product accelerated downward with no operator contact - free flight confirmed",
                    )
                )
            if p.get("landed_on_product"):
                out.append(
                    RiskFactor("Landed on other goods", 8.0, "Impact absorbed by other stacked product")
                )

        elif behaviour is BehaviourType.ROLLING_PRODUCT:
            cycles = int(p.get("inversion_cycles", 0))
            if cycles >= 3:
                out.append(RiskFactor("Repeated tumbling", 12.0, f"{cycles} end-over-end inversions"))
            elif cycles >= 2:
                out.append(RiskFactor("Tumbling", 6.0, f"{cycles} end-over-end inversions"))
            if float(p.get("roll_distance_norm", 0.0)) > 0.3:
                out.append(
                    RiskFactor("Roll distance", 6.0, "Rolled a substantial distance over bare floor")
                )

        elif behaviour is BehaviourType.IMPROPER_STACKING:
            ratio = float(p.get("width_ratio", 1.0))
            if p.get("heavy_on_light"):
                out.append(
                    RiskFactor(
                        "Heavy item on lighter base",
                        18.0,
                        "Rigid/heavy package resting on a lighter carton",
                    )
                )
            if ratio > 1.5:
                out.append(
                    RiskFactor(
                        "Severe overhang",
                        14.0,
                        f"Upper item is {ratio:.2f}x the width of its base - unsupported span",
                    )
                )
            elif ratio > 1.25:
                out.append(
                    RiskFactor("Overhang", 7.0, f"Upper item is {ratio:.2f}x the width of its base")
                )
            if float(p.get("stable_seconds", 0.0)) >= 2.0:
                out.append(
                    RiskFactor(
                        "Persistent configuration",
                        5.0,
                        f"Unsafe stack held for {p['stable_seconds']:.1f} s (not a transient hand-over)",
                    )
                )

        elif behaviour is BehaviourType.STEPPING_ON_CARTON:
            elev = float(p.get("elevation_above_floor_norm", 0.0))
            if elev > 0.12:
                out.append(
                    RiskFactor(
                        "Operator elevated on load",
                        12.0,
                        f"Feet {elev:.2f} frame-heights above the estimated floor line",
                    )
                )
            if float(p.get("dwell_sec", 0.0)) > 1.0:
                out.append(
                    RiskFactor(
                        "Sustained standing",
                        8.0,
                        f"Weight borne by packaging for {p['dwell_sec']:.1f} s",
                    )
                )

        elif behaviour is BehaviourType.UNSUPPORTED_HANDLING:
            if p.get("handling_equipment_present") is False:
                out.append(
                    RiskFactor(
                        "No handling equipment in use",
                        10.0,
                        "Heavy item moved manually with no trolley or pallet truck detected",
                    )
                )
            if float(p.get("item_size_norm", 0.0)) > 0.35:
                out.append(
                    RiskFactor(
                        "Oversized for single-person lift",
                        10.0,
                        "Item exceeds a safe one-person manual-handling envelope",
                    )
                )

        elif behaviour is BehaviourType.WET_FLOOR_HAZARD:
            cov = float(p.get("wet_zone_coverage", 0.0))
            if cov > 0.25:
                out.append(
                    RiskFactor(
                        "Large wet area",
                        12.0,
                        f"{cov * 100:.0f}% of the working floor shows standing moisture",
                    )
                )
            elif cov > 0.08:
                out.append(
                    RiskFactor("Wet area", 6.0, f"{cov * 100:.0f}% of the working floor shows moisture")
                )
            if p.get("product_in_zone"):
                out.append(
                    RiskFactor(
                        "Goods in wet zone",
                        10.0,
                        "Product base in contact with the wet area - moisture ingress risk",
                    )
                )

        elif behaviour is BehaviourType.ORIENTATION_VIOLATION:
            if p.get("observed_transition"):
                out.append(
                    RiskFactor(
                        "Observed upright-to-flat transition",
                        16.0,
                        "The same tracked item was seen upright and then laid flat",
                    )
                )
            if float(p.get("flat_seconds", 0.0)) > 3.0:
                out.append(
                    RiskFactor(
                        "Sustained incorrect orientation",
                        6.0,
                        f"Held flat for {p['flat_seconds']:.1f} s",
                    )
                )

        elif behaviour is BehaviourType.DOCK_LEVEL_HAZARD:
            if p.get("vehicle_present"):
                out.append(
                    RiskFactor("Vehicle at dock", 8.0, "Transfer occurring across a vehicle threshold")
                )
            if p.get("no_leveller_detected"):
                out.append(
                    RiskFactor(
                        "No dock leveller detected",
                        10.0,
                        "No bridge plate or leveller visible at the transition point",
                    )
                )

        elif behaviour is BehaviourType.OUTSIDE_DESIGNATED_AREA:
            out.append(
                RiskFactor(
                    "Outside configured staging zone",
                    10.0,
                    "Product at rest beyond the supervisor-defined staging polygon",
                )
            )

        elif behaviour is BehaviourType.UNSAFE_LOADING_SEQUENCE:
            n = int(p.get("concurrent_items", 0))
            if n >= 3:
                out.append(
                    RiskFactor(
                        "Uncontrolled concurrent handling",
                        12.0,
                        f"{n} items in motion simultaneously in one transfer point",
                    )
                )
            if p.get("blocked_walkway"):
                out.append(
                    RiskFactor("Obstructed path", 8.0, "Staged goods obstructing the transfer path")
                )

        return out

    # ------------------------------------------------------------ prescribe
    RECOMMENDATIONS: Dict[BehaviourType, str] = {
        BehaviourType.PRODUCT_DROP: (
            "Quarantine and physically inspect the item and its internal contents before dispatch. "
            "Re-brief the operator on controlled lowering and two-person lifting for heavy packages."
        ),
        BehaviourType.PRODUCT_DRAG: (
            "Provide a trolley or hand pallet truck at this transfer point and stop floor dragging. "
            "Inspect the carton base for abrasion and seam damage."
        ),
        BehaviourType.PRODUCT_THROW: (
            "Intervene now and stop throwing. Enforce lift-and-place handling; "
            "inspect both the thrown item and whatever it landed on."
        ),
        BehaviourType.ROLLING_PRODUCT: (
            "Supply a hand truck or arrange a team lift. Rolling loads the package corners and edges "
            "they were not designed to carry - inspect corner integrity."
        ),
        BehaviourType.IMPROPER_STACKING: (
            "Restack now with the heaviest and largest packages at the base and full perimeter support. "
            "Inspect the lower cartons for crushing before they move on."
        ),
        BehaviourType.STEPPING_ON_CARTON: (
            "Stop the practice immediately - this is both a product-damage and a fall hazard. "
            "Clear a walkway and provide a step platform where height access is genuinely needed."
        ),
        BehaviourType.UNSUPPORTED_HANDLING: (
            "Move this item with the correct equipment or a team lift. "
            "Review whether handling equipment is actually available at this bay."
        ),
        BehaviourType.WET_FLOOR_HAZARD: (
            "Pause material movement across this area, dry the floor, then resume. "
            "Inspect the base of any package that crossed the wet zone for moisture ingress."
        ),
        BehaviourType.ORIENTATION_VIOLATION: (
            "Return the package to its marked upright orientation and confirm handling arrows are visible. "
            "Check for panel deflection before loading."
        ),
        BehaviourType.DOCK_LEVEL_HAZARD: (
            "Fit a dock leveller or bridge plate before continuing this transfer, "
            "and verify the vehicle bed height matches the dock."
        ),
        BehaviourType.OUTSIDE_DESIGNATED_AREA: (
            "Return the product to the designated staging area and confirm the zone markings are visible "
            "and adequate for the current volume."
        ),
        BehaviourType.UNSAFE_LOADING_SEQUENCE: (
            "Sequence the transfer one item at a time to a planned loading order and keep the transfer "
            "path clear."
        ),
    }

    @classmethod
    def recommendation(cls, behaviour_type: BehaviourType, risk_level: RiskLevel) -> str:
        body = cls.RECOMMENDATIONS.get(
            behaviour_type,
            "Review the handling step against the site SOP with the shift supervisor.",
        )
        return f"[{risk_level.value} PRIORITY] {body}"

    # Legacy alias retained for older imports.
    @classmethod
    def _get_recommendation(
        cls, behaviour_type: BehaviourType, risk_level: RiskLevel, modifiers: Optional[list] = None
    ) -> str:
        return cls.recommendation(behaviour_type, risk_level)
