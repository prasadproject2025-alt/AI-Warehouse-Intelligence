"""
Transparent Multi-Factor Risk Scoring Engine
Calculates risk levels (LOW, MEDIUM, HIGH, CRITICAL) and actionable operational recommendations
based on physical factors: drop height, velocity, impact deceleration, object fragility, 
duration, and spatial proximity.
"""

from typing import Dict, Any, Tuple
from behaviour.base import BehaviourType, RiskLevel

class RiskEngine:
    """
    Computes deterministic, transparent risk scores for detected warehouse behaviours.
    Distinguishes: Observed Behaviour -> Potential Risk -> Confirmed Damage
    """
    
    # Base risk scores per behaviour (0 - 100)
    BASE_SCORES: Dict[BehaviourType, float] = {
        BehaviourType.PRODUCT_DROP: 75.0,
        BehaviourType.PRODUCT_DRAG: 45.0,
        BehaviourType.PRODUCT_THROW: 80.0,
        BehaviourType.ROLLING_PRODUCT: 60.0,
        BehaviourType.IMPROPER_STACKING: 70.0,
        BehaviourType.STEPPING_ON_CARTON: 85.0,
        BehaviourType.STRAP_PULLING: 65.0,
        BehaviourType.WET_FLOOR_HAZARD: 75.0,
        BehaviourType.ORIENTATION_VIOLATION: 55.0,
        BehaviourType.DOCK_LEVEL_HAZARD: 80.0,
    }

    @classmethod
    def evaluate_risk(
        cls,
        behaviour_type: BehaviourType,
        physical_params: Dict[str, Any]
    ) -> Tuple[RiskLevel, float, str]:
        """
        Evaluate risk level, numeric score (0-100), and prescriptive recommendation.
        """
        base = cls.BASE_SCORES.get(behaviour_type, 50.0)
        score = base
        modifiers = []

        # Multi-factor adjustments
        if behaviour_type == BehaviourType.PRODUCT_DROP:
            # Impact velocity / downward speed factor
            vy = physical_params.get("downward_velocity", 0.0)
            drop_height_px = physical_params.get("drop_height_px", 0.0)
            if drop_height_px > 150: # > ~1 meter equivalent
                score += 15
                modifiers.append("severe drop height (>1m approx)")
            elif drop_height_px > 70:
                score += 5
                modifiers.append("moderate drop height")
                
            if physical_params.get("stationary_after_impact", False):
                score += 5
                modifiers.append("high impact floor landing")

        elif behaviour_type == BehaviourType.PRODUCT_DRAG:
            drag_distance = physical_params.get("drag_distance_px", 0.0)
            on_wet_floor = physical_params.get("on_wet_floor", False)
            if drag_distance > 200:
                score += 15
                modifiers.append("extended dragging distance along concrete floor")
            if on_wet_floor:
                score += 20
                modifiers.append("wet floor contact creating moisture damage risk")

        elif behaviour_type == BehaviourType.PRODUCT_THROW:
            release_speed = physical_params.get("release_velocity", 0.0)
            if release_speed > 250:
                score += 15
                modifiers.append("high velocity trajectory")

        elif behaviour_type == BehaviourType.STEPPING_ON_CARTON:
            score += 10 # Direct human weight on cardboard
            modifiers.append("concentrated operator body weight exceeding carton crush limit")

        elif behaviour_type == BehaviourType.IMPROPER_STACKING:
            heavy_on_light = physical_params.get("heavy_on_light", False)
            if heavy_on_light:
                score += 20
                modifiers.append("heavy rigid product placed on top of lighter/smaller packets")

        elif behaviour_type == BehaviourType.DOCK_LEVEL_HAZARD:
            score += 10
            modifiers.append("transition across vehicle/dock gap without dock leveller")

        score = max(5.0, min(98.0, score))

        # Categorize into RiskLevel
        if score >= 85.0:
            level = RiskLevel.CRITICAL
        elif score >= 65.0:
            level = RiskLevel.HIGH
        elif score >= 40.0:
            level = RiskLevel.MEDIUM
        else:
            level = RiskLevel.LOW

        recommendation = cls._get_recommendation(behaviour_type, level, modifiers)
        return level, score, recommendation

    @classmethod
    def _get_recommendation(
        cls,
        behaviour_type: BehaviourType,
        risk_level: RiskLevel,
        modifiers: list
    ) -> str:
        prefix = f"[{risk_level.value} PRIORITY] "
        if behaviour_type == BehaviourType.PRODUCT_DROP:
            return prefix + "Halt handling of this item. Conduct immediate physical inspection of carton and internal goods for structural/aesthetic damage. Review manual handling technique with operator."
        elif behaviour_type == BehaviourType.PRODUCT_DRAG:
            return prefix + "Mandate immediate use of hydraulic trolley or pallet truck. Stop dragging cartons across the floor to prevent carton bottom abrasion and KD seam tears."
        elif behaviour_type == BehaviourType.PRODUCT_THROW:
            return prefix + "Intervene immediately: Enforce two-handed lift-and-place protocol. Products and mattresses must never be pitched or thrown into vehicle/staging bays."
        elif behaviour_type == BehaviourType.ROLLING_PRODUCT:
            return prefix + "Deploy hand truck or team lift. Do not roll cartons or mattresses as rotational impact crushes corner edges and weakens structural rigidity."
        elif behaviour_type == BehaviourType.IMPROPER_STACKING:
            return prefix + "Restack staging area immediately: Place heaviest and largest cartons at the base. Ensure complete bottom perimeter support for all upper lighter packages."
        elif behaviour_type == BehaviourType.STEPPING_ON_CARTON:
            return prefix + "Safety violation: Strictly prohibit standing, stepping, or walking on merchandise. Ensure clear walkway access around pallets and loading bays."
        elif behaviour_type == BehaviourType.STRAP_PULLING:
            return prefix + "Use base lifting points or mechanical aids. Packaging straps are designed for carton closure only and will snap or tear carton walls when used as lifting handles."
        elif behaviour_type == BehaviourType.WET_FLOOR_HAZARD:
            return prefix + "Stop material movement immediately. Divert traffic, dry the loading dock floor, and inspect goods for moisture ingress before resuming operations."
        elif behaviour_type == BehaviourType.ORIENTATION_VIOLATION:
            return prefix + "Restore vertical orientation immediately according to product packaging arrows to prevent internal component sagging or glass/panel stress."
        elif behaviour_type == BehaviourType.DOCK_LEVEL_HAZARD:
            return prefix + "Deploy proper dock leveller or dock plate before transferring goods between vehicle bed and warehouse dock to eliminate transition shock."
        return prefix + "Follow standard Godrej warehouse handling SOP and review safe handling checklist with shift supervisor."
