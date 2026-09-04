"""Risk engine: determinism, transparency, ordering and threshold behaviour."""

from behaviour.base import BehaviourType, EvidenceTier, RiskLevel
from risk.risk_engine import RiskEngine


def test_scoring_is_deterministic():
    params = {"drop_height_m": 1.2, "peak_fall_speed": 1.1, "impact_detected": True}
    runs = {RiskEngine.evaluate(BehaviourType.PRODUCT_DROP, params).score for _ in range(20)}
    assert len(runs) == 1, "risk scores must not vary between identical evaluations"


def test_score_equals_sum_of_published_factors():
    """Every point in the score must be attributable to a named factor."""
    a = RiskEngine.evaluate(
        BehaviourType.PRODUCT_DRAG,
        {"drag_distance_norm": 0.4, "duration_sec": 4.0, "detection_confidence": 0.9},
    )
    assert abs(sum(f.points for f in a.factors) - a.score) < 0.05
    assert all(f.name and f.detail for f in a.factors)


def test_score_is_the_clamped_sum_of_factors():
    """When factors exceed the ceiling, the score clamps rather than drifting."""
    a = RiskEngine.evaluate(
        BehaviourType.PRODUCT_DROP,
        {"drop_height_m": 1.4, "peak_fall_speed": 1.2, "impact_detected": True,
         "settled_after_impact": True, "detection_confidence": 0.9},
    )
    expected = max(5.0, min(98.0, sum(f.points for f in a.factors)))
    assert abs(a.score - expected) < 0.05


def test_higher_drop_scores_higher():
    low = RiskEngine.evaluate(BehaviourType.PRODUCT_DROP, {"drop_height_m": 0.3})
    high = RiskEngine.evaluate(BehaviourType.PRODUCT_DROP, {"drop_height_m": 1.5})
    assert high.score > low.score


def test_wet_floor_raises_drag_risk():
    dry = RiskEngine.evaluate(BehaviourType.PRODUCT_DRAG, {"drag_distance_norm": 0.4})
    wet = RiskEngine.evaluate(
        BehaviourType.PRODUCT_DRAG, {"drag_distance_norm": 0.4, "on_wet_floor": True}
    )
    assert wet.score > dry.score
    assert any("wet" in f.name.lower() for f in wet.factors)


def test_recurrence_increases_score():
    once = RiskEngine.evaluate(BehaviourType.PRODUCT_DRAG, {}, recurrence_count=0)
    often = RiskEngine.evaluate(BehaviourType.PRODUCT_DRAG, {}, recurrence_count=12)
    assert often.score > once.score
    assert any("Recurring" in f.name for f in often.factors)


def test_low_detection_confidence_discounts_score():
    confident = RiskEngine.evaluate(
        BehaviourType.PRODUCT_THROW, {"release_speed": 1.6, "detection_confidence": 0.9}
    )
    unsure = RiskEngine.evaluate(
        BehaviourType.PRODUCT_THROW, {"release_speed": 1.6, "detection_confidence": 0.3}
    )
    assert unsure.score < confident.score
    assert any(f.points < 0 for f in unsure.factors)


def test_levels_follow_documented_thresholds():
    for score, expected in [(95, RiskLevel.CRITICAL), (70, RiskLevel.HIGH),
                            (50, RiskLevel.MEDIUM), (20, RiskLevel.LOW)]:
        if score >= RiskEngine.THRESHOLDS["CRITICAL"]:
            assert expected is RiskLevel.CRITICAL
        elif score >= RiskEngine.THRESHOLDS["HIGH"]:
            assert expected is RiskLevel.HIGH
        elif score >= RiskEngine.THRESHOLDS["MEDIUM"]:
            assert expected is RiskLevel.MEDIUM
        else:
            assert expected is RiskLevel.LOW


def test_score_is_bounded():
    extreme = RiskEngine.evaluate(
        BehaviourType.STEPPING_ON_CARTON,
        {"elevation_above_floor_norm": 9.9, "dwell_sec": 900.0},
        recurrence_count=9999,
    )
    assert 5.0 <= extreme.score <= 98.0


def test_never_asserts_confirmed_damage():
    """The pipeline may report potential risk; only a human can confirm damage."""
    for behaviour in BehaviourType:
        a = RiskEngine.evaluate(behaviour, {"drop_height_m": 3.0}, recurrence_count=50)
        assert a.evidence_tier is not EvidenceTier.CONFIRMED_DAMAGE


def test_every_behaviour_has_an_actionable_recommendation():
    for behaviour in BehaviourType:
        a = RiskEngine.evaluate(behaviour, {})
        assert a.recommendation.startswith("[")
        assert len(a.recommendation) > 40
        # Recommendations must be about the process, not about blaming a person.
        assert "punish" not in a.recommendation.lower()
        assert "discipline" not in a.recommendation.lower()


def test_legacy_tuple_api_still_works():
    level, score, rec = RiskEngine.evaluate_risk(
        BehaviourType.PRODUCT_DROP, {"drop_height_m": 1.2}
    )
    assert isinstance(level, RiskLevel) and 0 < score <= 98 and rec
