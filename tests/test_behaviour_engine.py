import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from behaviour.base import BehaviourType, RiskLevel
from behaviour.behaviour_engine import BehaviourEngine
from risk.risk_engine import RiskEngine
from detection.tracker import TrackedObject
from detection.detector import Detection
from detection.object_classes import WarehouseEntity

def test_risk_engine():
    # Test product drop risk calculation
    level, score, rec = RiskEngine.evaluate_risk(
        BehaviourType.PRODUCT_DROP,
        {"downward_velocity": 180.0, "drop_height_px": 160.0, "stationary_after_impact": True}
    )
    assert level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
    assert score >= 80.0
    assert "inspection" in rec.lower()
    print("Risk Engine Test Passed:", level, score, rec[:50])

def test_behaviour_engine_instantiation():
    engine = BehaviourEngine(is_wet_floor=True, is_dock_scene=True)
    assert engine.drop_detector is not None
    assert engine.drag_detector is not None
    assert engine.throw_detector is not None
    assert engine.roll_detector is not None
    assert engine.stacking_detector is not None
    assert engine.step_detector is not None
    assert engine.strap_detector is not None
    assert engine.wet_floor_detector is not None
    assert engine.orientation_detector is not None
    assert engine.dock_detector is not None
    print("Behaviour Engine Instantiation Test Passed!")

if __name__ == "__main__":
    test_risk_engine()
    test_behaviour_engine_instantiation()
