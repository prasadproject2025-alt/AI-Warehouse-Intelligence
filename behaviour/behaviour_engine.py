"""
Master Behaviour Intelligence Engine
Orchestrates the 10 modular behaviour detectors, manages temporal reasoning,
debounces recurring alerts, and assigns validated risk levels.
"""

from typing import List, Dict, Any, Optional
from detection.tracker import TrackedObject
from behaviour.base import BehaviourEvent, BehaviourType
from behaviour.drop_detector import DropDetector
from behaviour.drag_detector import DragDetector
from behaviour.throw_detector import ThrowDetector
from behaviour.roll_detector import RollDetector
from behaviour.stacking_detector import StackingDetector
from behaviour.step_detector import StepDetector
from behaviour.strap_detector import StrapDetector
from behaviour.wet_floor_detector import WetFloorDetector
from behaviour.orientation_detector import OrientationDetector
from behaviour.dock_detector import DockDetector

class BehaviourEngine:
    def __init__(self, is_wet_floor: bool = False, is_dock_scene: bool = False):
        self.drop_detector = DropDetector()
        self.drag_detector = DragDetector()
        self.throw_detector = ThrowDetector()
        self.roll_detector = RollDetector()
        self.stacking_detector = StackingDetector()
        self.step_detector = StepDetector()
        self.strap_detector = StrapDetector()
        self.wet_floor_detector = WetFloorDetector()
        self.orientation_detector = OrientationDetector()
        self.dock_detector = DockDetector()

        self.is_wet_floor = is_wet_floor
        self.is_dock_scene = is_dock_scene
        self.detected_events: List[BehaviourEvent] = []
        self._last_alert_time: Dict[str, float] = {}

    def process_frame(
        self,
        tracked_objects: List[TrackedObject],
        frame_idx: int,
        timestamp: float
    ) -> List[BehaviourEvent]:
        """
        Run all detectors on current tracked objects and return newly triggered unique events.
        """
        new_events: List[BehaviourEvent] = []

        # 1. Product Drop
        new_events.extend(self.drop_detector.process(tracked_objects, frame_idx, timestamp))

        # 2. Product Dragging
        new_events.extend(self.drag_detector.process(tracked_objects, frame_idx, timestamp))

        # 3. Product Throwing / Pushing
        new_events.extend(self.throw_detector.process(tracked_objects, frame_idx, timestamp))

        # 4. Product Rolling / Tumbling
        new_events.extend(self.roll_detector.process(tracked_objects, frame_idx, timestamp))

        # 5. Improper Stacking
        new_events.extend(self.stacking_detector.process(tracked_objects, frame_idx, timestamp))

        # 6. Stepping on Cartons
        new_events.extend(self.step_detector.process(tracked_objects, frame_idx, timestamp))

        # 7. Strap Pulling
        new_events.extend(self.strap_detector.process(tracked_objects, frame_idx, timestamp))

        # 8. Wet Floor Hazard
        new_events.extend(self.wet_floor_detector.process(
            tracked_objects, frame_idx, timestamp, is_wet_environment=self.is_wet_floor
        ))

        # 9. Orientation Violation
        new_events.extend(self.orientation_detector.process(tracked_objects, frame_idx, timestamp))

        # 10. Dock Level Hazard
        new_events.extend(self.dock_detector.process(
            tracked_objects, frame_idx, timestamp, is_dock_scene=self.is_dock_scene
        ))

        # Debounce to avoid flooding alerts for same track + behaviour within 2 seconds
        debounced_events: List[BehaviourEvent] = []
        for ev in new_events:
            key = f"{ev.behaviour_type}_{ev.object_track_id}"
            last_time = self._last_alert_time.get(key, -999.0)
            if timestamp - last_time >= 2.0:
                self._last_alert_time[key] = timestamp
                debounced_events.append(ev)
                self.detected_events.append(ev)

        return debounced_events
