"""
Master Behaviour Intelligence Engine.

Orchestrates the twelve behaviour detectors, builds the shared per-frame
reasoning context (floor line, scene scale, equipment presence, declared scene
conditions, historical recurrence) and de-duplicates events.

The engine is the layer that turns per-frame perception into temporal
understanding: it owns the scene model that individual detectors read, and it
guarantees that one continuous real-world action produces one incident.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import config
from behaviour.base import BehaviourEvent, BehaviourType, ImplementationStatus
from behaviour.kinematic_detectors import (
    DragDetector,
    DropDetector,
    RollDetector,
    SceneScale,
    ThrowDetector,
)
from behaviour.scene_detectors import DockDetector, WetFloorDetector
from behaviour.spatial_detectors import (
    DesignatedAreaDetector,
    LoadingSequenceDetector,
    OrientationDetector,
    StackingDetector,
    SteppingDetector,
    UnsupportedHandlingDetector,
    handling_equipment_present,
)
from detection.object_classes import WarehouseEntity
from detection.tracker import PersistentTracker, TrackedObject

logger = logging.getLogger(__name__)


class SceneContext:
    """Declared, site-specific facts about what a camera is looking at."""

    def __init__(
        self,
        bay: str = "Unassigned Bay",
        shift: str = "Unassigned Shift",
        camera_id: str = "CAM-01",
        floor_condition: str = "unknown",
        dock_transfer: bool = False,
        staging_zone: Optional[List[List[float]]] = None,
    ) -> None:
        self.bay = bay
        self.shift = shift
        self.camera_id = camera_id
        self.floor_condition = (floor_condition or "unknown").lower()
        self.dock_transfer = bool(dock_transfer)
        self.staging_zone = staging_zone

    @property
    def wet(self) -> bool:
        return self.floor_condition == "wet"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bay": self.bay,
            "shift": self.shift,
            "camera_id": self.camera_id,
            "floor_condition": self.floor_condition,
            "dock_transfer": self.dock_transfer,
            "staging_zone_configured": bool(self.staging_zone),
        }


class BehaviourEngine:
    def __init__(
        self,
        scene: Optional[SceneContext] = None,
        recurrence_baseline: Optional[Dict[str, int]] = None,
        cooldown_sec: Optional[float] = None,
    ) -> None:
        cd = config.ALERT_COOLDOWN_SEC if cooldown_sec is None else cooldown_sec
        self.scene = scene or SceneContext()
        self.recurrence: Dict[str, int] = dict(recurrence_baseline or {})

        self.drop_detector = DropDetector(cd)
        self.throw_detector = ThrowDetector(cd)
        self.drag_detector = DragDetector(cd)
        self.roll_detector = RollDetector(cd)
        self.stacking_detector = StackingDetector(max(cd, 6.0))
        self.step_detector = SteppingDetector(max(cd, 6.0))
        self.orientation_detector = OrientationDetector(max(cd, 8.0))
        self.unsupported_detector = UnsupportedHandlingDetector(max(cd, 8.0))
        self.wet_floor_detector = WetFloorDetector(max(cd, 8.0))
        self.dock_detector = DockDetector(max(cd, 8.0))
        self.designated_area_detector = DesignatedAreaDetector(max(cd, 10.0))
        self.loading_sequence_detector = LoadingSequenceDetector(max(cd, 10.0))

        self.detectors = [
            self.drop_detector,
            self.throw_detector,
            self.drag_detector,
            self.roll_detector,
            self.stacking_detector,
            self.step_detector,
            self.orientation_detector,
            self.unsupported_detector,
            self.wet_floor_detector,
            self.dock_detector,
            self.designated_area_detector,
            self.loading_sequence_detector,
        ]

        self.scale = SceneScale()
        self.detected_events: List[BehaviourEvent] = []
        self._tracker: Optional[PersistentTracker] = None

    # ---------------------------------------------------------------- context
    def bind_tracker(self, tracker: PersistentTracker) -> None:
        """Give the engine access to the scene model the tracker maintains."""
        self._tracker = tracker

    def _build_context(self, tracks: List[TrackedObject]) -> Dict[str, Any]:
        floor = self._tracker.floor_line_norm if self._tracker else 0.82
        # Depth-aware ground plane: returns the floor height at the depth implied
        # by an operator's apparent stature, or None while the fit is unsupported.
        ground_plane = self._tracker.expected_floor_y if self._tracker else None
        residual = self._tracker.floor_fit_residual if self._tracker else 0.05
        if ground_plane is not None and ground_plane(0.4) is None:
            ground_plane = None
        return {
            "floor_line_norm": floor,
            "ground_plane": ground_plane,
            "ground_plane_residual": residual,
            "scale": self.scale,
            "recurrence": self.recurrence,
            "bay": self.scene.bay,
            "shift": self.scene.shift,
            "camera_id": self.scene.camera_id,
            "wet_floor_active": self.scene.wet,
            "floor_condition_source": "declared_at_ingest",
            "dock_transfer_active": self.scene.dock_transfer,
            "staging_zone": self.scene.staging_zone,
            "handling_equipment_present": handling_equipment_present(tracks),
            "vehicle_detected": any(
                t.entity_type is WarehouseEntity.VEHICLE and t.hits >= 3 for t in tracks
            ),
        }

    # ------------------------------------------------------------------ frame
    def process_frame(
        self, tracked_objects: List[TrackedObject], frame_idx: int, timestamp: float
    ) -> List[BehaviourEvent]:
        """Run every detector over the current scene state and return new events."""
        confirmed = [t for t in tracked_objects if t.hits >= 3 and t.consecutive_lost == 0]
        self.scale.observe(confirmed)
        ctx = self._build_context(confirmed)

        new_events: List[BehaviourEvent] = []
        for detector in self.detectors:
            try:
                new_events.extend(detector.process(confirmed, frame_idx, timestamp, ctx))
            except Exception:  # noqa: BLE001 - one detector must not stop the pipeline
                logger.exception(
                    "Behaviour detector %s failed at frame %s", type(detector).__name__, frame_idx
                )

        for ev in new_events:
            self.recurrence[ev.behaviour_type.value] = (
                self.recurrence.get(ev.behaviour_type.value, 0) + 1
            )
            self.detected_events.append(ev)

        return new_events

    # --------------------------------------------------------------- coverage
    @staticmethod
    def coverage_report() -> List[Dict[str, str]]:
        """
        Honest per-behaviour implementation status, rendered by the dashboard.

        This is generated from the detector classes themselves, so the UI cannot
        drift out of step with what the code actually does.
        """
        from behaviour.kinematic_detectors import (
            DragDetector as _Drag,
            DropDetector as _Drop,
            RollDetector as _Roll,
            ThrowDetector as _Throw,
        )
        from behaviour.scene_detectors import DockDetector as _Dock, WetFloorDetector as _Wet
        from behaviour.spatial_detectors import (
            DesignatedAreaDetector as _Zone,
            LoadingSequenceDetector as _Seq,
            OrientationDetector as _Orient,
            StackingDetector as _Stack,
            SteppingDetector as _Step,
            UnsupportedHandlingDetector as _Unsup,
        )

        labels = {
            BehaviourType.PRODUCT_DROP: ("Product dropped", "Descent -> impact -> at rest state chain"),
            BehaviourType.PRODUCT_DRAG: ("Product dragged", "Sustained floor-plane sliding with operator contact"),
            BehaviourType.PRODUCT_THROW: ("Product thrown / pushed", "Release velocity + unsupported flight"),
            BehaviourType.ROLLING_PRODUCT: ("Product rolled / tumbled", "Cyclical aspect-ratio inversion on floor"),
            BehaviourType.IMPROPER_STACKING: ("Improper / unstable stacking", "Persistent overhang or heavy-on-light geometry"),
            BehaviourType.STEPPING_ON_CARTON: ("Stepping on cartons", "Operator feet above floor line inside package footprint"),
            BehaviourType.UNSUPPORTED_HANDLING: ("Handled without equipment", "Large item carried with no trolley in scene"),
            BehaviourType.WET_FLOOR_HAZARD: ("Handling on wet floor", "Declared floor condition + observed floor movement"),
            BehaviourType.ORIENTATION_VIOLATION: ("Upright product kept flat", "Observed upright-to-flat transition"),
            BehaviourType.DOCK_LEVEL_HAZARD: ("Dock level / transition hazard", "Declared dock + unaided heavy slide"),
            BehaviourType.OUTSIDE_DESIGNATED_AREA: ("Product outside designated area", "Settled outside configured staging polygon"),
            BehaviourType.UNSAFE_LOADING_SEQUENCE: ("Unsafe loading sequence", "Concurrent uncontrolled handling at one point"),
        }
        classes = [_Drop, _Drag, _Throw, _Roll, _Stack, _Step, _Unsup, _Wet, _Orient, _Dock, _Zone, _Seq]

        report = []
        for cls in classes:
            name, method = labels[cls.behaviour_type]
            d = cls.describe()
            d.update({"label": name, "method": method})
            report.append(d)
        return report
