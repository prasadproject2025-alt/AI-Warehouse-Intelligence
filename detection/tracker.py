"""
Persistent Multi-Object Tracker for Warehouse Operations.

Maintains persistent track IDs and, crucially for behaviour reasoning, a
*normalised* kinematic history: velocities are expressed in frame-heights per
second rather than pixels per second, so every downstream threshold is
independent of camera resolution and zoom.

Each track also carries a coarse motion state (STATIONARY / CARRIED / SLIDING /
FALLING / SETTLED). The behaviour detectors consume state *transitions*, which
is what turns a per-frame observation into a temporal event.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from detection.detector import Detection
from detection.object_classes import PRODUCT_ENTITIES, WarehouseEntity

#: Track history length in analysed frames (~4 s at a 5 Hz analysis rate).
HISTORY_LEN = 60


class MotionState(str, Enum):
    STATIONARY = "stationary"
    CARRIED = "carried"      # moving while an operator is in contact range
    SLIDING = "sliding"      # horizontal motion with the base on the floor band
    FALLING = "falling"      # sustained downward motion
    SETTLED = "settled"      # came to rest after a period of motion


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class TrackedObject:
    """One tracked entity with normalised kinematics and a motion state."""

    def __init__(
        self,
        track_id: int,
        detection: Detection,
        frame_idx: int,
        timestamp: float,
        frame_height: float,
        frame_width: float,
    ) -> None:
        self.track_id = track_id
        self.entity_type = detection.entity_type
        self.raw_class = detection.raw_class
        self.confidence = detection.confidence
        self.box = list(detection.box)
        self.center = list(detection.center)
        self.width = detection.width
        self.height = detection.height

        self.frame_height = max(1.0, float(frame_height))
        self.frame_width = max(1.0, float(frame_width))

        # Velocity in frame-heights per second (resolution independent).
        self.vx = 0.0
        self.vy = 0.0
        self.ay = 0.0

        self.distance_travelled = 0.0  # normalised units
        self.start_frame = frame_idx
        self.start_time = timestamp
        self.last_seen_frame = frame_idx
        self.last_seen_time = timestamp
        self.consecutive_lost = 0
        self.hits = 1

        self.state = MotionState.STATIONARY
        self.state_since = timestamp
        self.state_history: List[Tuple[str, float]] = [(self.state.value, timestamp)]
        # Set by the tracker each frame from operator proximity.
        self.operator_contact_id: Optional[int] = None
        self.max_height_seen = self.height
        self.max_aspect_seen = self.width / max(1.0, self.height)
        self.min_aspect_seen = self.max_aspect_seen

        self.history: List[Dict[str, Any]] = [self._record(frame_idx, timestamp)]

    # ------------------------------------------------------------------ utils
    @property
    def age_sec(self) -> float:
        return max(0.0, self.last_seen_time - self.start_time)

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(1.0, self.height)

    @property
    def bottom_y_norm(self) -> float:
        """Base of the object as a fraction of frame height (1.0 = frame bottom)."""
        return self.box[3] / self.frame_height

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))

    @property
    def is_product(self) -> bool:
        return self.entity_type in PRODUCT_ENTITIES

    def _record(self, frame_idx: int, timestamp: float) -> Dict[str, Any]:
        return {
            "frame": frame_idx,
            "time": timestamp,
            "box": list(self.box),
            "center": list(self.center),
            "vx": round(self.vx, 4),
            "vy": round(self.vy, 4),
            "ay": round(self.ay, 4),
            "width": round(self.width, 1),
            "height": round(self.height, 1),
            "aspect_ratio": round(self.aspect_ratio, 3),
            "bottom_y_norm": round(self.bottom_y_norm, 4),
            "state": self.state.value,
            "operator_contact": self.operator_contact_id,
        }

    # ----------------------------------------------------------------- update
    def update(
        self, detection: Detection, frame_idx: int, timestamp: float, fps: float = 30.0
    ) -> None:
        dt = max(1.0 / max(fps, 1.0), timestamp - self.last_seen_time)
        new_center = detection.center

        # Normalise displacement by frame height before differentiating.
        inst_vx = (new_center[0] - self.center[0]) / self.frame_height / dt
        inst_vy = (new_center[1] - self.center[1]) / self.frame_height / dt

        alpha = 0.55
        prev_vy = self.vy
        self.vx = alpha * inst_vx + (1 - alpha) * self.vx
        self.vy = alpha * inst_vy + (1 - alpha) * self.vy
        self.ay = (self.vy - prev_vy) / dt

        step = np.hypot(
            (new_center[0] - self.center[0]) / self.frame_height,
            (new_center[1] - self.center[1]) / self.frame_height,
        )
        self.distance_travelled += float(step)

        self.box = list(detection.box)
        self.center = list(new_center)
        self.width = detection.width
        self.height = detection.height
        self.confidence = 0.7 * detection.confidence + 0.3 * self.confidence
        self.last_seen_frame = frame_idx
        self.last_seen_time = timestamp
        self.consecutive_lost = 0
        self.hits += 1

        self.max_height_seen = max(self.max_height_seen, self.height)
        self.max_aspect_seen = max(self.max_aspect_seen, self.aspect_ratio)
        self.min_aspect_seen = min(self.min_aspect_seen, self.aspect_ratio)

        self.history.append(self._record(frame_idx, timestamp))
        if len(self.history) > HISTORY_LEN:
            self.history.pop(0)

    def set_state(self, state: MotionState, timestamp: float) -> None:
        if state is not self.state:
            self.state = state
            self.state_since = timestamp
            self.state_history.append((state.value, timestamp))
            if len(self.state_history) > 40:
                self.state_history.pop(0)
            if self.history:
                self.history[-1]["state"] = state.value

    def state_sequence(self, last_n: int = 6) -> List[str]:
        return [s for s, _ in self.state_history[-last_n:]]

    def mark_missed(self) -> None:
        self.consecutive_lost += 1

    def recent(self, seconds: float) -> List[Dict[str, Any]]:
        """History entries within the last ``seconds`` of track time."""
        if not self.history:
            return []
        cutoff = self.history[-1]["time"] - seconds
        return [h for h in self.history if h["time"] >= cutoff]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "entity_type": self.entity_type.value,
            "raw_class": self.raw_class,
            "box": [round(b, 1) for b in self.box],
            "center": [round(c, 1) for c in self.center],
            "vx": round(self.vx, 3),
            "vy": round(self.vy, 3),
            "state": self.state.value,
            "distance": round(self.distance_travelled, 3),
            "confidence": round(self.confidence, 3),
            "age_sec": round(self.age_sec, 2),
            "hits": self.hits,
        }


class PersistentTracker:
    """
    Greedy IoU + proximity association tracker.

    Also estimates a scene floor line from the observed foot positions of
    operators, which lets the behaviour layer tell "object resting on the
    floor" apart from "object up on a stack", without any hard-coded pixel
    constants.
    """

    def __init__(
        self,
        max_lost_frames: int = 12,
        iou_threshold: float = 0.2,
        frame_height: float = 720.0,
        frame_width: float = 1280.0,
        contact_radius_norm: float = 0.22,
    ) -> None:
        self.max_lost_frames = max_lost_frames
        self.iou_threshold = iou_threshold
        self.frame_height = max(1.0, float(frame_height))
        self.frame_width = max(1.0, float(frame_width))
        self.contact_radius_norm = contact_radius_norm
        self.tracks: Dict[int, TrackedObject] = {}
        self._next_track_id = 1
        self._foot_samples: List[Tuple[float, float]] = []
        self._floor_fit_residual = 0.05

    # ------------------------------------------------------------ floor model
    @property
    def floor_line_norm(self) -> float:
        """
        Median floor level as a fraction of frame height, for coarse "is this
        object down at floor level" tests. Perspective-sensitive decisions must
        use :meth:`expected_floor_y` instead.
        """
        if len(self._foot_samples) < 20:
            return 0.82
        return float(np.median([f for f, _ in self._foot_samples]))

    def expected_floor_y(self, person_height_norm: float) -> Optional[float]:
        """
        Ground-plane height at the depth implied by an operator's apparent size.

        A ceiling or wall camera sees the floor recede up the image, so a person
        standing far away legitimately has their feet high in the frame. Using
        one global floor line therefore flags every distant worker as standing
        on something. Apparent stature is a usable depth cue: for anyone on the
        floor, foot position varies almost linearly with apparent height, so a
        least-squares fit over observed operators gives the floor line *at that
        depth*. Residuals above it are genuine elevation.

        Returns ``None`` until the fit is supported by enough spread.
        """
        if len(self._foot_samples) < 25:
            return None
        heights = np.array([h for _, h in self._foot_samples], dtype=float)
        feet = np.array([f for f, _ in self._foot_samples], dtype=float)
        if float(heights.max() - heights.min()) < 0.05:
            return None  # everyone at the same depth: no usable gradient
        slope, intercept = np.polyfit(heights, feet, 1)
        self._floor_fit_residual = float(np.std(feet - (slope * heights + intercept)))
        return float(slope * person_height_norm + intercept)

    @property
    def floor_fit_residual(self) -> float:
        """Spread of the ground-plane fit; the noise floor for elevation tests."""
        return getattr(self, "_floor_fit_residual", 0.05)

    def _sample_floor(self, tracks: List[TrackedObject]) -> None:
        for t in tracks:
            if t.entity_type is WarehouseEntity.OPERATOR and t.hits > 3:
                self._foot_samples.append(
                    (t.box[3] / self.frame_height, t.height / self.frame_height)
                )
        if len(self._foot_samples) > 600:
            self._foot_samples = self._foot_samples[-600:]

    # ------------------------------------------------------------------ state
    def _annotate_states(self, timestamp: float) -> None:
        operators = [
            t for t in self.tracks.values() if t.entity_type is WarehouseEntity.OPERATOR
        ]
        floor = self.floor_line_norm

        for trk in self.tracks.values():
            if trk.entity_type is WarehouseEntity.OPERATOR:
                continue

            # Nearest operator whose reach envelope contains the product.
            trk.operator_contact_id = None
            best = self.contact_radius_norm
            for op in operators:
                dx = abs(op.center[0] - trk.center[0]) / self.frame_height
                dy = abs(op.center[1] - trk.center[1]) / self.frame_height
                # Reach scales with the operator's apparent size (depth proxy).
                reach = max(self.contact_radius_norm, (op.height / self.frame_height) * 0.9)
                dist = float(np.hypot(dx, dy))
                if dist < reach and dist < best * 1.6:
                    best = dist
                    trk.operator_contact_id = op.track_id

            speed = trk.speed
            on_floor = trk.bottom_y_norm > (floor - 0.06)

            if speed < 0.03:
                # Distinguish "never moved" from "came to rest after motion".
                was_moving = any(
                    np.hypot(h["vx"], h["vy"]) > 0.10 for h in trk.recent(1.5)
                )
                new_state = MotionState.SETTLED if was_moving else MotionState.STATIONARY
            elif trk.vy > 0.28 and trk.vy > abs(trk.vx) * 1.1:
                new_state = MotionState.FALLING
            elif on_floor and abs(trk.vx) > 0.05 and abs(trk.vy) < abs(trk.vx) * 0.7:
                new_state = MotionState.SLIDING
            elif trk.operator_contact_id is not None:
                new_state = MotionState.CARRIED
            else:
                new_state = MotionState.STATIONARY if speed < 0.06 else MotionState.CARRIED

            trk.set_state(new_state, timestamp)

    # ----------------------------------------------------------------- update
    def update(
        self,
        detections: List[Detection],
        frame_idx: int,
        timestamp: float,
        fps: float = 30.0,
    ) -> List[TrackedObject]:
        active_tracks = list(self.tracks.values())
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())
        matches: List[Tuple[int, int]] = []

        if active_tracks and detections:
            cost = np.full((len(active_tracks), len(detections)), -1.0)
            for i, trk in enumerate(active_tracks):
                for j, det in enumerate(detections):
                    # Never associate an operator with a product.
                    trk_is_op = trk.entity_type is WarehouseEntity.OPERATOR
                    det_is_op = det.entity_type is WarehouseEntity.OPERATOR
                    if trk_is_op != det_is_op:
                        continue
                    iou = compute_iou(trk.box, det.box)
                    dist = np.hypot(
                        trk.center[0] - det.center[0], trk.center[1] - det.center[1]
                    ) / self.frame_height
                    proximity = max(0.0, 1.0 - dist / 0.25)
                    class_bonus = 0.15 if trk.entity_type is det.entity_type else 0.0
                    cost[i, j] = (iou * 0.6) + (proximity * 0.25) + class_bonus

            while True:
                max_val = float(np.max(cost)) if cost.size else -1.0
                if max_val < 0.22:
                    break
                i, j = np.unravel_index(int(np.argmax(cost)), cost.shape)
                trk = active_tracks[i]
                matches.append((trk.track_id, int(j)))
                cost[i, :] = -1.0
                cost[:, j] = -1.0
                if j in unmatched_detections:
                    unmatched_detections.remove(int(j))
                if trk.track_id in unmatched_tracks:
                    unmatched_tracks.remove(trk.track_id)

        for track_id, det_idx in matches:
            det = detections[det_idx]
            det.track_id = track_id
            self.tracks[track_id].update(det, frame_idx, timestamp, fps)

        for track_id in list(unmatched_tracks):
            trk = self.tracks[track_id]
            trk.mark_missed()
            if trk.consecutive_lost > self.max_lost_frames:
                del self.tracks[track_id]

        for det_idx in unmatched_detections:
            det = detections[det_idx]
            new_id = self._next_track_id
            self._next_track_id += 1
            det.track_id = new_id
            self.tracks[new_id] = TrackedObject(
                new_id, det, frame_idx, timestamp, self.frame_height, self.frame_width
            )

        live = list(self.tracks.values())
        self._sample_floor(live)
        self._annotate_states(timestamp)
        return live

    def confirmed_tracks(self, min_hits: int = 3) -> List[TrackedObject]:
        """Tracks seen often enough to reason about (suppresses one-frame blips)."""
        return [t for t in self.tracks.values() if t.hits >= min_hits and t.consecutive_lost == 0]
