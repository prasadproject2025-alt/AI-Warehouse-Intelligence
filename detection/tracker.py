"""
Persistent Multi-Object Tracker for Warehouse Operations
Maintains persistent Track IDs, velocity, acceleration, and temporal trajectory buffers.
"""

from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from detection.detector import Detection
from detection.object_classes import WarehouseEntity

def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea
    if unionArea <= 0:
        return 0.0
    return interArea / unionArea

class TrackedObject:
    def __init__(self, track_id: int, initial_detection: Detection, frame_idx: int, timestamp: float):
        self.track_id = track_id
        self.entity_type = initial_detection.entity_type
        self.raw_class = initial_detection.raw_class
        self.confidence = initial_detection.confidence
        self.box = list(initial_detection.box)
        self.center = list(initial_detection.center)
        self.width = initial_detection.width
        self.height = initial_detection.height
        
        # Velocity & Acceleration
        self.vx = 0.0
        self.vy = 0.0
        self.ax = 0.0
        self.ay = 0.0
        
        # Cumulative metrics
        self.distance_travelled = 0.0
        self.start_frame = frame_idx
        self.start_time = timestamp
        self.last_seen_frame = frame_idx
        self.last_seen_time = timestamp
        self.consecutive_lost = 0
        
        # Trajectory history buffer: list of dicts
        self.history: List[Dict[str, Any]] = [{
            "frame": frame_idx,
            "time": timestamp,
            "box": list(self.box),
            "center": list(self.center),
            "vx": 0.0,
            "vy": 0.0,
            "width": self.width,
            "height": self.height,
            "aspect_ratio": self.width / max(1.0, self.height)
        }]

    def update(self, detection: Detection, frame_idx: int, timestamp: float, fps: float = 30.0):
        dt = max(1.0 / fps, timestamp - self.last_seen_time)
        new_center = detection.center
        
        # Calculate instantaneous velocity (pixels/sec)
        instant_vx = (new_center[0] - self.center[0]) / dt
        instant_vy = (new_center[1] - self.center[1]) / dt
        
        # Smooth velocity with exponential moving average
        alpha = 0.6
        prev_vx, prev_vy = self.vx, self.vy
        self.vx = alpha * instant_vx + (1 - alpha) * self.vx
        self.vy = alpha * instant_vy + (1 - alpha) * self.vy
        
        # Calculate acceleration (pixels/sec^2)
        self.ax = (self.vx - prev_vx) / dt
        self.ay = (self.vy - prev_vy) / dt
        
        # Cumulative distance
        step_dist = np.hypot(new_center[0] - self.center[0], new_center[1] - self.center[1])
        self.distance_travelled += step_dist

        # Update spatial attributes
        self.box = list(detection.box)
        self.center = list(new_center)
        self.width = detection.width
        self.height = detection.height
        self.confidence = 0.7 * detection.confidence + 0.3 * self.confidence
        self.last_seen_frame = frame_idx
        self.last_seen_time = timestamp
        self.consecutive_lost = 0
        
        # Maintain history buffer (keep last 90 frames / 3 seconds)
        record = {
            "frame": frame_idx,
            "time": timestamp,
            "box": list(self.box),
            "center": list(self.center),
            "vx": round(self.vx, 2),
            "vy": round(self.vy, 2),
            "ax": round(self.ax, 2),
            "ay": round(self.ay, 2),
            "width": round(self.width, 1),
            "height": round(self.height, 1),
            "aspect_ratio": round(self.width / max(1.0, self.height), 3)
        }
        self.history.append(record)
        if len(self.history) > 90:
            self.history.pop(0)

    def mark_missed(self):
        self.consecutive_lost += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "entity_type": self.entity_type.value,
            "raw_class": self.raw_class,
            "box": [round(b, 1) for b in self.box],
            "center": [round(c, 1) for c in self.center],
            "vx": round(self.vx, 1),
            "vy": round(self.vy, 1),
            "distance": round(self.distance_travelled, 1),
            "confidence": round(self.confidence, 3),
            "active_frames": len(self.history)
        }

class PersistentTracker:
    def __init__(self, max_lost_frames: int = 25, iou_threshold: float = 0.25):
        self.max_lost_frames = max_lost_frames
        self.iou_threshold = iou_threshold
        self.tracks: Dict[int, TrackedObject] = {}
        self._next_track_id = 1

    def update(
        self,
        detections: List[Detection],
        frame_idx: int,
        timestamp: float,
        fps: float = 30.0
    ) -> List[TrackedObject]:
        """
        Associate detections with existing tracks using IoU & spatial proximity.
        """
        active_tracks = list(self.tracks.values())
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())
        matches: List[Tuple[int, int]] = []

        if active_tracks and detections:
            # Build cost matrix based on IoU and center distance
            cost_matrix = np.zeros((len(active_tracks), len(detections)))
            for i, trk in enumerate(active_tracks):
                for j, det in enumerate(detections):
                    iou = compute_iou(trk.box, det.box)
                    # Class match bonus
                    class_bonus = 0.2 if trk.entity_type == det.entity_type else -0.1
                    
                    # Normalized center distance penalty
                    dist = np.hypot(trk.center[0] - det.center[0], trk.center[1] - det.center[1])
                    norm_dist = max(0.0, 1.0 - (dist / 200.0))
                    
                    score = (iou * 0.6) + (norm_dist * 0.3) + class_bonus
                    cost_matrix[i, j] = score

            # Greedy bipartite matching
            while True:
                max_val = np.max(cost_matrix) if cost_matrix.size > 0 else -1.0
                if max_val < 0.2:
                    break
                i, j = np.unravel_index(np.argmax(cost_matrix), cost_matrix.shape)
                trk = active_tracks[i]
                matches.append((trk.track_id, j))
                cost_matrix[i, :] = -1.0
                cost_matrix[:, j] = -1.0
                if j in unmatched_detections:
                    unmatched_detections.remove(j)
                if trk.track_id in unmatched_tracks:
                    unmatched_tracks.remove(trk.track_id)

        # Update matched tracks
        for track_id, det_idx in matches:
            det = detections[det_idx]
            det.track_id = track_id
            self.tracks[track_id].update(det, frame_idx, timestamp, fps)

        # Mark unmatched tracks as missed
        for track_id in unmatched_tracks:
            self.tracks[track_id].mark_missed()
            if self.tracks[track_id].consecutive_lost > self.max_lost_frames:
                del self.tracks[track_id]

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            new_id = self._next_track_id
            self._next_track_id += 1
            det.track_id = new_id
            new_track = TrackedObject(new_id, det, frame_idx, timestamp)
            self.tracks[new_id] = new_track

        return list(self.tracks.values())
