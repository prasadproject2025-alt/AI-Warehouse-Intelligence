import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
from detection.detector import WarehouseDetector
from detection.tracker import PersistentTracker

def test_detector_and_tracker():
    video_path = "data/raw/Rolling and dropping carton.mp4"
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open {video_path}"
    
    detector = WarehouseDetector(conf_threshold=0.25)
    tracker = PersistentTracker()
    
    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    total_tracks_seen = set()
    while frame_idx < 30: # test first 30 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = frame_idx / fps
        detections = detector.detect(frame)
        active_tracks = tracker.update(detections, frame_idx, timestamp, fps)
        for trk in active_tracks:
            total_tracks_seen.add(trk.track_id)
        frame_idx += 1
        
    cap.release()
    print(f"Tracking Test Passed: Processed {frame_idx} frames, discovered persistent track IDs: {total_tracks_seen}")
    assert len(total_tracks_seen) > 0, "No tracks discovered"

if __name__ == "__main__":
    test_detector_and_tracker()
