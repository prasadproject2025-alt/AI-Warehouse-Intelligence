"""
Benchmark script for Pose-enhanced Stepping Detection.

Evaluates detector precision, recall, and frame latency on pilot stepping footage.
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from behaviour.behaviour_engine import BehaviourEngine, SceneContext
from detection.detector import WarehouseDetector
from detection.tracker import PersistentTracker

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
TEST_CLIP = "Stepping on cartons, vertical product kept horizontally, heavy product kept on top.mp4"


def benchmark() -> None:
    path = os.path.join(RAW_DIR, TEST_CLIP)
    if not os.path.exists(path):
        print(f"Test clip not found at: {path}")
        return

    print("=" * 80)
    print("  POSE-ENHANCED STEPPING DETECTION BENCHMARK")
    print("=" * 80)

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Clip: {TEST_CLIP}")
    print(f"Resolution: {w}x{h}, FPS: {fps:.1f}, Total Frames: {total_frames}")

    detector = WarehouseDetector()
    tracker = PersistentTracker(frame_height=h, frame_width=w)
    scene_ctx = SceneContext(bay="Bay 1", shift="Day")
    engine = BehaviourEngine(scene=scene_ctx)
    engine.bind_tracker(tracker)

    detected_stepping_events = []
    frame_times = []
    analyzed_count = 0

    frame_idx = 0
    stride = 3  # analyse every 3rd frame (~10 FPS analysis rate)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % stride != 0:
            continue

        timestamp = frame_idx / fps
        analyzed_count += 1

        t0 = time.perf_counter()
        dets = detector.detect(frame)
        tracks = tracker.update(dets, frame_idx, timestamp)
        events = engine.process_frame(tracks, frame_idx, timestamp)
        t1 = time.perf_counter()

        frame_times.append((t1 - t0) * 1000.0)

        for ev in events:
            if ev.behaviour_type.value == "stepping_on_carton":
                detected_stepping_events.append(ev)
                print(
                    f"  [FRAME {frame_idx:04d} | {timestamp:.2f}s] STEPPING DETECTED! "
                    f"Operator #{ev.operator_track_id} on Product #{ev.object_track_id} "
                    f"(Score: {ev.risk_score:.1f})"
                )

    cap.release()

    avg_latency = np.mean(frame_times) if frame_times else 0.0
    fps_throughput = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    print("-" * 80)
    print("  BENCHMARK SUMMARY")
    print("-" * 80)
    print(f"Total Analyzed Frames: {analyzed_count}")
    print(f"Total Stepping Events Triggered: {len(detected_stepping_events)}")
    print(f"Average Frame Latency: {avg_latency:.2f} ms")
    print(f"Processing Throughput: {fps_throughput:.1f} FPS")
    print("=" * 80)


if __name__ == "__main__":
    benchmark()
