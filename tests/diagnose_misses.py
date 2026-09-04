"""
Miss diagnosis.

When a clip is labelled with a behaviour the system did not report, this script
answers *which gate rejected it*: perception (no product track at all) or a
specific threshold inside the behaviour detector.

Guessing at thresholds without this is how false positives get reintroduced.

    python tests/diagnose_misses.py "Rolling and dragging on wet floor.mp4"
"""

from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

import config
from detection.detector import WarehouseDetector
from detection.object_classes import WarehouseEntity
from detection.tracker import PersistentTracker

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

TARGETS = [
    "Rolling and dragging on wet floor.mp4",
    "Stepping on cartons, vertical product kept horizontally, heavy product kept on top.mp4",
]


def diagnose(clip: str, stride: int = 3) -> None:
    path = os.path.join(RAW, clip)
    if not os.path.exists(path):
        print(f"missing: {clip}")
        return

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detector = WarehouseDetector()
    tracker = PersistentTracker(frame_height=h, frame_width=w)
    scale = config.MAX_INFERENCE_WIDTH / float(w) if w > config.MAX_INFERENCE_WIDTH else 1.0

    frames_with_product = 0
    raw_product_dets = 0
    frames_with_raw_product = 0
    all_product_hits = []
    analysed = 0
    product_track_life = defaultdict(int)   # track_id -> frames survived
    entity_counts = Counter()
    # Stepping-specific probes
    feet_inside_footprint = 0
    ground_plane_available = 0
    elevation_samples = []

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            infer = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame
            dets = detector.detect(infer)
            if scale != 1.0:
                for d in dets:
                    d.box = [v / scale for v in d.box]
            n_raw = sum(1 for d in dets if d.entity_type is not WarehouseEntity.OPERATOR
                        and d.entity_type is not WarehouseEntity.VEHICLE)
            raw_product_dets += n_raw
            if n_raw:
                frames_with_raw_product += 1
            tracks = tracker.update(dets, idx, idx / fps, fps)
            all_product_hits.extend(t.hits for t in tracks if t.is_product)
            analysed += 1

            products = [t for t in tracks if t.is_product and t.hits >= 4]
            operators = [t for t in tracks if t.entity_type is WarehouseEntity.OPERATOR and t.hits >= 4]
            for t in tracks:
                entity_counts[t.entity_type.value] += 1
            if products:
                frames_with_product += 1
            for p in products:
                product_track_life[p.track_id] += 1

            gp = tracker.expected_floor_y
            if gp is not None and gp(0.4) is None:
                gp = None
            if gp is not None:
                ground_plane_available += 1
                for op in operators:
                    exp = gp(op.height / h)
                    if exp is None:
                        continue
                    elevation = exp - (op.box[3] / h)
                    for pr in products:
                        if pr.box[0] <= op.center[0] <= pr.box[2] and abs(op.box[3] - pr.box[1]) <= 0.10 * h:
                            feet_inside_footprint += 1
                            elevation_samples.append(elevation)
        idx += 1
    cap.release()

    print("=" * 92)
    print(f"  {clip}")
    print("=" * 92)
    print(f"  frames analysed              : {analysed} of {total}")
    print(f"  RAW product detections       : {raw_product_dets} across {frames_with_raw_product} frame(s) ({frames_with_raw_product/max(1,analysed)*100:.0f}%)")
    if all_product_hits:
        import numpy as _np
        print(f"  product track hit counts     : max {max(all_product_hits)}  p90 {int(_np.percentile(all_product_hits,90))}  (need >= 4 to confirm)")
    print(f"  CONFIRMED product tracks     : {frames_with_product} ({frames_with_product/max(1,analysed)*100:.0f}%)")
    print(f"  distinct product tracks      : {len(product_track_life)}")
    if product_track_life:
        lives = sorted(product_track_life.values(), reverse=True)
        print(f"  longest product track (frames): {lives[0]}  | median {int(np.median(lives))}")
        print(f"  tracks surviving >= 10 frames : {sum(1 for v in lives if v >= 10)}")
    print(f"  entity mix                   : {dict(entity_counts.most_common(6))}")
    print(f"  frames with a ground plane   : {ground_plane_available} ({ground_plane_available/max(1,analysed)*100:.0f}%)")
    print(f"  operator feet inside a package footprint: {feet_inside_footprint} frame(s)")
    if elevation_samples:
        arr = np.array(elevation_samples)
        print(f"  elevation above floor        : max {arr.max():.3f}  p90 {np.percentile(arr,90):.3f}  (needs >= ~0.045)")
    print()

    # Verdict
    if raw_product_dets == 0:
        print("  VERDICT: perception. The detector never saw a product at all.")
    elif frames_with_product == 0:
        print("  VERDICT: track confirmation. Products ARE detected but no track ever")
        print("           reaches the hits>=4 confirmation gate - detections are too")
        print("           intermittent for the tracker to hold an identity.")
    elif not product_track_life or max(product_track_life.values()) < 10:
        print("  VERDICT: track fragmentation. Products are detected but the tracks die")
        print("           before a behaviour window can accumulate.")
    elif feet_inside_footprint == 0:
        print("  VERDICT: geometry. Products are tracked, but the operator's feet were")
        print("           never inside a package footprint at its top surface.")
    else:
        print("  VERDICT: threshold. The geometric precondition was met; the elevation or")
        print("           dwell gate rejected it.")
    print()


if __name__ == "__main__":
    clips = sys.argv[1:] or TARGETS
    for c in clips:
        diagnose(c)
