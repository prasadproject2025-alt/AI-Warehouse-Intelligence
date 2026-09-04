"""
Perception coverage audit.

Measures how often the detector actually finds operators and *products* in each
pilot clip. This is the number that explains the behaviour-recall ceiling: no
behaviour detector can report a dropped carton that was never detected.

Run:  python -m tests.audit_perception [--frames 40]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config  # noqa: E402
from detection.detector import WarehouseDetector  # noqa: E402
from detection.object_classes import PRODUCT_ENTITIES, WarehouseEntity  # noqa: E402


def audit(raw_dir: str = "data/raw", n_frames: int = 40) -> int:
    clips = sorted(f for f in os.listdir(raw_dir) if f.lower().endswith(".mp4"))
    if not clips:
        print(f"No videos found in {raw_dir}")
        return 1

    detector = WarehouseDetector()
    print("=" * 104)
    print(f"  PERCEPTION COVERAGE AUDIT — backend: {detector.backend}, "
          f"{n_frames} frames sampled per clip")
    print("=" * 104)
    print(f"{'clip':<52}{'frames w/':>11}{'frames w/':>11}{'products':>10}{'mean prod':>11}")
    print(f"{'':<52}{'operator':>11}{'product':>11}{'per frame':>10}{'conf':>11}")
    print("-" * 104)

    summary: List[Dict] = []
    for clip in clips:
        cap = cv2.VideoCapture(os.path.join(raw_dir, clip))
        if not cap.isOpened():
            print(f"{clip[:50]:<52}  (could not open)")
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1)
        if total <= 0:
            cap.release()
            continue

        idxs = np.linspace(0, total - 1, min(n_frames, total)).astype(int)
        scale = min(1.0, (config.MAX_INFERENCE_WIDTH / width) if config.MAX_INFERENCE_WIDTH else 1.0)

        op_frames = prod_frames = prod_total = sampled = 0
        confs: List[float] = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                continue
            sampled += 1
            if scale < 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
            dets = detector.detect(frame)
            ops = [d for d in dets if d.entity_type is WarehouseEntity.OPERATOR]
            prods = [d for d in dets if d.entity_type in PRODUCT_ENTITIES]
            if ops:
                op_frames += 1
            if prods:
                prod_frames += 1
            prod_total += len(prods)
            confs.extend(d.confidence for d in prods)
        cap.release()

        if sampled == 0:
            continue
        op_pct = op_frames / sampled * 100
        prod_pct = prod_frames / sampled * 100
        per_frame = prod_total / sampled
        mean_conf = float(np.mean(confs)) if confs else 0.0

        print(f"{clip[:50]:<52}{op_pct:>10.0f}%{prod_pct:>10.0f}%{per_frame:>10.2f}{mean_conf:>11.2f}")
        summary.append({
            "clip": clip, "operator_frame_pct": op_pct, "product_frame_pct": prod_pct,
            "products_per_frame": per_frame, "mean_product_conf": mean_conf,
        })

    if not summary:
        return 1

    print("-" * 104)
    print(f"{'MEAN':<52}"
          f"{np.mean([s['operator_frame_pct'] for s in summary]):>10.0f}%"
          f"{np.mean([s['product_frame_pct'] for s in summary]):>10.0f}%"
          f"{np.mean([s['products_per_frame'] for s in summary]):>10.2f}"
          f"{np.mean([s['mean_product_conf'] for s in summary]):>11.2f}")
    print()
    print("  Reading this table:")
    print("    * 'frames w/ operator' near 100% means person detection is reliable.")
    print("    * 'frames w/ product' is the ceiling on every product-centric behaviour.")
    print("      A clip at 0% cannot yield drop/drag/throw/roll events no matter how")
    print("      good the behaviour logic is - the carton is never seen.")
    print("    * Closing that gap requires a detector trained on warehouse packaging;")
    print("      it is not a threshold-tuning problem.")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--raw-dir", default="data/raw")
    args = parser.parse_args()
    sys.exit(audit(args.raw_dir, args.frames))
