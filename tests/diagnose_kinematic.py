"""
Gate-level diagnosis for the kinematic detectors (throw / roll / drop).

Reports how far each candidate product track progressed through each detector's
sequence of gates, so a miss can be attributed to a specific condition rather
than guessed at.

    python tests/diagnose_kinematic.py
"""

from __future__ import annotations

import os
import sys
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

import config
from behaviour.behaviour_engine import BehaviourEngine, SceneContext
from behaviour.kinematic_detectors import DropDetector, RollDetector, ThrowDetector
from detection.detector import WarehouseDetector
from detection.tracker import PersistentTracker

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

CLIPS = {
    "Throwing Mattresses.mp4": "throw",
    "Throwing seating cartons, using strap to hold.mp4": "throw",
    "Rolling and dropping carton.mp4": "roll/drop",
    "Rolling and dragging on wet floor.mp4": "roll",
}


def diagnose(clip: str, expect: str) -> None:
    path = os.path.join(RAW, clip)
    if not os.path.exists(path):
        print(f"  missing: {clip}")
        return

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    det = WarehouseDetector()
    trk = PersistentTracker(frame_height=h, frame_width=w)
    eng = BehaviourEngine(scene=SceneContext(bay="D"), recurrence_baseline={})
    eng.bind_tracker(trk)
    scale = config.MAX_INFERENCE_WIDTH / float(w) if w > config.MAX_INFERENCE_WIDTH else 1.0

    gates = Counter()
    peak_speeds, dxs, flips = [], [], []

    throw = next(d for d in eng.detectors if isinstance(d, ThrowDetector))
    roll = next(d for d in eng.detectors if isinstance(d, RollDetector))
    drop = next(d for d in eng.detectors if isinstance(d, DropDetector))

    t_orig = throw.process

    def traced_throw(tracks, fi, ts, ctx):
        for t in tracks:
            if not t.is_product or t.hits < 4:
                continue
            gates["throw: product track considered"] += 1
            hist = t.recent(2.0)
            if len(hist) < 4:
                gates["throw: STOPPED history < 4 samples"] += 1
                continue
            gates["throw: has >=4 samples"] += 1
            speeds = [float(np.hypot(x["vx"], x["vy"])) for x in hist]
            pi = int(np.argmax(speeds))
            peak = speeds[pi]
            peak_speeds.append(peak)
            if peak < throw.RELEASE_SPEED:
                gates["throw: STOPPED peak speed < RELEASE_SPEED"] += 1
                continue
            gates["throw: peak speed OK"] += 1
            had = any(x.get("operator_contact") is not None for x in hist[: pi + 1])
            free = hist[pi].get("operator_contact") is None
            if not had:
                gates["throw: STOPPED no operator contact before release"] += 1
                continue
            if not free:
                gates["throw: STOPPED still in contact at peak"] += 1
                continue
            gates["throw: contact-then-release OK"] += 1
            dx = abs(hist[-1]["center"][0] - hist[0]["center"][0]) / t.frame_height
            dxs.append(dx)
            if dx < throw.MIN_HORIZONTAL:
                gates["throw: STOPPED horizontal travel too small"] += 1
                continue
            gates["throw: ALL GATES PASSED"] += 1
        return t_orig(tracks, fi, ts, ctx)

    throw.process = traced_throw

    r_orig = roll.process

    def traced_roll(tracks, fi, ts, ctx):
        for t in tracks:
            if not t.is_product or t.hits < 4:
                continue
            gates["roll: product track considered"] += 1
            hist = t.recent(getattr(roll, "WINDOW_SEC", 2.0))
            if len(hist) < getattr(roll, "MIN_SAMPLES", 6):
                gates["roll: STOPPED not enough samples"] += 1
                continue
            ars = [x.get("aspect_ratio", 1.0) for x in hist]
            n = 0
            for i in range(1, len(ars)):
                if (ars[i - 1] > 1.15 and ars[i] < 0.85) or (ars[i - 1] < 0.85 and ars[i] > 1.15):
                    n += 1
            flips.append(n)
            if n < getattr(roll, "MIN_INVERSIONS", 2):
                gates["roll: STOPPED too few aspect inversions"] += 1
                continue
            gates["roll: ALL GATES PASSED"] += 1
        return r_orig(tracks, fi, ts, ctx)

    roll.process = traced_roll

    idx, evs = 0, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if idx % 3 == 0:
            infer = cv2.resize(f, None, fx=scale, fy=scale) if scale != 1.0 else f
            ds = det.detect(infer)
            if scale != 1.0:
                for d in ds:
                    d.box = [v / scale for v in d.box]
            tracks = trk.update(ds, idx, idx / fps, fps)
            evs += len(eng.process_frame(tracks, idx, idx / fps))
        idx += 1
    cap.release()

    print("=" * 90)
    print(f"  {clip[:70]}   (expects: {expect})")
    print("=" * 90)
    print(f"  events raised: {evs}")
    for k in sorted(gates):
        print(f"    {k:52s} {gates[k]}")
    if peak_speeds:
        a = np.array(peak_speeds)
        print(f"    peak speed observed  : max {a.max():.2f}  p95 {np.percentile(a,95):.2f}  "
              f"(needs >= {throw.RELEASE_SPEED})")
    if dxs:
        a = np.array(dxs)
        print(f"    horizontal travel    : max {a.max():.2f}  (needs >= {throw.MIN_HORIZONTAL})")
    if flips:
        a = np.array(flips)
        print(f"    aspect inversions    : max {int(a.max())}  "
              f"(needs >= {getattr(roll,'MIN_INVERSIONS',2)})")
    print()


if __name__ == "__main__":
    items = sys.argv[1:]
    if items:
        for c in items:
            diagnose(c, CLIPS.get(c, "?"))
    else:
        for c, e in CLIPS.items():
            diagnose(c, e)
