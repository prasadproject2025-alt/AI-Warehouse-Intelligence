"""
Detector benchmark.

Product detection is the binding constraint on behaviour recall, so this script
measures competing detector configurations on identical sampled frames from the
pilot clips and reports which actually finds more product.

Metric: the fraction of sampled frames containing at least one product
detection. The clips show product handling throughout, so a higher rate means
better recall. Speed is reported because the target is CPU inference.

    python tests/bench_detectors.py
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

CLIPS = [
    "Rolling and dropping carton.mp4",
    "KD packets dragged, heavy box kept on other packets.mp4",
    "Dock level, dragging cupboard.mp4",
    "Throwing seating cartons, using strap to hold.mp4",
    "Stepping on cartons, vertical product kept horizontally, heavy product kept on top.mp4",
    "Rolling and dragging on wet floor.mp4",
    "Throwing Mattresses.mp4",
]

# Nouns describing handled goods. Person is included so the prompt has a
# well-detected anchor class, but it is not counted as product.
PRODUCT_WORDS = ["box", "carton", "package", "mattress", "pallet", "trolley", "cupboard", "furniture"]
VOCAB = ["person"] + PRODUCT_WORDS

SAMPLES = 20


def sample_frames(path, n=SAMPLES):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    idxs = np.linspace(0, total - 1, min(n, total)).astype(int)
    out = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            out.append(f)
    cap.release()
    return out


def letterbox_crop(frame):
    """
    Crop away static screen furniture around the CCTV panel.

    The pilot clips are phone recordings of a monitor, so the real footage sits
    inside application chrome. Cropping to the busy centre and upscaling raises
    the effective resolution of the content that matters.
    """
    h, w = frame.shape[:2]
    x0, x1 = int(w * 0.04), int(w * 0.99)
    y0, y1 = int(h * 0.16), int(h * 0.88)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return frame
    return cv2.resize(crop, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)


def run_world(model, frames, imgsz, conf, crop=False):
    """Return (product_hit_rate, mean_conf, mean_ms)."""
    hits, confs, times = 0, [], []
    for f in frames:
        img = letterbox_crop(f) if crop else f
        t = time.time()
        r = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
        times.append((time.time() - t) * 1000)
        found = []
        for b in r.boxes:
            name = model.names[int(b.cls[0])]
            if name in PRODUCT_WORDS:
                found.append(float(b.conf[0]))
        if found:
            hits += 1
            confs.append(max(found))
    n = max(1, len(frames))
    return hits / n, (float(np.mean(confs)) if confs else 0.0), float(np.mean(times))


def bench_world(weights, imgsz, conf, crop=False, label=None):
    from ultralytics import YOLOWorld

    model = YOLOWorld(weights)
    model.set_classes(VOCAB)
    name = label or f"{weights} imgsz={imgsz} conf={conf}{' +crop' if crop else ''}"

    rates, cs, ms = [], [], []
    print(f"\n--- {name}")
    for clip in CLIPS:
        path = os.path.join(RAW, clip)
        if not os.path.exists(path):
            continue
        frames = sample_frames(path)
        rate, c, t = run_world(model, frames, imgsz, conf, crop)
        rates.append(rate)
        cs.append(c)
        ms.append(t)
        print(f"    {clip[:44]:46s} product {rate*100:5.0f}%  conf {c:.2f}  {t:5.0f}ms")
    mean_rate = float(np.mean(rates)) if rates else 0.0
    print(f"    {'MEAN':46s} product {mean_rate*100:5.0f}%  conf {np.mean(cs):.2f}  {np.mean(ms):5.0f}ms")
    return name, mean_rate, float(np.mean(ms))


def main():
    print("=" * 96)
    print("  DETECTOR BENCHMARK - product detection rate on the pilot clips")
    print("=" * 96)
    print(f"  {SAMPLES} frames sampled per clip · vocabulary: {VOCAB}")

    results = []
    # Baseline: what the system ships with today.
    results.append(bench_world("yolov8s-worldv2.pt", 640, 0.12, label="BASELINE  small @640"))
    # Does more inference resolution help on its own?
    results.append(bench_world("yolov8s-worldv2.pt", 1280, 0.12, label="small @1280"))
    # Does cropping the monitor chrome help?
    results.append(bench_world("yolov8s-worldv2.pt", 1280, 0.12, crop=True, label="small @1280 + ROI crop"))
    # Does a larger backbone help?
    results.append(bench_world("yolov8x-worldv2.pt", 1280, 0.12, label="xlarge @1280"))
    results.append(bench_world("yolov8x-worldv2.pt", 1280, 0.12, crop=True, label="xlarge @1280 + ROI crop"))

    print("\n" + "=" * 96)
    print("  RANKING (by product detection rate)")
    print("=" * 96)
    for name, rate, ms in sorted(results, key=lambda r: -r[1]):
        print(f"  {rate*100:5.0f}%   {ms:6.0f} ms/frame   {name}")
    print("=" * 96)


if __name__ == "__main__":
    main()
