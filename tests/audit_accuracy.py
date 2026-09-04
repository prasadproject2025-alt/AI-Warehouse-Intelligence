"""
Ground-truth accuracy audit for the GEG pilot footage.

This reports what the system actually detected against what each clip is
labelled as containing, including the misses. It is deliberately not a pass/fail
gate that can be satisfied by loosening detectors: undetected ground-truth
behaviours are printed as misses and counted against recall.

Run after process_all_pilot_videos.py:

    python -m tests.audit_accuracy
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.database.db import DatabaseManager  # noqa: E402

#: What each pilot clip is labelled as showing, from the clip titles and the
#: burnt-in captions in the source footage.
GROUND_TRUTH: Dict[str, List[str]] = {
    "Dock level, dragging cupboard.mp4": ["dock_level_hazard", "product_drag"],
    "KD packets dragged, heavy box kept on other packets.mp4": [
        "product_drag", "improper_stacking",
    ],
    "Rolling and dragging on wet floor.mp4": [
        "rolling_product", "product_drag", "wet_floor_hazard",
    ],
    "Rolling and dropping carton.mp4": ["rolling_product", "product_drop"],
    "Stepping on cartons, vertical product kept horizontally, heavy product kept on top.mp4": [
        "stepping_on_carton", "orientation_violation", "improper_stacking",
    ],
    "Throwing Mattresses.mp4": ["product_throw"],
    "Throwing seating cartons, using strap to hold.mp4": ["product_throw"],
}


def audit() -> int:
    videos = DatabaseManager.get_all_videos()
    if not videos:
        print("No analysed videos in the database. Run process_all_pilot_videos.py first.")
        return 1

    print("=" * 100)
    print("  VISIONGUARD GROUND-TRUTH AUDIT — pilot footage")
    print("=" * 100)
    print()

    tp = fn = 0
    extra_by_behaviour: Dict[str, int] = {}
    rows = []

    for filename, expected in GROUND_TRUTH.items():
        match = [v for v in videos if v["filename"] == filename]
        if not match:
            print(f"[SKIP]  not analysed: {filename}")
            continue
        video = match[0]
        incidents = DatabaseManager.get_incidents(video_id=video["id"], limit=1000)
        detected = {i["behaviour_type"] for i in incidents}

        hits = [b for b in expected if b in detected]
        misses = [b for b in expected if b not in detected]
        extra = sorted(detected - set(expected))
        for b in extra:
            extra_by_behaviour[b] = extra_by_behaviour.get(b, 0) + 1

        tp += len(hits)
        fn += len(misses)
        recall = len(hits) / len(expected) * 100 if expected else 0.0

        print(f"{filename[:78]}")
        print(f"   expected      : {expected}")
        print(f"   detected      : {sorted(detected) or '(none)'}")
        print(f"   matched       : {hits}  -> scenario recall {recall:.0f}%")
        if misses:
            print(f"   MISSED        : {misses}")
        if extra:
            print(f"   additional    : {extra}  (not in the clip label; may still be real)")
        print(f"   events logged : {len(incidents)}")
        print("-" * 100)

        rows.append({
            "video": filename, "expected": expected, "hits": hits,
            "misses": misses, "extra": extra, "events": len(incidents),
        })

    total_expected = tp + fn
    recall = (tp / total_expected * 100) if total_expected else 0.0

    print()
    print("=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    print(f"  Ground-truth behaviours across the pilot set : {total_expected}")
    print(f"  Detected                                     : {tp}")
    print(f"  Missed                                       : {fn}")
    print(f"  Behaviour-level recall                       : {recall:.1f}%")
    print()
    print("  Behaviours reported beyond the clip labels (each needs review;")
    print("  a clip labelled 'dragging' can genuinely also contain other behaviours):")
    if extra_by_behaviour:
        for b, c in sorted(extra_by_behaviour.items(), key=lambda kv: -kv[1]):
            print(f"    - {b}: in {c} clip(s)")
    else:
        print("    (none)")
    print()
    print("  NOTE: recall here is limited primarily by product detection, not by the")
    print("  behaviour logic. See tests/audit_perception.py for the measured product")
    print("  detection rate per clip.")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(audit())
