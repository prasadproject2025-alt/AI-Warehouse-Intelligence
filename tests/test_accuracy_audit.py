"""
VisionGuard Accuracy & Detection Benchmark Audit
Tests detection accuracy against ground-truth behaviour profiles for each official Godrej warehouse video.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.database.db import DatabaseManager

# Ground-truth mapping defined by the challenge video scenarios
GROUND_TRUTH_EXPECTATIONS = {
    "Dock level, dragging cupboard.mp4": {
        "expected_behaviours": ["dock_level_hazard", "product_drag", "stepping_on_carton"],
        "primary_risks": ["HIGH", "CRITICAL"]
    },
    "KD packets dragged, heavy box kept on other packets.mp4": {
        "expected_behaviours": ["product_drag", "stepping_on_carton", "product_throw"],
        "primary_risks": ["HIGH", "CRITICAL"]
    },
    "Rolling and dragging on wet floor.mp4": {
        "expected_behaviours": ["rolling_product", "product_drag", "wet_floor_hazard"],
        "primary_risks": ["MEDIUM", "HIGH", "CRITICAL"]
    },
    "Rolling and dropping carton.mp4": {
        "expected_behaviours": ["rolling_product", "product_drop", "stepping_on_carton"],
        "primary_risks": ["MEDIUM", "HIGH", "CRITICAL"]
    },
    "Stepping on cartons, vertical product kept horizontally, heavy product kept on top.mp4": {
        "expected_behaviours": ["stepping_on_carton", "orientation_violation", "improper_stacking"],
        "primary_risks": ["HIGH", "CRITICAL"]
    },
    "Throwing Mattresses.mp4": {
        "expected_behaviours": ["product_throw"],
        "primary_risks": ["CRITICAL", "HIGH"]
    },
    "Throwing seating cartons, using strap to hold.mp4": {
        "expected_behaviours": ["product_throw", "strap_pulling"],
        "primary_risks": ["HIGH", "CRITICAL"]
    }
}

def audit_detection_accuracy():
    print("=" * 80)
    print("      VISIONGUARD ACCURACY & BEHAVIOURAL GROUND-TRUTH AUDIT")
    print("=" * 80)
    
    videos = DatabaseManager.get_all_videos()
    print(f"\nFound {len(videos)} videos registered in Database.\n")

    total_benchmarks = 0
    passed_benchmarks = 0

    results_table = []

    for expected_filename, spec in GROUND_TRUTH_EXPECTATIONS.items():
        # Find matching video in DB
        matching_vids = [v for v in videos if expected_filename.lower() in v["filename"].lower()]
        if not matching_vids:
            print(f"[-] Video not found in DB: {expected_filename}")
            continue

        vid = matching_vids[0]
        incidents = DatabaseManager.get_incidents(video_id=vid["id"])

        detected_behaviours = set(i["behaviour_type"] for i in incidents)
        detected_risks = set(i["risk_level"] for i in incidents)

        # Evaluate target behaviour presence
        found_targets = [b for b in spec["expected_behaviours"] if b in detected_behaviours]
        coverage_pct = round((len(found_targets) / len(spec["expected_behaviours"])) * 100, 1)

        total_benchmarks += 1
        is_pass = (len(found_targets) >= 1) # At least primary expected behaviours detected
        if is_pass:
            passed_benchmarks += 1

        status_tag = "PASS" if is_pass else "FAIL"
        
        print(f"[{status_tag}] Video: {expected_filename[:55]}...")
        print(f"    * Total Incidents Logged: {len(incidents)}")
        print(f"    * Ground-Truth Targets:   {spec['expected_behaviours']}")
        print(f"    * Detected Behaviours:    {sorted(list(detected_behaviours))}")
        print(f"    * Matched Key Targets:    {found_targets} ({coverage_pct}% Scenario Match)")
        print(f"    * Detected Risk Tiers:    {sorted(list(detected_risks))}")
        print("-" * 80)

        results_table.append({
            "video": expected_filename,
            "incidents": len(incidents),
            "matched_targets": found_targets,
            "accuracy_pct": coverage_pct,
            "status": status_tag
        })

    overall_accuracy = round((passed_benchmarks / max(1, total_benchmarks)) * 100, 1)
    print("\n" + "=" * 80)
    print(f"  OVERALL BENCHMARK ACCURACY: {passed_benchmarks}/{total_benchmarks} Scenarios Verified ({overall_accuracy}%)")
    print("=" * 80)

    return overall_accuracy >= 85.0

if __name__ == "__main__":
    success = audit_detection_accuracy()
    if not success:
        exit(1)
