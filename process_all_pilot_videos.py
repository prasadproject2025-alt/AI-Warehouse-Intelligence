"""
Re-analyse the GEG pilot videos from scratch.

Wipes previous analysis rows (so repeated runs never accumulate duplicates) and
processes each pilot clip with its declared scene context. Scene context is
supplied here explicitly rather than being guessed from the filename, which is
how a real deployment works: the bay, shift and floor condition belong to the
camera installation, not to what a file happens to be called.

Usage:  python process_all_pilot_videos.py [--keep-existing]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from backend.database.db import DatabaseManager, get_connection, init_db
from behaviour.behaviour_engine import SceneContext
from video.processor import VideoProcessor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("pilot")

# Declared installation context for each pilot clip.
PILOT_VIDEOS = [
    (
        "Dock level, dragging cupboard.mp4",
        SceneContext(bay="Dock 09 - Inside", shift="Shift A", camera_id="CAM-D09",
                     floor_condition="dry", dock_transfer=True),
    ),
    (
        "KD packets dragged, heavy box kept on other packets.mp4",
        SceneContext(bay="Dock 11 - Staging", shift="Shift A", camera_id="CAM-D11",
                     floor_condition="dry", dock_transfer=False),
    ),
    (
        "Rolling and dragging on wet floor.mp4",
        SceneContext(bay="Dock 07 - Outside", shift="Shift B", camera_id="CAM-D07",
                     floor_condition="wet", dock_transfer=False),
    ),
    (
        "Rolling and dropping carton.mp4",
        SceneContext(bay="Dock 05 - Inside", shift="Shift B", camera_id="CAM-D05",
                     floor_condition="dry", dock_transfer=False),
    ),
    (
        "Stepping on cartons, vertical product kept horizontally, heavy product kept on top.mp4",
        SceneContext(bay="Dock 03 - Loading", shift="Shift A", camera_id="CAM-D03",
                     floor_condition="dry", dock_transfer=False),
    ),
    (
        "Throwing Mattresses.mp4",
        SceneContext(bay="Dock 10 - Outside", shift="Shift C", camera_id="CAM-D10",
                     floor_condition="dry", dock_transfer=True),
    ),
    (
        "Throwing seating cartons, using strap to hold.mp4",
        SceneContext(bay="Dock 12 - Loading", shift="Shift C", camera_id="CAM-D12",
                     floor_condition="dry", dock_transfer=False),
    ),
]


def reset_analysis() -> None:
    """Remove all previous analysis rows so results reflect this run only."""
    with get_connection() as conn:
        conn.execute("DELETE FROM incidents")
        conn.execute("DELETE FROM videos")
        conn.commit()
    logger.info("Cleared previous analysis rows")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-existing", action="store_true",
                        help="Do not wipe previous analysis rows first")
    parser.add_argument("--raw-dir", default="data/raw")
    args = parser.parse_args()

    init_db()
    if not args.keep_existing:
        reset_analysis()

    processor = VideoProcessor()
    logger.info("Detector backend: %s", processor.detector.backend)

    total_incidents = 0
    t0 = time.time()
    for idx, (filename, scene) in enumerate(PILOT_VIDEOS, 1):
        path = os.path.join(args.raw_dir, filename)
        if not os.path.exists(path):
            logger.warning("[%d/%d] MISSING: %s", idx, len(PILOT_VIDEOS), filename)
            continue
        logger.info("[%d/%d] %s (bay=%s shift=%s)", idx, len(PILOT_VIDEOS),
                    filename, scene.bay, scene.shift)
        try:
            result = processor.process_video(path, generate_annotated_video=True, scene=scene)
        except Exception:
            logger.exception("Analysis failed for %s", filename)
            continue
        total_incidents += result["incidents_count"]
        logger.info(
            "    -> %d risk events, %.1fs (%.2fx realtime)",
            result["incidents_count"], result["processing_seconds"], result["realtime_ratio"],
        )

    summary = DatabaseManager.get_analytics_summary()
    logger.info("=" * 70)
    logger.info("Processed %d videos in %.1fs", summary["total_videos_analyzed"], time.time() - t0)
    logger.info("Total risk events: %d", total_incidents)
    logger.info("Risk breakdown: %s", summary["risk_breakdown"])
    logger.info("Behaviours observed: %s", summary["top_behaviours"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
