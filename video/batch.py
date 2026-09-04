"""
Batch analysis of a whole dataset.

Lets the dashboard run every video in the library (or a chosen subset) as one
tracked job, then scope the analytics, prevention and coverage pages to exactly
that run. Without this, the pages aggregate everything ever stored, so figures
from an old experiment mix with the dataset the user actually cares about.

Each run gets a ``batch_id`` that is stamped onto every video and incident it
produces, which is what makes the scoping possible.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import config
from backend.database.db import DatabaseManager
from behaviour.behaviour_engine import SceneContext

logger = logging.getLogger(__name__)

#: Live progress for batch runs, keyed by batch id.
BATCH_STATUS: Dict[str, Dict[str, Any]] = {}

_lock = threading.Lock()


def _scene_for(filename: str, overrides: Optional[Dict[str, Any]] = None) -> SceneContext:
    """
    Scene context for a library video.

    Known pilot clips carry their declared installation context; anything else
    falls back to explicit "unassigned" values rather than inventing a bay from
    the filename. Guessing scene facts from a file name is exactly the shortcut
    this project removed.
    """
    from process_all_pilot_videos import PILOT_VIDEOS

    if overrides:
        return SceneContext(
            bay=overrides.get("bay") or "Unassigned Bay",
            shift=overrides.get("shift") or "Unassigned Shift",
            camera_id=overrides.get("camera_id") or "CAM-01",
            floor_condition=overrides.get("floor_condition") or "unknown",
            dock_transfer=bool(overrides.get("dock_transfer")),
        )
    for name, scene in PILOT_VIDEOS:
        if name == filename:
            return scene
    return SceneContext()


def library_videos() -> List[str]:
    """Video files currently present in the raw directory."""
    if not os.path.isdir(config.RAW_VIDEOS_DIR):
        return []
    return sorted(
        f
        for f in os.listdir(config.RAW_VIDEOS_DIR)
        if os.path.splitext(f)[1].lower() in config.ALLOWED_VIDEO_EXTENSIONS
    )


def start_batch(
    processor,
    filenames: List[str],
    replace_existing: bool = True,
    scene_overrides: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Queue a batch run and return its id immediately.

    ``replace_existing`` clears previous non-live analysis first, which is what
    makes a run reproducible: without it, re-running accumulates duplicate rows
    for the same footage.
    """
    with _lock:
        running = [
            b for b in BATCH_STATUS.values() if b["status"] in ("queued", "running")
        ]
        if running:
            raise RuntimeError(
                f"Batch {running[0]['batch_id']} is already running "
                f"({running[0]['completed']}/{running[0]['total']} videos)"
            )

        batch_id = f"batch_{uuid.uuid4().hex[:8]}"
        BATCH_STATUS[batch_id] = {
            "batch_id": batch_id,
            "status": "queued",
            "total": len(filenames),
            "completed": 0,
            "failed": 0,
            "incidents": 0,
            "current": None,
            "started_at": time.time(),
            "elapsed_sec": 0.0,
            "errors": [],
            "replace_existing": replace_existing,
        }

    thread = threading.Thread(
        target=_run_batch,
        args=(processor, batch_id, filenames, replace_existing, scene_overrides),
        name=f"batch-{batch_id}",
        daemon=True,
    )
    thread.start()
    return batch_id


def _run_batch(
    processor,
    batch_id: str,
    filenames: List[str],
    replace_existing: bool,
    scene_overrides: Optional[Dict[str, Any]],
) -> None:
    state = BATCH_STATUS[batch_id]
    state["status"] = "running"
    started = time.time()

    try:
        if replace_existing:
            removed = DatabaseManager.clear_analysis()
            logger.info("Batch %s cleared %d previous video row(s)", batch_id, removed)

        for name in filenames:
            if state.get("cancel"):
                state["status"] = "cancelled"
                break

            path = os.path.join(config.RAW_VIDEOS_DIR, name)
            state["current"] = name
            state["elapsed_sec"] = round(time.time() - started, 1)

            if not os.path.exists(path):
                state["failed"] += 1
                state["errors"].append({"video": name, "error": "file not found"})
                continue

            try:
                result = processor.process_video(
                    path,
                    scene=_scene_for(name, scene_overrides),
                    batch_id=batch_id,
                )
                state["incidents"] += result.get("incidents_count", 0)
                state["completed"] += 1
                logger.info(
                    "Batch %s: %s -> %d event(s)",
                    batch_id, name, result.get("incidents_count", 0),
                )
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
                logger.exception("Batch %s failed on %s", batch_id, name)
                state["failed"] += 1
                state["errors"].append({"video": name, "error": str(exc)})

        if state["status"] != "cancelled":
            state["status"] = "completed"
    except Exception as exc:  # noqa: BLE001
        logger.exception("Batch %s aborted", batch_id)
        state["status"] = "error"
        state["errors"].append({"video": None, "error": str(exc)})
    finally:
        state["current"] = None
        state["elapsed_sec"] = round(time.time() - started, 1)


def get_status(batch_id: str) -> Optional[Dict[str, Any]]:
    state = BATCH_STATUS.get(batch_id)
    if state and state["status"] == "running":
        state = dict(state)
        state["elapsed_sec"] = round(time.time() - state["started_at"], 1)
    return state


def active_batch() -> Optional[Dict[str, Any]]:
    for state in BATCH_STATUS.values():
        if state["status"] in ("queued", "running"):
            return get_status(state["batch_id"])
    return None


def cancel(batch_id: str) -> bool:
    state = BATCH_STATUS.get(batch_id)
    if not state or state["status"] not in ("queued", "running"):
        return False
    state["cancel"] = True
    return True
