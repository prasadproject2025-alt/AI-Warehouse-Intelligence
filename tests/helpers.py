"""
Synthetic track builders and database seeders shared by the tests.

Kept out of conftest.py so the test modules can import them by name without
colliding with an unrelated ``tests`` package that may exist on sys.path.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.database.db import DatabaseManager
from detection.detector import Detection
from detection.object_classes import WarehouseEntity
from detection.tracker import TrackedObject

FRAME_W, FRAME_H = 1280.0, 720.0


def make_detection(
    cx: float, cy: float, w: float, h: float, entity: WarehouseEntity, conf: float = 0.8
) -> Detection:
    return Detection(
        box=[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
        confidence=conf,
        raw_class=entity.value,
        entity_type=entity,
    )


def build_track(
    positions: List[tuple],
    entity: WarehouseEntity = WarehouseEntity.CARTON,
    size: tuple = (140.0, 120.0),
    fps: float = 10.0,
    track_id: int = 1,
) -> TrackedObject:
    """
    Build a TrackedObject by feeding it a scripted path.

    ``positions`` is a list of (cx, cy) pixel centres, one per analysed frame.
    Sizes may be overridden per step with (cx, cy, w, h).
    """
    w, h = size
    first = positions[0]
    det = make_detection(first[0], first[1], first[2] if len(first) > 2 else w,
                         first[3] if len(first) > 3 else h, entity)
    trk = TrackedObject(track_id, det, 0, 0.0, FRAME_H, FRAME_W)
    for i, pos in enumerate(positions[1:], start=1):
        pw = pos[2] if len(pos) > 2 else w
        ph = pos[3] if len(pos) > 3 else h
        trk.update(make_detection(pos[0], pos[1], pw, ph, entity), i, i / fps, fps)
    return trk


def base_context(
    ground_plane=None,
    recurrence: Optional[dict] = None,
    **overrides,
):
    """Minimal reasoning context matching what BehaviourEngine supplies."""
    from behaviour.kinematic_detectors import SceneScale

    ctx = {
        "floor_line_norm": 0.85,
        "ground_plane": ground_plane,
        "ground_plane_residual": 0.02,
        "scale": SceneScale(),
        "recurrence": recurrence or {},
        "bay": "Test Bay",
        "shift": "Shift A",
        "camera_id": "CAM-TEST",
        "wet_floor_active": False,
        "floor_condition_source": "test",
        "dock_transfer_active": False,
        "staging_zone": None,
        "handling_equipment_present": False,
        "vehicle_detected": False,
    }
    ctx.update(overrides)
    return ctx


def seed_incident(**overrides) -> dict:
    """Insert one realistic incident row and return it."""
    inc = {
        "id": overrides.get("id", "inc_test01"),
        "video_id": overrides.get("video_id", "vid_test01"),
        "timestamp_sec": overrides.get("timestamp_sec", 4.5),
        "frame_idx": overrides.get("frame_idx", 135),
        "behaviour_type": overrides.get("behaviour_type", "product_drop"),
        "object_track_id": 7,
        "operator_track_id": 3,
        "confidence": overrides.get("confidence", 0.82),
        "risk_level": overrides.get("risk_level", "HIGH"),
        "risk_score": overrides.get("risk_score", 74.0),
        "evidence_description": overrides.get("evidence_description", "Carton #7 descended rapidly."),
        "root_cause": "Package released instead of lowered.",
        "recommended_action": "[HIGH PRIORITY] Inspect the package.",
        "bounding_box": [10.0, 20.0, 110.0, 140.0],
        "risk_factors": [{"name": "Drop height", "points": 18.0, "detail": "approximately 1.1 m"}],
        "evidence_stages": [{"stage": "carried", "at_sec": 4.1}, {"stage": "falling", "at_sec": 4.4}],
        "evidence_tier": overrides.get("evidence_tier", "POTENTIAL_RISK"),
        "camera_id": overrides.get("camera_id", "CAM-TEST"),
        "bay": overrides.get("bay", "Dock 05"),
        "shift": overrides.get("shift", "Shift A"),
        "review_status": overrides.get("review_status", "PENDING_REVIEW"),
        "duration_sec": 0.6,
        "evidence_image_path": overrides.get("evidence_image_path", "/x/evidence_inc_test01.jpg"),
    }
    DatabaseManager.save_incident(inc)
    return inc


def seed_video(video_id: str = "vid_test01", **overrides) -> None:
    DatabaseManager.save_video(
        video_id=video_id,
        filename=overrides.get("filename", "test_clip.mp4"),
        filepath=overrides.get("filepath", "/x/test_clip.mp4"),
        duration=overrides.get("duration", 30.0),
        fps=30.0,
        frame_count=900,
        width=1280,
        height=720,
        status="completed",
        bay=overrides.get("bay", "Dock 05"),
        shift=overrides.get("shift", "Shift A"),
        camera_id=overrides.get("camera_id", "CAM-TEST"),
    )
