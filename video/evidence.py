"""
Evidence Generator & Frame Visualiser.

Produces the three artefacts a supervisor needs to act on an alert:
the live annotated stream, a still evidence frame carrying the reasoning, and a
short replay clip around the event.

Responsible-AI: evidence frames are captioned "POTENTIAL DAMAGE RISK", never
"damaged", and optional face blurring is on by default so a stored incident
documents the *handling step* rather than identifying an individual.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import cv2
from video.encoder import BrowserVideoWriter
import numpy as np

import config
from behaviour.base import RiskLevel
from detection.object_classes import ENTITY_COLORS, WarehouseEntity

logger = logging.getLogger(__name__)

RISK_COLORS = {
    RiskLevel.LOW: (0, 200, 0),
    RiskLevel.MEDIUM: (0, 215, 255),
    RiskLevel.HIGH: (0, 120, 255),
    RiskLevel.CRITICAL: (0, 0, 230),
}

_FACE_CASCADE: Optional[cv2.CascadeClassifier] = None


def _face_cascade() -> Optional[cv2.CascadeClassifier]:
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        cascade = cv2.CascadeClassifier(path)
        _FACE_CASCADE = cascade if not cascade.empty() else None
        if _FACE_CASCADE is None:
            logger.warning("Face cascade unavailable; evidence frames will not be blurred")
    return _FACE_CASCADE


def blur_faces(frame: np.ndarray) -> np.ndarray:
    """Blur detected faces in place-safe fashion for privacy-preserving evidence."""
    cascade = _face_cascade()
    if cascade is None:
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(24, 24))
    for (x, y, w, h) in faces:
        roi = frame[y : y + h, x : x + w]
        if roi.size:
            k = max(9, (w // 4) * 2 + 1)
            frame[y : y + h, x : x + w] = cv2.GaussianBlur(roi, (k, k), 0)
    return frame


def draw_hud_overlay(
    frame: np.ndarray,
    frame_idx: int,
    timestamp: float,
    fps: float,
    active_tracks_count: int,
    incidents_count: int,
) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    hud = (
        f"VISIONGUARD | T {timestamp:6.2f}s  F#{frame_idx}  "
        f"TRACKS {active_tracks_count}  RISK EVENTS {incidents_count}"
    )
    cv2.putText(
        frame, hud, (18, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 230, 242), 2, cv2.LINE_AA
    )
    return frame


def draw_track_annotations(frame: np.ndarray, tracked_objects: List[Any]) -> np.ndarray:
    """Draw boxes, persistent track IDs, entity class and current motion state."""
    for trk in tracked_objects:
        if getattr(trk, "consecutive_lost", 0) > 0 or getattr(trk, "hits", 0) < 2:
            continue
        box = [int(b) for b in trk.box]
        color = ENTITY_COLORS.get(trk.entity_type, (200, 200, 200))
        thickness = 2 if trk.entity_type is not WarehouseEntity.OPERATOR else 2
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, thickness)

        state = getattr(trk, "state", None)
        label = f"#{trk.track_id} {trk.entity_type.value.upper()}"
        if state is not None and state.value not in ("stationary",):
            label += f" [{state.value.upper()}]"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        ly = max(lh + 4, box[1])
        cv2.rectangle(frame, (box[0], ly - lh - 6), (box[0] + lw + 8, ly), color, -1)
        cv2.putText(
            frame, label, (box[0] + 4, ly - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
            (20, 20, 20), 1, cv2.LINE_AA,
        )

        # Velocity vector (normalised units scaled for visibility).
        if abs(trk.vx) > 0.05 or abs(trk.vy) > 0.05:
            cx, cy = int(trk.center[0]), int(trk.center[1])
            scale = frame.shape[0] * 0.25
            tip = (
                int(cx + float(np.clip(trk.vx * scale, -70, 70))),
                int(cy + float(np.clip(trk.vy * scale, -70, 70))),
            )
            cv2.arrowedLine(frame, (cx, cy), tip, (0, 255, 255), 2, tipLength=0.3)
    return frame


def _wrap(text: str, width_chars: int, max_lines: int) -> List[str]:
    words, lines, cur = text.split(), [], ""
    for word in words:
        candidate = f"{cur} {word}".strip()
        if len(candidate) <= width_chars:
            cur = candidate
        else:
            lines.append(cur)
            cur = word
            if len(lines) == max_lines:
                break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    return lines[:max_lines]


def create_evidence_snapshot(
    raw_frame: np.ndarray,
    incident_data: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> str:
    """Annotated still: highlighted subject, risk header, finding and action."""
    output_dir = output_dir or config.EVIDENCE_DIR
    os.makedirs(output_dir, exist_ok=True)
    annotated = raw_frame.copy()
    if config.BLUR_FACES_IN_EVIDENCE:
        annotated = blur_faces(annotated)
    h, w = annotated.shape[:2]

    box = [int(b) for b in incident_data["bounding_box"]]
    box = [
        max(0, min(box[0], w - 1)), max(0, min(box[1], h - 1)),
        max(0, min(box[2], w - 1)), max(0, min(box[3], h - 1)),
    ]
    try:
        risk_level = RiskLevel(incident_data["risk_level"])
    except ValueError:
        risk_level = RiskLevel.MEDIUM
    risk_color = RISK_COLORS.get(risk_level, (0, 0, 255))

    cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), risk_color, 3)
    c = 18
    for (px, py, dx, dy) in (
        (box[0], box[1], 1, 1), (box[2], box[1], -1, 1),
        (box[0], box[3], 1, -1), (box[2], box[3], -1, -1),
    ):
        cv2.line(annotated, (px, py), (px + dx * c, py), (255, 255, 255), 2)
        cv2.line(annotated, (px, py), (px, py + dy * c), (255, 255, 255), 2)

    # Header: risk tier + behaviour + time. Wording stays at "potential risk".
    cv2.rectangle(annotated, (0, 0), (w, 58), (15, 18, 24), -1)
    cv2.rectangle(annotated, (0, 55), (w, 58), risk_color, -1)
    behaviour = incident_data["behaviour_type"].replace("_", " ").upper()
    header = (
        f"POTENTIAL DAMAGE RISK | {risk_level.value} | {behaviour} | "
        f"T {incident_data['timestamp_sec']:.2f}s | SCORE {incident_data.get('risk_score', 0):.0f}/100"
    )
    cv2.putText(
        annotated, header, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA
    )

    # Footer: observed finding + recommended action + review reminder.
    footer_h = 92
    cv2.rectangle(annotated, (0, h - footer_h), (w, h), (15, 18, 24), -1)
    chars = max(60, int(w / 8.2))
    y = h - footer_h + 20
    for line in _wrap("OBSERVED: " + incident_data["evidence_description"], chars, 2):
        cv2.putText(annotated, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 220, 240), 1, cv2.LINE_AA)
        y += 18
    for line in _wrap("ACTION: " + incident_data["recommended_action"], chars, 1):
        cv2.putText(annotated, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 220, 160), 1, cv2.LINE_AA)
        y += 17
    cv2.putText(
        annotated, "Requires human review before any corrective decision.",
        (18, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 160, 175), 1, cv2.LINE_AA,
    )

    filepath = os.path.join(output_dir, f"evidence_{incident_data['id']}.jpg")
    cv2.imwrite(filepath, annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return filepath


def write_incident_clip(
    source_video: str,
    timestamp_sec: float,
    out_path: str,
    pre_sec: float = 2.0,
    post_sec: float = 2.0,
) -> Optional[str]:
    """Extract a short replay clip centred on an incident."""
    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        return None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not np.isfinite(fps) or fps <= 1.0:
            fps = 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if w <= 0 or h <= 0:
            return None

        start = max(0, int((timestamp_sec - pre_sec) * fps))
        end = int((timestamp_sec + post_sec) * fps)
        if total > 0:
            end = min(end, total - 1)
        if end <= start:
            return None

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # Replay clips are played back in the dashboard, so they must be H.264;
        # OpenCV's mp4v output renders as a black <video> in every browser.
        writer = BrowserVideoWriter(out_path, fps, w, h)
        if not writer.isOpened():
            return None
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for _ in range(end - start + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
        finally:
            writer.release()
        return out_path if os.path.exists(out_path) and os.path.getsize(out_path) > 0 else None
    finally:
        cap.release()
