"""
Evidence Generator & Frame Visualizer
Generates annotated evidence snapshots with risk banners, timestamps, and target bounding box highlights.
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Any, Optional
from detection.object_classes import ENTITY_COLORS, WarehouseEntity
from behaviour.base import RiskLevel

RISK_COLORS = {
    RiskLevel.LOW: (0, 200, 0),       # Green
    RiskLevel.MEDIUM: (0, 215, 255),   # Yellow/Amber
    RiskLevel.HIGH: (0, 120, 255),     # Orange
    RiskLevel.CRITICAL: (0, 0, 230)    # Red
}

def draw_hud_overlay(
    frame: np.ndarray,
    frame_idx: int,
    timestamp: float,
    fps: float,
    active_tracks_count: int,
    incidents_count: int
) -> np.ndarray:
    """
    Draw clean industrial HUD at top-left.
    """
    h, w = frame.shape[:2]
    # Header bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Title & Telemetry
    hud_text = f"VISIONGUARD AI | WAREHOUSE OPS | T: {timestamp:.2f}s (F#{frame_idx}) | TRACKS: {active_tracks_count} | INCIDENTS: {incidents_count}"
    cv2.putText(frame, hud_text, (18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 230, 242), 2, cv2.LINE_AA)
    return frame

def draw_track_annotations(frame: np.ndarray, tracked_objects: list) -> np.ndarray:
    """
    Draw bounding boxes, entity labels, and persistent track IDs.
    """
    for trk in tracked_objects:
        box = [int(b) for b in trk.box]
        color = ENTITY_COLORS.get(trk.entity_type, (200, 200, 200))
        
        # Bounding box
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        
        # Label badge
        label = f"#{trk.track_id} {trk.entity_type.value.upper()}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (box[0], max(0, box[1] - 20)), (box[0] + lw + 8, box[1]), color, -1)
        cv2.putText(frame, label, (box[0] + 4, max(12, box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)

        # Velocity indicator arrow if moving
        if abs(trk.vx) > 15 or abs(trk.vy) > 15:
            cx, cy = int(trk.center[0]), int(trk.center[1])
            tip_x = int(cx + np.clip(trk.vx * 0.15, -40, 40))
            tip_y = int(cy + np.clip(trk.vy * 0.15, -40, 40))
            cv2.arrowedLine(frame, (cx, cy), (tip_x, tip_y), (0, 255, 255), 2, tipLength=0.3)

    return frame

def create_evidence_snapshot(
    raw_frame: np.ndarray,
    incident_data: Dict[str, Any],
    output_dir: str = "data/evidence"
) -> str:
    """
    Annotate incident evidence frame with highlight box, risk banner, and root cause.
    """
    os.makedirs(output_dir, exist_ok=True)
    annotated = raw_frame.copy()
    h, w = annotated.shape[:2]

    box = [int(b) for b in incident_data["bounding_box"]]
    risk_level = RiskLevel(incident_data["risk_level"])
    risk_color = RISK_COLORS.get(risk_level, (0, 0, 255))

    # Highlight box with pulse glow
    cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), risk_color, 3)
    
    # Corner brackets for tactical look
    c_len = 16
    cv2.line(annotated, (box[0], box[1]), (box[0] + c_len, box[1]), (255, 255, 255), 2)
    cv2.line(annotated, (box[0], box[1]), (box[0], box[1] + c_len), (255, 255, 255), 2)
    cv2.line(annotated, (box[2], box[3]), (box[2] - c_len, box[3]), (255, 255, 255), 2)
    cv2.line(annotated, (box[2], box[3]), (box[2], box[3] - c_len), (255, 255, 255), 2)

    # Top Incident Banner
    cv2.rectangle(annotated, (0, 0), (w, 55), (15, 18, 24), -1)
    cv2.rectangle(annotated, (0, 52), (w, 55), risk_color, -1)

    banner_text = f"EVIDENCE LOG | {risk_level.value} RISK EVENT: {incident_data['behaviour_type'].upper()} | T: {incident_data['timestamp_sec']:.2f}s"
    cv2.putText(annotated, banner_text, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Bottom Explanation Footer
    cv2.rectangle(annotated, (0, h - 60), (w, h), (15, 18, 24), -1)
    desc = incident_data["evidence_description"]
    if len(desc) > 95:
        desc = desc[:92] + "..."
    cv2.putText(annotated, f"FINDING: {desc}", (20, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 220, 240), 1, cv2.LINE_AA)
    cv2.putText(annotated, f"ACTION: {incident_data['recommended_action'][:100]}", (20, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 220, 160), 1, cv2.LINE_AA)

    filename = f"evidence_{incident_data['id']}.jpg"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, annotated)
    return filepath
