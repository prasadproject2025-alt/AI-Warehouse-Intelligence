"""
Video Processing Pipeline
Executes end-to-end video intelligence:
Ingestion -> Detection -> Tracking -> Behaviour Reasoning -> Risk Engine -> Incidents -> Annotated Video
"""

import cv2
import numpy as np
import os
import uuid
import time
from typing import Dict, Any, List, Optional
from detection.detector import WarehouseDetector
from detection.tracker import PersistentTracker
from behaviour.behaviour_engine import BehaviourEngine
from behaviour.base import BehaviourEvent
from video.evidence import draw_hud_overlay, draw_track_annotations, create_evidence_snapshot
from backend.database.db import DatabaseManager

class VideoProcessor:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        output_dir: str = "data/processed",
        evidence_dir: str = "data/evidence"
    ):
        self.detector = WarehouseDetector(model_path=model_path, conf_threshold=conf_threshold)
        self.output_dir = output_dir
        self.evidence_dir = evidence_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(evidence_dir, exist_ok=True)

    def process_video(
        self,
        video_path: str,
        video_id: Optional[str] = None,
        generate_annotated_video: bool = True,
        frame_skip: int = 1 # 1 = process every frame, 2 = every other frame for high throughput
    ) -> Dict[str, Any]:
        """
        Run end-to-end pipeline on input video.
        """
        assert os.path.exists(video_path), f"Video file not found: {video_path}"
        filename = os.path.basename(video_path)
        if not video_id:
            video_id = f"vid_{uuid.uuid4().hex[:8]}"

        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f"Cannot open video: {video_path}"

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0.0

        # Scenario heuristics based on filename
        is_wet = "wet" in filename.lower()
        is_dock = "dock" in filename.lower()

        tracker = PersistentTracker()
        behaviour_engine = BehaviourEngine(is_wet_floor=is_wet, is_dock_scene=is_dock)

        annotated_path = None
        writer = None
        if generate_annotated_video:
            annotated_filename = f"annotated_{filename}"
            annotated_path = os.path.join(self.output_dir, annotated_filename)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))

        frame_idx = 0
        all_incidents: List[Dict[str, Any]] = []
        recent_alert_banner: Optional[str] = None
        recent_alert_expiry: float = 0.0

        print(f"Starting Video Processing: {filename} ({frame_count} frames, {duration:.1f}s)")
        t_start = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps
            raw_frame_copy = frame.copy() if generate_annotated_video else None

            # 1. Detection
            detections = self.detector.detect(frame)

            # 2. Tracking
            active_tracks = tracker.update(detections, frame_idx, timestamp, fps)

            # 3. Behaviour Reasoning & Risk
            new_events = behaviour_engine.process_frame(active_tracks, frame_idx, timestamp)

            # 4. Handle new incidents
            for event in new_events:
                inc_id = f"inc_{uuid.uuid4().hex[:8]}"
                inc_dict = {
                    "id": inc_id,
                    "video_id": video_id,
                    "timestamp_sec": event.timestamp_sec,
                    "frame_idx": event.frame_idx,
                    "behaviour_type": event.behaviour_type.value,
                    "object_track_id": event.object_track_id,
                    "operator_track_id": event.operator_track_id,
                    "confidence": event.confidence,
                    "risk_level": event.risk_level.value,
                    "risk_score": event.risk_score,
                    "evidence_description": event.evidence_description,
                    "root_cause": event.root_cause,
                    "recommended_action": event.recommended_action,
                    "bounding_box": event.bounding_box,
                }
                
                # Generate evidence snapshot
                evidence_img_path = create_evidence_snapshot(frame, inc_dict, self.evidence_dir)
                inc_dict["evidence_image_path"] = evidence_img_path
                
                # Save to database
                DatabaseManager.save_incident(inc_dict)
                all_incidents.append(inc_dict)

                # Set on-screen banner for video replay
                recent_alert_banner = f"[{event.risk_level.value}] {event.behaviour_type.value.upper()}"
                recent_alert_expiry = timestamp + 1.8

            # 5. Render annotated frame if video recording requested
            if generate_annotated_video and writer is not None:
                vis_frame = draw_hud_overlay(frame, frame_idx, timestamp, fps, len(active_tracks), len(all_incidents))
                vis_frame = draw_track_annotations(vis_frame, active_tracks)
                
                # Draw live alert badge if recent
                if recent_alert_banner and timestamp < recent_alert_expiry:
                    cv2.rectangle(vis_frame, (width - 340, 50), (width - 15, 95), (0, 0, 210), -1)
                    cv2.putText(vis_frame, recent_alert_banner, (width - 330, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                writer.write(vis_frame)

            frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()

        elapsed = time.time() - t_start
        print(f"Video {filename} processed in {elapsed:.2f}s ({frame_idx/max(0.1, elapsed):.1f} FPS). Detected {len(all_incidents)} incidents.")

        # Save video record in DB
        DatabaseManager.save_video(
            video_id=video_id,
            filename=filename,
            filepath=video_path,
            duration=round(duration, 2),
            fps=round(fps, 2),
            frame_count=frame_idx,
            width=width,
            height=height,
            status="completed",
            annotated_path=annotated_path
        )

        return {
            "video_id": video_id,
            "filename": filename,
            "duration": round(duration, 2),
            "frame_count": frame_idx,
            "incidents_count": len(all_incidents),
            "incidents": all_incidents,
            "annotated_video": annotated_path
        }
