"""
Video Processing Pipeline.

Ingestion -> detection -> tracking -> temporal behaviour reasoning -> risk
scoring -> incident + evidence generation -> annotated video.

Design notes:
* The detector runs on a strided subset of frames (``DETECTION_FRAME_STRIDE``)
  while the annotated writer still receives every frame, so the output video
  plays at the original rate with tracks held between analysed frames.
* Frames are downscaled once for inference and detections are mapped back to
  full resolution, which is the main throughput lever.
* Failures are captured and recorded against the video row, so a crashed
  analysis surfaces in the UI as an error rather than a task stuck at 99%.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

import config
from backend.database.db import DatabaseManager
from behaviour.behaviour_engine import BehaviourEngine, SceneContext
from detection.detector import WarehouseDetector
from detection.tracker import PersistentTracker
from video.encoder import BrowserVideoWriter
from video.evidence import (
    create_evidence_snapshot,
    draw_hud_overlay,
    draw_track_annotations,
    write_incident_clip,
)

logger = logging.getLogger(__name__)

#: Live progress for background analyses, keyed by video id.
TASK_STATUS: Dict[str, Dict[str, Any]] = {}


class VideoProcessor:
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        output_dir: Optional[str] = None,
        evidence_dir: Optional[str] = None,
        clips_dir: Optional[str] = None,
    ) -> None:
        self.detector = WarehouseDetector(
            model_path=model_path, conf_threshold=conf_threshold
        )
        self.output_dir = output_dir or config.PROCESSED_VIDEOS_DIR
        self.evidence_dir = evidence_dir or config.EVIDENCE_DIR
        self.clips_dir = clips_dir or config.CLIPS_DIR
        for d in (self.output_dir, self.evidence_dir, self.clips_dir):
            os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------ entry
    def process_video(
        self,
        video_path: str,
        video_id: Optional[str] = None,
        generate_annotated_video: bool = True,
        frame_stride: Optional[int] = None,
        scene: Optional[SceneContext] = None,
    ) -> Dict[str, Any]:
        video_id = video_id or f"vid_{uuid.uuid4().hex[:8]}"
        try:
            return self._run(
                video_path, video_id, generate_annotated_video, frame_stride, scene
            )
        except Exception as exc:  # noqa: BLE001 - background task must not die silently
            logger.exception("Video analysis failed for %s", video_path)
            TASK_STATUS[video_id] = {
                "video_id": video_id,
                "filename": os.path.basename(video_path),
                "status": "failed",
                "progress_percent": 100,
                "error": str(exc),
                "incidents_count": 0,
            }
            try:
                DatabaseManager.update_video_status(video_id, "failed", str(exc))
            except Exception:  # noqa: BLE001
                logger.exception("Could not persist failure state for %s", video_id)
            raise

    # -------------------------------------------------------------- pipeline
    def _run(
        self,
        video_path: str,
        video_id: str,
        generate_annotated_video: bool,
        frame_stride: Optional[int],
        scene: Optional[SceneContext],
    ) -> Dict[str, Any]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        filename = os.path.basename(video_path)
        scene = scene or SceneContext()
        stride = max(1, int(frame_stride or config.DETECTION_FRAME_STRIDE))

        TASK_STATUS[video_id] = {
            "video_id": video_id,
            "filename": filename,
            "status": "processing",
            "progress_percent": 0,
            "current_frame": 0,
            "total_frames": 0,
            "incidents_count": 0,
            "stage": "opening video",
        }

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot decode video: {filename}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not np.isfinite(fps) or fps <= 1.0:
            fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            cap.release()
            raise ValueError(f"Video reports invalid dimensions ({width}x{height})")
        duration = frame_count / fps if fps > 0 else 0.0

        TASK_STATUS[video_id].update(total_frames=frame_count, stage="analysing frames")

        # Inference downscale factor.
        infer_scale = 1.0
        if config.MAX_INFERENCE_WIDTH and width > config.MAX_INFERENCE_WIDTH:
            infer_scale = config.MAX_INFERENCE_WIDTH / float(width)

        tracker = PersistentTracker(frame_height=height, frame_width=width)
        recurrence_baseline = {
            b: DatabaseManager.get_behaviour_history(b, scene.bay)
            for b in ("product_drop", "product_drag", "product_throw", "rolling_product",
                      "improper_stacking", "stepping_on_carton", "unsupported_handling",
                      "wet_floor_hazard", "orientation_violation", "dock_level_hazard",
                      "outside_designated_area", "unsafe_loading_sequence")
        }
        engine = BehaviourEngine(scene=scene, recurrence_baseline=recurrence_baseline)
        engine.bind_tracker(tracker)

        annotated_path = None
        writer = None
        if generate_annotated_video:
            safe_stem = os.path.splitext(filename)[0].replace(" ", "_")[:60]
            annotated_path = os.path.join(self.output_dir, f"annotated_{video_id}_{safe_stem}.mp4")
            # H.264 via ffmpeg: OpenCV's mp4v output is MPEG-4 Part 2, which
            # browsers refuse to decode, so the overlay rendered as a black
            # <video> in the dashboard.
            writer = BrowserVideoWriter(annotated_path, fps, width, height)
            if not writer.isOpened():
                logger.warning("Could not open video writer; continuing without annotated output")
                writer, annotated_path = None, None
            elif not writer.browser_playable:
                logger.warning(
                    "ffmpeg unavailable - annotated video will not play in the dashboard. "
                    "Install it with: pip install imageio-ffmpeg"
                )

        frame_idx = 0
        analysed = 0
        all_incidents: List[Dict[str, Any]] = []
        pending_clips: List[Dict[str, Any]] = []
        banner: Optional[str] = None
        banner_until = 0.0

        logger.info(
            "Analysing %s (%d frames, %.1fs, %dx%d) backend=%s stride=%d",
            filename, frame_count, duration, width, height, self.detector.backend, stride,
        )
        t_start = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = frame_idx / fps

            if frame_idx % stride == 0:
                infer_frame = (
                    cv2.resize(frame, None, fx=infer_scale, fy=infer_scale)
                    if infer_scale < 1.0
                    else frame
                )
                detections = self.detector.detect(infer_frame)
                if infer_scale < 1.0:
                    inv = 1.0 / infer_scale
                    for d in detections:
                        d.box = [v * inv for v in d.box]

                active_tracks = tracker.update(detections, frame_idx, timestamp, fps / stride)
                analysed += 1

                for event in engine.process_frame(active_tracks, frame_idx, timestamp):
                    inc = self._materialise_incident(event, video_id, scene, frame)
                    all_incidents.append(inc)
                    if config.GENERATE_EVIDENCE_CLIPS:
                        pending_clips.append(
                            {"id": inc["id"], "timestamp_sec": inc["timestamp_sec"]}
                        )
                    banner = f"[{event.risk_level.value}] {event.behaviour_type.value.replace('_', ' ').upper()}"
                    banner_until = timestamp + 2.0

            if frame_count > 0 and frame_idx % 5 == 0:
                TASK_STATUS[video_id].update(
                    current_frame=frame_idx,
                    progress_percent=min(97, int((frame_idx / frame_count) * 100)),
                    incidents_count=len(all_incidents),
                )

            if writer is not None:
                vis = draw_hud_overlay(
                    frame, frame_idx, timestamp, fps, len(tracker.tracks), len(all_incidents)
                )
                vis = draw_track_annotations(vis, list(tracker.tracks.values()))
                if banner and timestamp < banner_until:
                    _draw_alert_banner(vis, banner, width)
                writer.write(vis)

            frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()

        # Evidence clips need a second pass over the source video.
        if pending_clips:
            TASK_STATUS[video_id].update(stage="building evidence clips", progress_percent=98)
            self._attach_clips(video_path, pending_clips, all_incidents)

        elapsed = time.time() - t_start
        realtime_ratio = (duration / elapsed) if elapsed > 0 else 0.0
        logger.info(
            "%s: %d incidents in %.1fs (%.2fx realtime, %d frames analysed)",
            filename, len(all_incidents), elapsed, realtime_ratio, analysed,
        )

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
            annotated_path=annotated_path,
            camera_id=scene.camera_id,
            bay=scene.bay,
            shift=scene.shift,
            processing_seconds=round(elapsed, 2),
            detector_backend=self.detector.backend,
            frames_analysed=analysed,
            scene_flags=scene.to_dict(),
        )

        result = {
            "video_id": video_id,
            "filename": filename,
            "duration": round(duration, 2),
            "frame_count": frame_idx,
            "frames_analysed": analysed,
            "detector_backend": self.detector.backend,
            "processing_seconds": round(elapsed, 2),
            "realtime_ratio": round(realtime_ratio, 2),
            "incidents_count": len(all_incidents),
            "incidents": all_incidents,
            "annotated_video": annotated_path,
            "scene": scene.to_dict(),
        }
        TASK_STATUS[video_id] = {
            "video_id": video_id,
            "filename": filename,
            "status": "completed",
            "progress_percent": 100,
            "current_frame": frame_idx,
            "total_frames": frame_count,
            "incidents_count": len(all_incidents),
            "stage": "complete",
            "result": result,
        }
        return result

    # ------------------------------------------------------------- incidents
    def _materialise_incident(
        self, event, video_id: str, scene: SceneContext, frame: np.ndarray
    ) -> Dict[str, Any]:
        inc_id = f"inc_{uuid.uuid4().hex[:8]}"
        inc = {
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
            "risk_factors": event.risk_factors,
            "evidence_stages": event.evidence_stages,
            "evidence_tier": event.evidence_tier.value,
            "duration_sec": event.duration_sec,
            "camera_id": scene.camera_id,
            "bay": scene.bay,
            "shift": scene.shift,
            "review_status": "PENDING_REVIEW",
        }
        try:
            inc["evidence_image_path"] = create_evidence_snapshot(
                frame, inc, self.evidence_dir
            )
        except Exception:  # noqa: BLE001
            logger.exception("Evidence snapshot failed for %s", inc_id)
            inc["evidence_image_path"] = None

        DatabaseManager.save_incident(inc)
        return inc

    def _attach_clips(
        self, video_path: str, pending: List[Dict[str, Any]], incidents: List[Dict[str, Any]]
    ) -> None:
        by_id = {i["id"]: i for i in incidents}
        for item in pending:
            try:
                path = write_incident_clip(
                    video_path,
                    item["timestamp_sec"],
                    os.path.join(self.clips_dir, f"clip_{item['id']}.mp4"),
                    pre_sec=config.EVIDENCE_CLIP_PRE_SEC,
                    post_sec=config.EVIDENCE_CLIP_POST_SEC,
                )
            except Exception:  # noqa: BLE001
                logger.exception("Evidence clip failed for %s", item["id"])
                continue
            if path:
                inc = by_id.get(item["id"])
                if inc is not None:
                    inc["evidence_clip_path"] = path
                    DatabaseManager.save_incident(inc)


def _draw_alert_banner(frame: np.ndarray, text: str, width: int) -> None:
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    x1 = max(10, width - tw - 40)
    cv2.rectangle(frame, (x1, 52), (width - 12, 96), (0, 0, 200), -1)
    cv2.putText(
        frame, text, (x1 + 14, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA
    )
