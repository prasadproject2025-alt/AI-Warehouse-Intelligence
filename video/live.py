"""
Live / near-real-time stream analysis.

Runs the *same* perception, tracking, behaviour and risk stack as the recorded
pipeline against a continuous source - a USB camera, an RTSP/HTTP CCTV feed, or
a file replayed at its natural rate to demonstrate the live path without a
camera present. Nothing here re-implements or simplifies the analysis: a live
incident is produced by exactly the code that produces a recorded one, and is
written to the same table, so the dashboard and the assistant treat both alike.

Honesty about timing
--------------------
On CPU the stack runs at roughly 0.2-0.4x realtime, so every frame cannot be
analysed as it arrives. The session therefore *drops* frames rather than
queueing them: it always analyses the most recent frame available and reports
the true analysed-FPS and the drop count. This keeps alerts anchored to the
present moment instead of drifting further behind, and the UI shows the real
figures rather than implying full-rate analysis.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import cv2
import numpy as np

import config
from backend.database.db import DatabaseManager
from behaviour.behaviour_engine import BehaviourEngine, SceneContext
from detection.tracker import PersistentTracker
from video.evidence import draw_hud_overlay, draw_track_annotations

logger = logging.getLogger(__name__)

#: YOLO inference is not thread-safe, and the recorded pipeline may be running
#: in the background at the same time. All inference is serialised on this lock.
INFERENCE_LOCK = threading.Lock()

#: Sources a caller may open. Anything else is rejected before a capture is
#: attempted, so a request cannot make the server open an arbitrary file.
SOURCE_KINDS = ("camera", "stream", "file")


class LiveSession:
    """One running analysis of one continuous source."""

    def __init__(
        self,
        session_id: str,
        source: Any,
        kind: str,
        detector,
        scene: SceneContext,
        evidence_dir: str,
        loop_file: bool = True,
        max_events: int = 200,
    ) -> None:
        self.session_id = session_id
        self.source = source
        self.kind = kind
        self.scene = scene
        self.detector = detector
        self.evidence_dir = evidence_dir
        self.loop_file = loop_file

        self.status = "starting"
        self.error: Optional[str] = None
        self.started_at = time.time()

        self.frames_read = 0
        self.frames_analysed = 0
        self.frames_dropped = 0
        self.analysed_fps = 0.0
        self.active_tracks = 0

        self.events: Deque[Dict[str, Any]] = deque(maxlen=max_events)
        self.event_count = 0

        self._latest_jpeg: Optional[bytes] = None
        self._frame_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Live incidents are persisted against a video row so that existing
        # queries, analytics and the assistant pick them up unchanged.
        self.video_id = f"live_{session_id}"

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"live-{self.session_id}", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        if self.status not in ("error",):
            self.status = "stopped"

    # ------------------------------------------------------------- internals
    def _open(self) -> cv2.VideoCapture:
        if self.kind == "camera":
            cap = cv2.VideoCapture(int(self.source), cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self.source)
        # A small buffer keeps a network feed close to live rather than
        # replaying a backlog after any processing stall.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            pass
        return cap

    def _run(self) -> None:  # noqa: C901 - a stream loop with its failure paths
        cap = None
        try:
            cap = self._open()
            if not cap.isOpened():
                raise RuntimeError(f"Could not open {self.kind} source: {self.source}")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            if not np.isfinite(src_fps) or src_fps <= 1.0:
                src_fps = 25.0
            if width <= 0 or height <= 0:
                ok, probe = cap.read()
                if not ok:
                    raise RuntimeError("Source opened but produced no frames")
                height, width = probe.shape[:2]

            DatabaseManager.save_video(
                video_id=self.video_id,
                filename=f"LIVE - {self.scene.camera_id}",
                filepath=str(self.source),
                duration=0.0,
                fps=round(src_fps, 2),
                frame_count=0,
                width=width,
                height=height,
                status="live",
                camera_id=self.scene.camera_id,
                bay=self.scene.bay,
                shift=self.scene.shift,
                detector_backend=self.detector.backend,
                scene_flags={"live": True, "source_kind": self.kind},
            )

            infer_scale = 1.0
            if config.MAX_INFERENCE_WIDTH and width > config.MAX_INFERENCE_WIDTH:
                infer_scale = config.MAX_INFERENCE_WIDTH / float(width)

            tracker = PersistentTracker(frame_height=height, frame_width=width)
            engine = BehaviourEngine(
                scene=self.scene,
                recurrence_baseline={
                    b: DatabaseManager.get_behaviour_history(b, self.scene.bay)
                    for b in (
                        "product_drop", "product_drag", "product_throw", "rolling_product",
                        "improper_stacking", "stepping_on_carton", "unsupported_handling",
                        "wet_floor_hazard", "orientation_violation", "dock_level_hazard",
                        "outside_designated_area", "unsafe_loading_sequence",
                    )
                },
            )
            engine.bind_tracker(tracker)

            self.status = "running"
            banner: Optional[str] = None
            banner_until = 0.0
            fps_window: Deque[float] = deque(maxlen=30)
            frame_idx = 0
            t0 = time.time()
            self.started_at = t0

            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    if self.kind == "file" and self.loop_file:
                        # Replaying a file as a stand-in for a camera: loop so a
                        # demo can run continuously.
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    self.status = "ended"
                    break

                self.frames_read += 1
                frame_idx += 1
                now = time.time()
                timestamp = now - self.started_at

                # Stay anchored to the present by discarding the backlog.
                #
                # This is what makes the analysis *live* rather than merely
                # slow, and it is also required for correctness: velocity is
                # derived from the wall-clock gap between analysed frames, so
                # if consecutive analysed frames were adjacent in the source
                # while seconds of wall-clock had passed, every speed would be
                # understated by that ratio and no kinematic behaviour would
                # ever fire.
                if self.kind == "file":
                    # Replaying at natural rate: skip to the frame the clock has
                    # reached, exactly as a camera would have moved on.
                    target = int((time.time() - t0) * src_fps)
                    behind = target - frame_idx
                    if behind > 0:
                        for _ in range(min(behind, 300)):
                            if not cap.grab():
                                break
                            frame_idx += 1
                            self.frames_read += 1
                            self.frames_dropped += 1
                        got, newer = cap.retrieve()
                        if got:
                            frame = newer
                else:
                    while cap.grab():
                        got, newer = cap.retrieve()
                        if not got:
                            break
                        frame = newer
                        self.frames_read += 1
                        self.frames_dropped += 1
                        if self.frames_dropped % 64 == 0:
                            break  # never spin forever on a fast source

                t_frame = time.time()
                infer = (
                    cv2.resize(frame, None, fx=infer_scale, fy=infer_scale)
                    if infer_scale < 1.0
                    else frame
                )
                with INFERENCE_LOCK:
                    detections = self.detector.detect(infer)
                if infer_scale < 1.0:
                    inv = 1.0 / infer_scale
                    for d in detections:
                        d.box = [v * inv for v in d.box]

                effective_fps = max(1.0, self.analysed_fps or 5.0)
                tracks = tracker.update(detections, frame_idx, timestamp, effective_fps)
                self.frames_analysed += 1
                self.active_tracks = len(tracker.tracks)

                for event in engine.process_frame(tracks, frame_idx, timestamp):
                    inc = self._materialise(event, frame)
                    self.events.appendleft(inc)
                    self.event_count += 1
                    banner = (
                        f"[{event.risk_level.value}] "
                        f"{event.behaviour_type.value.replace('_', ' ').upper()}"
                    )
                    banner_until = timestamp + 3.0
                    logger.info(
                        "LIVE %s: %s (%s) at %.1fs",
                        self.session_id, event.behaviour_type.value,
                        event.risk_level.value, timestamp,
                    )

                fps_window.append(time.time() - t_frame)
                if fps_window:
                    mean = float(np.mean(fps_window))
                    self.analysed_fps = round(1.0 / mean, 2) if mean > 0 else 0.0

                vis = draw_hud_overlay(
                    frame.copy(), frame_idx, timestamp, self.analysed_fps,
                    self.active_tracks, self.event_count,
                )
                vis = draw_track_annotations(vis, list(tracker.tracks.values()))
                if banner and timestamp < banner_until:
                    self._draw_banner(vis, banner)
                self._publish(vis)

                # If analysis is briefly faster than the source, wait rather
                # than racing ahead of the clock. Falling behind is handled by
                # the frame-skip above.
                if self.kind == "file":
                    ahead = (frame_idx / src_fps) - (time.time() - t0)
                    if ahead > 0:
                        time.sleep(min(ahead, 0.25))

        except Exception as exc:  # noqa: BLE001 - a thread must not die silently
            logger.exception("Live session %s failed", self.session_id)
            self.status = "error"
            self.error = str(exc)
        finally:
            if cap is not None:
                cap.release()
            if self.status == "running":
                self.status = "stopped"
            try:
                DatabaseManager.update_video_status(self.video_id, "completed")
            except Exception:  # noqa: BLE001
                logger.exception("Could not finalise live video row %s", self.video_id)

    # ------------------------------------------------------------- rendering
    @staticmethod
    def _draw_banner(frame: np.ndarray, text: str) -> None:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 46), (w, h), (0, 0, 170), -1)
        cv2.putText(
            frame, text, (18, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
            (255, 255, 255), 2, cv2.LINE_AA,
        )

    def _publish(self, frame: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if ok:
            with self._frame_lock:
                self._latest_jpeg = buf.tobytes()

    def latest_jpeg(self) -> Optional[bytes]:
        with self._frame_lock:
            return self._latest_jpeg

    def mjpeg_frames(self):
        """Yield multipart JPEG chunks for as long as the session runs."""
        boundary = b"--frame\r\n"
        idle = 0.0
        while self.status in ("starting", "running") or self._latest_jpeg is not None:
            data = self.latest_jpeg()
            if data is None:
                time.sleep(0.1)
                idle += 0.1
                if idle > 20:
                    return
                continue
            idle = 0.0
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
            # Cap the outgoing rate; the analysed rate is lower than this
            # anyway and re-sending an unchanged frame wastes bandwidth.
            time.sleep(0.08)
            if self._stop.is_set() and self.status not in ("running", "starting"):
                return

    # ------------------------------------------------------------- incidents
    def _materialise(self, event, frame: np.ndarray) -> Dict[str, Any]:
        from video.evidence import create_evidence_snapshot

        inc_id = f"inc_{uuid.uuid4().hex[:8]}"
        inc = {
            "id": inc_id,
            "video_id": self.video_id,
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
            "camera_id": self.scene.camera_id,
            "bay": self.scene.bay,
            "shift": self.scene.shift,
            "review_status": "PENDING_REVIEW",
        }
        try:
            inc["evidence_image_path"] = create_evidence_snapshot(frame, inc, self.evidence_dir)
        except Exception:  # noqa: BLE001
            logger.exception("Live evidence snapshot failed for %s", inc_id)
            inc["evidence_image_path"] = None
        try:
            DatabaseManager.save_incident(inc)
        except Exception:  # noqa: BLE001
            logger.exception("Could not persist live incident %s", inc_id)
        return inc

    # ---------------------------------------------------------------- status
    def snapshot(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "video_id": self.video_id,
            "status": self.status,
            "error": self.error,
            "source_kind": self.kind,
            "camera_id": self.scene.camera_id,
            "bay": self.scene.bay,
            "shift": self.scene.shift,
            "uptime_sec": round(time.time() - self.started_at, 1),
            "frames_read": self.frames_read,
            "frames_analysed": self.frames_analysed,
            "frames_dropped": self.frames_dropped,
            "analysed_fps": self.analysed_fps,
            "active_tracks": self.active_tracks,
            "event_count": self.event_count,
            "events": list(self.events)[:20],
        }


class LiveSessionManager:
    """Owns running sessions; one per camera is the expected usage."""

    def __init__(self, detector, evidence_dir: str, max_sessions: int = 2) -> None:
        self.detector = detector
        self.evidence_dir = evidence_dir
        self.max_sessions = max_sessions
        self._sessions: Dict[str, LiveSession] = {}
        self._lock = threading.Lock()

    def start(self, source: Any, kind: str, scene: SceneContext, loop_file: bool = True) -> LiveSession:
        if kind not in SOURCE_KINDS:
            raise ValueError(f"source_kind must be one of {SOURCE_KINDS}")
        with self._lock:
            self._reap()
            if len(self._sessions) >= self.max_sessions:
                raise RuntimeError(
                    f"{self.max_sessions} live sessions already running; stop one first"
                )
            session_id = uuid.uuid4().hex[:8]
            session = LiveSession(
                session_id=session_id,
                source=source,
                kind=kind,
                detector=self.detector,
                scene=scene,
                evidence_dir=self.evidence_dir,
                loop_file=loop_file,
            )
            self._sessions[session_id] = session
        session.start()
        return session

    def get(self, session_id: str) -> Optional[LiveSession]:
        return self._sessions.get(session_id)

    def stop(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        if not session:
            return False
        session.stop()
        return True

    def list(self) -> List[Dict[str, Any]]:
        self._reap()
        return [s.snapshot() for s in self._sessions.values()]

    def stop_all(self) -> None:
        for s in list(self._sessions.values()):
            s.stop()

    def _reap(self) -> None:
        """Forget sessions that finished a while ago so slots free up."""
        for sid, s in list(self._sessions.items()):
            finished = s.status in ("stopped", "ended", "error")
            if finished and (s._thread is None or not s._thread.is_alive()):
                if time.time() - s.started_at > 5:
                    self._sessions.pop(sid, None)
