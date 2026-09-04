"""
Browser-playable video writer.

OpenCV's default MP4 writer emits MPEG-4 Part 2 (FOURCC "FMP4"/"mp4v"), which
no mainstream browser can decode - the file downloads fine and the <video>
element then renders a black frame. Every clip this system produces is meant to
be watched in the dashboard, so output must be H.264 in an MP4 container with
`faststart`, which is what browsers actually support.

OpenCV cannot be relied on for H.264: on Windows its bundled OpenH264 DLL is
frequently the wrong version, and `VideoWriter.isOpened()` still returns True
while writing an unplayable ~1 KB file. So frames are piped to ffmpeg instead
(supplied by imageio-ffmpeg, no system install needed).

If ffmpeg is genuinely unavailable the writer falls back to OpenCV and marks
itself `browser_playable = False`, so callers can warn rather than silently
producing clips nobody can watch.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_FFMPEG_PATH: Optional[str] = None
_FFMPEG_RESOLVED = False


def get_ffmpeg() -> Optional[str]:
    """Locate an ffmpeg binary once, preferring the pip-installed one."""
    global _FFMPEG_PATH, _FFMPEG_RESOLVED
    if _FFMPEG_RESOLVED:
        return _FFMPEG_PATH

    _FFMPEG_RESOLVED = True
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            _FFMPEG_PATH = exe
            return _FFMPEG_PATH
    except Exception:  # pragma: no cover - optional dependency
        pass

    _FFMPEG_PATH = shutil.which("ffmpeg")
    if not _FFMPEG_PATH:
        logger.warning(
            "ffmpeg not found - annotated video will use a codec browsers cannot "
            "play. Install it with: pip install imageio-ffmpeg"
        )
    return _FFMPEG_PATH


class BrowserVideoWriter:
    """
    Frame sink that produces an H.264 MP4 the dashboard can actually play.

    Mirrors the cv2.VideoWriter surface used by the pipeline (`write`,
    `release`, `isOpened`) so it is a drop-in replacement.
    """

    def __init__(self, path: str, fps: float, width: int, height: int, crf: int = 23):
        self.path = path
        # H.264 requires even dimensions; odd ones make libx264 fail outright.
        self.width = int(width) - (int(width) % 2)
        self.height = int(height) - (int(height) % 2)
        self.fps = float(fps) if fps and fps > 0 else 25.0
        self.frames_written = 0
        self._proc: Optional[subprocess.Popen] = None
        self._cv: Optional[cv2.VideoWriter] = None
        self.browser_playable = False

        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

        ffmpeg = get_ffmpeg()
        if ffmpeg:
            cmd = [
                ffmpeg, "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{self.width}x{self.height}",
                "-r", f"{self.fps:.4f}",
                "-i", "-",
                "-an",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", str(crf),
                # yuv420p is required for broad browser/device compatibility.
                "-pix_fmt", "yuv420p",
                # Move the index to the front so playback can start while the
                # rest of the file is still downloading.
                "-movflags", "+faststart",
                path,
            ]
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                self.browser_playable = True
            except Exception as exc:  # pragma: no cover - environment dependent
                logger.error("Could not start ffmpeg (%s); falling back to OpenCV", exc)
                self._proc = None

        if self._proc is None:
            self._cv = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.width, self.height)
            )

    def isOpened(self) -> bool:  # noqa: N802 - mirrors the cv2 API
        if self._proc is not None:
            return self._proc.poll() is None
        return bool(self._cv and self._cv.isOpened())

    def write(self, frame: np.ndarray) -> None:
        if frame is None:
            return
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)

        if self._proc is not None:
            if self._proc.poll() is not None:
                return  # encoder died; release() reports why
            try:
                self._proc.stdin.write(frame.tobytes())
                self.frames_written += 1
            except (BrokenPipeError, OSError) as exc:
                logger.error("ffmpeg pipe closed after %d frames: %s", self.frames_written, exc)
                self._proc = None
        elif self._cv is not None:
            self._cv.write(frame)
            self.frames_written += 1

    def release(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                _, err = self._proc.communicate(timeout=120)
                if self._proc.returncode not in (0, None):
                    logger.error(
                        "ffmpeg exited %s: %s",
                        self._proc.returncode,
                        (err or b"").decode("utf-8", "replace")[:500],
                    )
            except subprocess.TimeoutExpired:  # pragma: no cover
                self._proc.kill()
                logger.error("ffmpeg timed out finalising %s", self.path)
            self._proc = None
        if self._cv is not None:
            self._cv.release()
            self._cv = None


def probe_codec(path: str) -> str:
    """Return the FOURCC tag of an existing video, for diagnostics and tests."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return ""
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00")


def is_browser_playable(path: str) -> bool:
    """
    True when the container holds H.264, the only codec every target browser
    decodes. FMP4/mp4v files are readable by OpenCV but render black in a page.
    """
    if not os.path.exists(path) or os.path.getsize(path) < 1024:
        return False
    ffmpeg = get_ffmpeg()
    if ffmpeg:
        try:
            out = subprocess.run(
                [ffmpeg, "-i", path, "-hide_banner"],
                capture_output=True, text=True, timeout=30,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return "h264" in (out.stderr or "").lower()
        except Exception:
            pass
    return probe_codec(path).lower() in {"avc1", "h264"}
