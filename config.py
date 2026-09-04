"""
Central configuration for VisionGuard.

All tunables are environment-driven so that nothing operational is hard-coded in
the pipeline. Values are read once at import time; a .env file (if present) is
loaded first so local development does not need shell exports.
"""

from __future__ import annotations

import os
from pathlib import Path

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv missing is non-fatal
    pass

BASE_DIR = Path(__file__).resolve().parent


def _env(key: str, default: str) -> str:
    val = os.environ.get(key)
    return default if val is None or val == "" else val


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(float(_env(key, str(default))))
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    return _env(key, "true" if default else "false").strip().lower() in {"1", "true", "yes", "on"}


def _resolve(p: str) -> str:
    path = Path(p)
    return str(path if path.is_absolute() else (BASE_DIR / path))


# --- Server -----------------------------------------------------------------
HOST = _env("HOST", "127.0.0.1")
PORT = _env_int("PORT", 8000)
DEBUG = _env_bool("DEBUG", False)
# Comma separated list. "*" is permitted only when credentials are disabled.
CORS_ORIGINS = [o.strip() for o in _env("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000").split(",") if o.strip()]
LOG_LEVEL = _env("LOG_LEVEL", "INFO").upper()

# --- Storage ----------------------------------------------------------------
DATABASE_PATH = _resolve(_env("DATABASE_URL", "sqlite:///./data/visionguard.db").replace("sqlite:///", ""))
RAW_VIDEOS_DIR = _resolve(_env("RAW_VIDEOS_DIR", "./data/raw"))
PROCESSED_VIDEOS_DIR = _resolve(_env("PROCESSED_VIDEOS_DIR", "./data/processed"))
EVIDENCE_DIR = _resolve(_env("EVIDENCE_DIR", "./data/evidence"))
CLIPS_DIR = _resolve(_env("CLIPS_DIR", "./data/clips"))

MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB", 512)
ALLOWED_VIDEO_EXTENSIONS = {
    e.strip().lower() if e.strip().startswith(".") else "." + e.strip().lower()
    for e in _env("ALLOWED_VIDEO_EXTENSIONS", ".mp4,.avi,.mov,.mkv,.webm").split(",")
    if e.strip()
}

# --- Perception -------------------------------------------------------------
# Open-vocabulary detector gives real "carton / pallet / trolley" classes that
# COCO does not contain. Falls back to the COCO model when unavailable.
OPEN_VOCAB_MODEL_PATH = _env("OPEN_VOCAB_MODEL_PATH", "yolov8s-worldv2.pt")
FALLBACK_MODEL_PATH = _env("YOLO_MODEL_PATH", "yolov8n.pt")
USE_OPEN_VOCAB = _env_bool("USE_OPEN_VOCAB", True)

PERSON_CONF = _env_float("PERSON_CONF", 0.25)
PRODUCT_CONF = _env_float("PRODUCT_CONF", 0.12)
EQUIPMENT_CONF = _env_float("EQUIPMENT_CONF", 0.20)
IOU_THRESHOLD = _env_float("IOU_THRESHOLD", 0.45)
INFERENCE_IMGSZ = _env_int("INFERENCE_IMGSZ", 640)

# Frames actually sent to the detector: 1 = every frame, 3 = every third frame.
DETECTION_FRAME_STRIDE = _env_int("DETECTION_FRAME_STRIDE", 3)
# Longest edge the frame is resized to before inference (0 = native resolution).
MAX_INFERENCE_WIDTH = _env_int("MAX_INFERENCE_WIDTH", 960)

# --- Behaviour engine -------------------------------------------------------
# Minimum seconds between two alerts of the same behaviour on the same track.
ALERT_COOLDOWN_SEC = _env_float("ALERT_COOLDOWN_SEC", 4.0)
# A track must be observed this long before behaviour reasoning trusts it.
MIN_TRACK_AGE_SEC = _env_float("MIN_TRACK_AGE_SEC", 0.35)

# --- Evidence ---------------------------------------------------------------
EVIDENCE_CLIP_PRE_SEC = _env_float("EVIDENCE_CLIP_PRE_SEC", 2.0)
EVIDENCE_CLIP_POST_SEC = _env_float("EVIDENCE_CLIP_POST_SEC", 2.0)
GENERATE_EVIDENCE_CLIPS = _env_bool("GENERATE_EVIDENCE_CLIPS", True)

# --- Responsible AI ---------------------------------------------------------
# Faces are blurred in stored evidence unless explicitly disabled for a pilot
# where operators have consented to identifiable review.
BLUR_FACES_IN_EVIDENCE = _env_bool("BLUR_FACES_IN_EVIDENCE", True)
EVIDENCE_RETENTION_DAYS = _env_int("EVIDENCE_RETENTION_DAYS", 30)


def ensure_directories() -> None:
    for d in (RAW_VIDEOS_DIR, PROCESSED_VIDEOS_DIR, EVIDENCE_DIR, CLIPS_DIR, os.path.dirname(DATABASE_PATH)):
        if d:
            os.makedirs(d, exist_ok=True)
