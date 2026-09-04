"""
VisionGuard FastAPI Backend Service.

REST API for video ingestion, incident timelines, evidence retrieval, shift
analytics, prevention insights, and the grounded supervisor assistant.

Security posture: CORS origins, upload size and allowed extensions all come
from configuration; uploaded filenames are sanitised and every served file path
is confined to its configured directory.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

import config
from assistant.llm import AIAssistant
from backend.database.db import DatabaseManager, init_db
from behaviour.behaviour_engine import BehaviourEngine, SceneContext
from video.processor import TASK_STATUS, VideoProcessor

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("visionguard.api")

app = FastAPI(
    title="VisionGuard - AI Warehouse Intelligence API",
    description=(
        "Backend API for AI video intelligence, behaviour understanding and "
        "damage prevention in warehouse loading/unloading operations."
    ),
    version="2.0.0",
)

# CORS: credentials and a wildcard origin are mutually exclusive per the spec,
# and browsers reject that combination, so allow_credentials tracks the config.
_wildcard = "*" in config.CORS_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=not _wildcard,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

config.ensure_directories()
init_db()

app.mount("/static/evidence", StaticFiles(directory=config.EVIDENCE_DIR), name="evidence")
app.mount("/static/raw", StaticFiles(directory=config.RAW_VIDEOS_DIR), name="raw_videos")
app.mount(
    "/static/processed", StaticFiles(directory=config.PROCESSED_VIDEOS_DIR), name="processed_videos"
)
app.mount("/static/clips", StaticFiles(directory=config.CLIPS_DIR), name="clips")

_DASHBOARD_DIST = os.path.join(config.BASE_DIR, "dashboard", "dist")
if os.path.isdir(os.path.join(_DASHBOARD_DIST, "assets")):
    app.mount(
        "/assets", StaticFiles(directory=os.path.join(_DASHBOARD_DIST, "assets")), name="assets"
    )

# One processor (one loaded model) shared across requests. Analysis runs in the
# background threadpool; loading the model per request would be far slower.
processor = VideoProcessor()

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
VALID_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
VALID_REVIEW_STATUSES = {
    "PENDING_REVIEW",
    "CONFIRMED_BY_SUPERVISOR",
    "FALSE_POSITIVE",
    "DAMAGE_CONFIRMED",
    "NO_ACTION_NEEDED",
}


# --------------------------------------------------------------------- models
class AssistantQueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    video_id: Optional[str] = None

    @field_validator("query")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v.strip()


class ReviewRequest(BaseModel):
    status: str
    note: Optional[str] = Field(default=None, max_length=2000)

    @field_validator("status")
    @classmethod
    def known_status(cls, v: str) -> str:
        up = v.strip().upper()
        if up not in VALID_REVIEW_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_REVIEW_STATUSES)}")
        return up


# ------------------------------------------------------------------- helpers
def _safe_filename(name: str) -> str:
    """Strip any directory component and unsafe characters from an upload name."""
    base = os.path.basename(name or "").replace("\\", "/").split("/")[-1]
    base = _SAFE_NAME.sub("_", base).strip("._") or "upload"
    return base[:120]


def _validate_id(value: str, label: str) -> str:
    if not _ID_RE.match(value or ""):
        raise HTTPException(status_code=400, detail=f"Invalid {label}")
    return value


def _static_url(path: Optional[str], mount: str) -> Optional[str]:
    """
    Convert a stored filesystem path into its public static URL.

    Only the basename is used, so a stored path can never escape its mount.
    The name is percent-encoded because pilot filenames contain spaces and
    commas, which are not valid in a raw URL path.
    """
    if not path:
        return None
    return f"/static/{mount}/{quote(os.path.basename(path))}"


def _decorate_incident(inc: Dict[str, Any]) -> Dict[str, Any]:
    inc["evidence_image_url"] = _static_url(inc.get("evidence_image_path"), "evidence")
    inc["evidence_clip_url"] = _static_url(inc.get("evidence_clip_path"), "clips")
    return inc


def _decorate_video(v: Dict[str, Any]) -> Dict[str, Any]:
    v["video_url"] = _static_url(v.get("filepath") or v.get("filename"), "raw")
    v["annotated_video_url"] = _static_url(v.get("annotated_filepath"), "processed")
    raw = v.get("scene_flags")
    if isinstance(raw, str):
        try:
            v["scene_flags"] = json.loads(raw) if raw else {}
        except ValueError:
            v["scene_flags"] = {}
    return v


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "path": request.url.path},
    )


# --------------------------------------------------------------------- routes
@app.get("/")
def serve_root():
    index = os.path.join(_DASHBOARD_DIST, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {
        "message": "VisionGuard API is running.",
        "hint": "Build the dashboard with: cd dashboard && npm install && npm run build",
        "docs": "/docs",
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "service": "VisionGuard AI Warehouse Intelligence",
        "version": "2.0.0",
        "detector_backend": processor.detector.backend,
        "open_vocabulary": processor.detector.backend == "open_vocab",
        "database": os.path.basename(config.DATABASE_PATH),
    }


@app.get("/api/capabilities")
def capabilities():
    """
    Honest, code-derived capability report.

    The coverage list is generated from the detector classes, so it cannot claim
    a behaviour is working when the implementation says otherwise.
    """
    coverage = BehaviourEngine.coverage_report()
    observed = DatabaseManager.get_analytics_summary()["top_behaviours"]
    for row in coverage:
        row["events_recorded"] = observed.get(row["behaviour_type"], 0)
    return {
        "detector_backend": processor.detector.backend,
        "behaviours": coverage,
        "counts": {
            "implemented": sum(1 for r in coverage if r["status"] == "IMPLEMENTED"),
            "partial": sum(1 for r in coverage if r["status"] == "PARTIALLY_IMPLEMENTED"),
            "requires_config": sum(1 for r in coverage if r["status"].startswith("REQUIRES")),
            "total": len(coverage),
        },
    }


@app.get("/api/videos")
def list_videos():
    videos = DatabaseManager.get_all_videos()
    for v in videos:
        incidents = DatabaseManager.get_incidents(video_id=v["id"], limit=1000)
        v["incident_count"] = len(incidents)
        v["critical_count"] = sum(1 for i in incidents if i["risk_level"] == "CRITICAL")
        v["high_count"] = sum(1 for i in incidents if i["risk_level"] == "HIGH")
        _decorate_video(v)
    return {"count": len(videos), "videos": videos}


@app.get("/api/videos/{video_id}")
def get_video_details(video_id: str):
    _validate_id(video_id, "video id")
    video = DatabaseManager.get_video_by_id(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    incidents = [
        _decorate_incident(i)
        for i in DatabaseManager.get_incidents(video_id=video_id, limit=1000)
    ]
    return {"video": _decorate_video(video), "incidents": incidents}


@app.delete("/api/videos/{video_id}")
def delete_video(video_id: str):
    _validate_id(video_id, "video id")
    removed = DatabaseManager.delete_video(video_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Video not found")
    TASK_STATUS.pop(video_id, None)
    return {"status": "deleted", "video_id": video_id}


@app.get("/api/videos/{video_id}/status")
def get_video_processing_status(video_id: str):
    _validate_id(video_id, "video id")
    if video_id in TASK_STATUS:
        return TASK_STATUS[video_id]
    video = DatabaseManager.get_video_by_id(video_id)
    if video:
        count = DatabaseManager.count_incidents(video_id=video_id)
        return {
            "video_id": video_id,
            "filename": video["filename"],
            "status": video.get("status") or "completed",
            "progress_percent": 100 if video.get("status") == "completed" else 0,
            "incidents_count": count,
            "error": video.get("error_message"),
        }
    return {"video_id": video_id, "status": "not_found", "progress_percent": 0}


@app.post("/api/videos/upload", status_code=202)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    bay: str = Form("Unassigned Bay"),
    shift: str = Form("Unassigned Shift"),
    camera_id: str = Form("CAM-01"),
    floor_condition: str = Form("unknown"),
    dock_transfer: bool = Form(False),
    staging_zone: Optional[str] = Form(None),
):
    """
    Ingest a warehouse video and start background analysis.

    Scene context (bay, shift, floor condition, dock transfer, staging zone) is
    supplied by the operator here rather than inferred from the filename, and is
    what the wet-floor, dock and designated-area detectors reason against.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    safe_name = _safe_filename(file.filename)
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in config.ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{ext or 'unknown'}'. "
                f"Allowed: {', '.join(sorted(config.ALLOWED_VIDEO_EXTENSIONS))}"
            ),
        )

    if floor_condition.lower() not in {"dry", "wet", "unknown"}:
        raise HTTPException(status_code=400, detail="floor_condition must be dry, wet or unknown")

    zone = None
    if staging_zone:
        try:
            parsed = json.loads(staging_zone)
            if not isinstance(parsed, list) or len(parsed) < 3:
                raise ValueError("need at least 3 points")
            zone = [[float(p[0]), float(p[1])] for p in parsed]
        except (ValueError, TypeError, IndexError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"staging_zone must be JSON [[x,y],...] in 0-1 coordinates ({exc})",
            ) from exc

    video_id = f"vid_{uuid.uuid4().hex[:8]}"
    # Prefix with the video id so re-uploading the same filename never
    # overwrites an earlier recording or its analysis.
    stored_name = f"{video_id}_{safe_name}"
    save_path = os.path.join(config.RAW_VIDEOS_DIR, stored_name)

    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    try:
        with open(save_path, "wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {config.MAX_UPLOAD_MB} MB upload limit",
                    )
                buffer.write(chunk)
    except HTTPException:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise
    except OSError as exc:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=500, detail=f"Could not store upload: {exc}") from exc

    if written == 0:
        os.remove(save_path)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    scene = SceneContext(
        bay=bay.strip()[:80] or "Unassigned Bay",
        shift=shift.strip()[:40] or "Unassigned Shift",
        camera_id=camera_id.strip()[:40] or "CAM-01",
        floor_condition=floor_condition,
        dock_transfer=dock_transfer,
        staging_zone=zone,
    )

    TASK_STATUS[video_id] = {
        "video_id": video_id,
        "filename": stored_name,
        "status": "processing",
        "progress_percent": 1,
        "current_frame": 0,
        "total_frames": 0,
        "incidents_count": 0,
        "stage": "queued",
    }
    background_tasks.add_task(
        _safe_process, save_path, video_id, True, None, scene
    )

    logger.info("Accepted upload %s (%.1f MB) as %s", safe_name, written / 1e6, video_id)
    return {
        "status": "processing",
        "video_id": video_id,
        "filename": stored_name,
        "size_bytes": written,
        "scene": scene.to_dict(),
        "message": f"'{safe_name}' uploaded. AI analysis started.",
    }


def _safe_process(path, video_id, annotate, stride, scene) -> None:
    """Background entry point: never let an exception escape unlogged."""
    try:
        processor.process_video(path, video_id, annotate, stride, scene)
    except Exception:  # noqa: BLE001
        logger.exception("Background analysis failed for %s", video_id)


@app.post("/api/videos/{video_id}/analyze", status_code=202)
def analyze_video(video_id: str, background_tasks: BackgroundTasks):
    _validate_id(video_id, "video id")
    video = DatabaseManager.get_video_by_id(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if not os.path.exists(video["filepath"]):
        raise HTTPException(status_code=410, detail="Source video file is no longer on disk")

    flags = video.get("scene_flags")
    if isinstance(flags, str):
        try:
            flags = json.loads(flags or "{}")
        except ValueError:
            flags = {}
    scene = SceneContext(
        bay=video.get("bay") or "Unassigned Bay",
        shift=video.get("shift") or "Unassigned Shift",
        camera_id=video.get("camera_id") or "CAM-01",
        floor_condition=(flags or {}).get("floor_condition", "unknown"),
        dock_transfer=(flags or {}).get("dock_transfer", False),
    )
    TASK_STATUS[video_id] = {
        "video_id": video_id,
        "filename": video["filename"],
        "status": "processing",
        "progress_percent": 1,
        "incidents_count": 0,
        "stage": "queued",
    }
    background_tasks.add_task(_safe_process, video["filepath"], video_id, True, None, scene)
    return {"status": "processing", "video_id": video_id}


@app.get("/api/incidents")
def list_incidents(
    video_id: Optional[str] = None,
    risk_level: Optional[str] = None,
    behaviour_type: Optional[str] = None,
    bay: Optional[str] = None,
    shift: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    if risk_level and risk_level.upper() not in VALID_RISK_LEVELS:
        raise HTTPException(
            status_code=400,
            detail=f"risk_level must be one of {sorted(VALID_RISK_LEVELS)}",
        )
    if limit < 1 or limit > 1000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 1000")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")

    incidents = [
        _decorate_incident(i)
        for i in DatabaseManager.get_incidents(
            video_id=video_id,
            risk_level=risk_level,
            behaviour_type=behaviour_type,
            bay=bay,
            shift=shift,
            search=search,
            limit=limit,
            offset=offset,
        )
    ]
    return {"count": len(incidents), "offset": offset, "incidents": incidents}


@app.get("/api/incidents/{incident_id}")
def get_incident_details(incident_id: str):
    _validate_id(incident_id, "incident id")
    incident = DatabaseManager.get_incident_by_id(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    return _decorate_incident(incident)


@app.patch("/api/incidents/{incident_id}/review")
def review_incident(incident_id: str, req: ReviewRequest):
    """
    Record a human review decision.

    This is the only path by which an incident can be marked as confirmed
    damage: the vision pipeline itself never asserts damage occurred.
    """
    _validate_id(incident_id, "incident id")
    if not DatabaseManager.set_review_status(incident_id, req.status, req.note):
        raise HTTPException(status_code=404, detail="Incident not found")
    return _decorate_incident(DatabaseManager.get_incident_by_id(incident_id))


@app.get("/api/analytics")
def get_analytics():
    return DatabaseManager.get_analytics_summary()


@app.get("/api/prevention")
def prevention_insights():
    """
    Prevention & learning view: recurring behaviours, the bays and shifts that
    need attention, and the training topics those imply. Derived entirely from
    recorded incidents.
    """
    summary = DatabaseManager.get_analytics_summary()
    behaviours = summary["top_behaviours"]
    total = max(1, summary["total_incidents"])

    training_map = {
        "product_drop": "Controlled lowering and two-person lifting for heavy packages",
        "product_drag": "Correct use of trolleys and hand pallet trucks",
        "product_throw": "Lift-and-place discipline at vehicle transfer points",
        "rolling_product": "Why rolling damages package corners; correct alternatives",
        "improper_stacking": "Stacking order, base support and overhang limits",
        "stepping_on_carton": "Working-at-height alternatives and walkway discipline",
        "unsupported_handling": "Manual handling limits and equipment provisioning",
        "wet_floor_hazard": "Floor condition reporting and stop-work authority",
        "orientation_violation": "Reading handling arrows and orientation labels",
        "dock_level_hazard": "Dock leveller and bridge plate procedure",
        "outside_designated_area": "Staging discipline and zone marking",
        "unsafe_loading_sequence": "Planned loading sequence and one-at-a-time handover",
    }

    recurring = [
        {
            "behaviour_type": b,
            "occurrences": c,
            "share_percent": round(c / total * 100, 1),
            "training_topic": training_map.get(b, "Site handling SOP refresher"),
        }
        for b, c in behaviours.items()
        if c >= 2
    ]

    hotspots = [b for b in summary["by_bay"] if b["high_risk"] > 0][:5]
    return {
        "recurring_behaviours": recurring,
        "high_risk_locations": hotspots,
        "shift_comparison": summary["by_shift"],
        "training_opportunities": [r["training_topic"] for r in recurring],
        "baseline": {
            "high_risk_events_per_minute": summary["high_risk_events_per_minute"],
            "total_footage_minutes": summary["total_footage_minutes"],
            "note": (
                "Improvement is tracked by comparing this rate across later shifts. "
                "A single pilot batch establishes the baseline only."
            ),
        },
        "review_breakdown": summary["review_breakdown"],
    }


@app.post("/api/assistant/chat")
def chat_with_assistant(req: AssistantQueryRequest):
    """Supervisor assistant, answering strictly from recorded incidents."""
    if req.video_id:
        _validate_id(req.video_id, "video id")
    return AIAssistant.answer_query(query=req.query, video_id=req.video_id)


# Serve the SPA for unknown non-API paths so client-side routing works.
@app.get("/{full_path:path}")
def spa_fallback(full_path: str):
    if full_path.startswith(("api/", "static/", "assets/", "docs", "openapi.json")):
        raise HTTPException(status_code=404, detail="Not found")
    index = os.path.join(_DASHBOARD_DIST, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    raise HTTPException(status_code=404, detail="Not found")
