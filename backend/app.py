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
import time
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
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

import config
from assistant.llm import AIAssistant
from backend.database.db import DatabaseManager, init_db
from behaviour.behaviour_engine import BehaviourEngine, SceneContext
from video import batch as batch_runner
from video.live import SOURCE_KINDS as LIVE_SOURCE_KINDS, LiveSessionManager
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

# Live sessions share the loaded model with the recorded pipeline; inference is
# serialised inside the session so the two cannot run concurrently on one model.
live_manager = LiveSessionManager(processor.detector, config.EVIDENCE_DIR)

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
        # Never cache the entry point. Asset filenames are content-hashed and
        # old ones are removed on rebuild, so a cached index.html points at a
        # bundle that no longer exists and the whole app fails to boot.
        return FileResponse(index, headers={"Cache-Control": "no-cache, must-revalidate"})
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
    if video_id.startswith("live_"):
        # A running session would otherwise keep writing incidents against
        # a row that no longer exists.
        live_manager.stop(video_id[len("live_"):])
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
    batch_id: Optional[str] = None,
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
            batch_id=batch_id,
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


@app.get("/api/batches")
def list_batches():
    """Analysis runs available to scope the dashboard to."""
    return {
        "batches": DatabaseManager.list_batches(),
        "active": batch_runner.active_batch(),
        "library_size": len(batch_runner.library_videos()),
    }


class BatchRunRequest(BaseModel):
    """
    Analyse a dataset in one tracked run.

    ``videos`` empty means the whole library. ``replace_existing`` clears prior
    non-live analysis so a run is reproducible instead of accumulating
    duplicates of the same footage.
    """

    videos: List[str] = Field(default_factory=list)
    replace_existing: bool = True


@app.post("/api/batches/run", status_code=202)
def run_batch(req: BatchRunRequest):
    library = batch_runner.library_videos()
    if not library:
        raise HTTPException(status_code=400, detail="No videos in the library to analyse")

    if req.videos:
        unknown = [v for v in req.videos if v not in library]
        if unknown:
            raise HTTPException(
                status_code=404, detail=f"Not in the library: {', '.join(unknown[:5])}"
            )
        selected = req.videos
    else:
        selected = library

    try:
        batch_id = batch_runner.start_batch(
            processor, selected, replace_existing=req.replace_existing
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    logger.info("Batch %s queued for %d video(s)", batch_id, len(selected))
    return batch_runner.get_status(batch_id)


@app.get("/api/batches/{batch_id}")
def batch_status(batch_id: str):
    _validate_id(batch_id, "batch id")
    state = batch_runner.get_status(batch_id)
    if not state:
        raise HTTPException(status_code=404, detail="No such batch")
    return state


@app.post("/api/batches/{batch_id}/cancel")
def cancel_batch(batch_id: str):
    _validate_id(batch_id, "batch id")
    if not batch_runner.cancel(batch_id):
        raise HTTPException(status_code=404, detail="No running batch with that id")
    return {"status": "cancelling", "batch_id": batch_id}


class ResetRequest(BaseModel):
    """Clear stored analysis. Source videos in the library are never touched."""

    delete_evidence: bool = True


@app.post("/api/reset")
def reset_analysis(req: ResetRequest):
    """
    Return the system to a clean slate.

    Removes every analysed video row, incident, batch and generated artefact,
    but never the source videos in the library, so the dataset can simply be
    analysed again. Any live session is stopped first, otherwise it would keep
    writing rows into the database that was just cleared.
    """
    for session in live_manager.list():
        live_manager.stop(session["session_id"])

    for state in list(batch_runner.BATCH_STATUS.values()):
        batch_runner.cancel(state["batch_id"])
    batch_runner.BATCH_STATUS.clear()

    removed = DatabaseManager.clear_analysis()
    TASK_STATUS.clear()

    files_removed = 0
    if req.delete_evidence:
        for directory in (config.EVIDENCE_DIR, config.CLIPS_DIR, config.PROCESSED_VIDEOS_DIR):
            if not os.path.isdir(directory):
                continue
            for name in os.listdir(directory):
                path = os.path.join(directory, name)
                # Confine deletion to the configured directory.
                if os.path.commonpath([os.path.abspath(path), directory]) != directory:
                    continue
                if os.path.isfile(path):
                    try:
                        os.remove(path)
                        files_removed += 1
                    except OSError:
                        logger.warning("Could not remove %s", path)

    logger.info("Reset: %d video row(s), %d artefact(s)", removed, files_removed)
    return {
        "status": "reset",
        "videos_removed": removed,
        "files_removed": files_removed,
        "library_size": len(batch_runner.library_videos()),
    }


@app.get("/api/analytics")
def get_analytics(batch_id: Optional[str] = None):
    """Shift analytics, optionally scoped to a single analysis run."""
    if batch_id:
        _validate_id(batch_id, "batch id")
    return DatabaseManager.get_analytics_summary(batch_id=batch_id)


@app.get("/api/prevention")
def prevention_insights(batch_id: Optional[str] = None):
    """
    Prevention & learning view: recurring behaviours, the bays and shifts that
    need attention, and the training topics those imply. Derived entirely from
    recorded incidents.
    """
    if batch_id:
        _validate_id(batch_id, "batch id")
    summary = DatabaseManager.get_analytics_summary(batch_id=batch_id)
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


# ----------------------------------------------------------------- live feed
class LiveStartRequest(BaseModel):
    """
    Start near-real-time analysis of a continuous source.

    ``source_kind``:
      * ``camera`` - a locally attached camera; ``source`` is its index ("0").
      * ``stream`` - an RTSP/HTTP CCTV feed; ``source`` is the URL.
      * ``file``   - a video under the configured raw directory, replayed at its
        natural rate. This exercises the real live path when no camera is
        present; it is labelled as a replay everywhere it is surfaced.
    """

    source_kind: str = Field(default="file")
    source: str = Field(default="", max_length=500)
    camera_id: str = Field(default="CAM-LIVE", max_length=60)
    bay: str = Field(default="Unassigned Bay", max_length=80)
    shift: str = Field(default="Unassigned Shift", max_length=60)
    floor_condition: str = Field(default="unknown")
    dock_transfer: bool = False
    loop_file: bool = True

    @field_validator("source_kind")
    @classmethod
    def known_kind(cls, v: str) -> str:
        kind = v.strip().lower()
        if kind not in LIVE_SOURCE_KINDS:
            raise ValueError(f"source_kind must be one of {list(LIVE_SOURCE_KINDS)}")
        return kind

    @field_validator("floor_condition")
    @classmethod
    def known_floor(cls, v: str) -> str:
        cond = (v or "unknown").strip().lower()
        if cond not in {"dry", "wet", "unknown"}:
            raise ValueError("floor_condition must be dry, wet or unknown")
        return cond


def _resolve_live_source(req: LiveStartRequest):
    """
    Turn a request into a capture source, refusing anything unsafe.

    A file source is confined to the configured raw directory so a request
    cannot make the server open an arbitrary path, and a stream must be a
    recognised video-transport URL.
    """
    if req.source_kind == "camera":
        raw = (req.source or "0").strip()
        if not raw.isdigit() or int(raw) > 8:
            raise HTTPException(status_code=400, detail="camera source must be an index 0-8")
        return int(raw)

    if req.source_kind == "stream":
        url = (req.source or "").strip()
        if not url.lower().startswith(("rtsp://", "rtmp://", "http://", "https://")):
            raise HTTPException(
                status_code=400,
                detail="stream source must be an rtsp://, rtmp://, http:// or https:// URL",
            )
        return url

    name = _safe_filename(req.source or "")
    if not name:
        raise HTTPException(status_code=400, detail="file source is required")
    candidates = [
        p for p in os.listdir(config.RAW_VIDEOS_DIR)
        if _safe_filename(p) == name or p == req.source
    ]
    if not candidates:
        raise HTTPException(status_code=404, detail=f"No such video in the library: {req.source}")
    path = os.path.join(config.RAW_VIDEOS_DIR, candidates[0])
    if os.path.commonpath([os.path.abspath(path), config.RAW_VIDEOS_DIR]) != config.RAW_VIDEOS_DIR:
        raise HTTPException(status_code=400, detail="Invalid file source")
    return path


@app.get("/api/live/sources")
def live_sources():
    """What this deployment can currently be pointed at."""
    library = sorted(
        f for f in os.listdir(config.RAW_VIDEOS_DIR)
        if os.path.splitext(f)[1].lower() in config.ALLOWED_VIDEO_EXTENSIONS
    )
    return {
        "source_kinds": list(LIVE_SOURCE_KINDS),
        "library": library,
        "active": live_manager.list(),
        "note": (
            "'file' replays a stored video at its natural rate through the live "
            "pipeline. It is a stand-in for a camera, not a separate code path - "
            "the analysis is identical."
        ),
    }


@app.post("/api/live/start", status_code=202)
def live_start(req: LiveStartRequest):
    source = _resolve_live_source(req)
    scene = SceneContext(
        bay=req.bay.strip() or "Unassigned Bay",
        shift=req.shift.strip() or "Unassigned Shift",
        camera_id=req.camera_id.strip() or "CAM-LIVE",
        floor_condition=req.floor_condition,
        dock_transfer=req.dock_transfer,
    )
    try:
        session = live_manager.start(source, req.source_kind, scene, loop_file=req.loop_file)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Surface an immediate failure (bad camera index, unreachable stream)
    # rather than returning a session that is already dead.
    for _ in range(20):
        if session.status in ("running", "error"):
            break
        time.sleep(0.1)
    if session.status == "error":
        raise HTTPException(status_code=502, detail=session.error or "Could not open source")

    logger.info("Live session %s started (%s)", session.session_id, req.source_kind)
    return session.snapshot()


@app.get("/api/live")
def live_list():
    return {"sessions": live_manager.list()}


@app.get("/api/live/{session_id}")
def live_status(session_id: str):
    _validate_id(session_id, "session id")
    session = live_manager.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="No such live session")
    return session.snapshot()


@app.get("/api/live/{session_id}/stream")
def live_stream(session_id: str):
    """Annotated frames as multipart JPEG, renderable in a plain <img>."""
    _validate_id(session_id, "session id")
    session = live_manager.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="No such live session")
    return StreamingResponse(
        session.mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.post("/api/live/{session_id}/stop")
def live_stop(session_id: str):
    _validate_id(session_id, "session id")
    session = live_manager.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="No such live session")
    session.stop()
    return session.snapshot()


# Serve the SPA for unknown non-API paths so client-side routing works.
@app.get("/{full_path:path}")
def spa_fallback(full_path: str):
    if full_path.startswith(("api/", "static/", "assets/", "docs", "openapi.json")):
        raise HTTPException(status_code=404, detail="Not found")
    index = os.path.join(_DASHBOARD_DIST, "index.html")
    if os.path.exists(index):
        return FileResponse(index, headers={"Cache-Control": "no-cache, must-revalidate"})
    raise HTTPException(status_code=404, detail="Not found")
