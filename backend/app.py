"""
VisionGuard FastAPI Backend Service
Provides REST endpoints for video ingestion, incident timelines, evidence retrieval,
analytics summaries, background AI video processing, and grounded AI assistant.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import shutil
import uuid

from backend.database.db import init_db, DatabaseManager
from video.processor import VideoProcessor, TASK_STATUS
from assistant.llm import AIAssistant

app = FastAPI(
    title="VisionGuard — AI Warehouse Intelligence API",
    description="Backend API for AI Video Intelligence, Behaviour Understanding, and Damage Prevention",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
init_db()

# Ensure directories exist and mount static files
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/evidence", exist_ok=True)

app.mount("/static/evidence", StaticFiles(directory="data/evidence"), name="evidence")
app.mount("/static/raw", StaticFiles(directory="data/raw"), name="raw_videos")
app.mount("/static/processed", StaticFiles(directory="data/processed"), name="processed_videos")

# Serve dashboard frontend if built
if os.path.exists("dashboard/dist"):
    app.mount("/assets", StaticFiles(directory="dashboard/dist/assets"), name="assets")

@app.get("/")
def serve_root():
    if os.path.exists("dashboard/dist/index.html"):
        return FileResponse("dashboard/dist/index.html")
    return {"message": "VisionGuard API is running. Build frontend with 'cd dashboard && npm run build'"}

processor = VideoProcessor(conf_threshold=0.25)

class AssistantQueryRequest(BaseModel):
    query: str
    video_id: Optional[str] = None

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "service": "VisionGuard AI Warehouse Intelligence",
        "version": "1.0.0"
    }

@app.get("/api/videos")
def list_videos():
    videos = DatabaseManager.get_all_videos()
    for v in videos:
        # Attach incident counts & clean video URL
        incidents = DatabaseManager.get_incidents(video_id=v["id"])
        v["incident_count"] = len(incidents)
        v["critical_count"] = sum(1 for i in incidents if i["risk_level"] == "CRITICAL")
        v["high_count"] = sum(1 for i in incidents if i["risk_level"] == "HIGH")
        v["video_url"] = f"/static/raw/{v['filename']}"
    return {"videos": videos}

@app.get("/api/videos/{video_id}")
def get_video_details(video_id: str):
    video = DatabaseManager.get_video_by_id(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    incidents = DatabaseManager.get_incidents(video_id=video_id)
    video["video_url"] = f"/static/raw/{video['filename']}"
    return {
        "video": video,
        "incidents": incidents
    }

@app.get("/api/videos/{video_id}/status")
def get_video_processing_status(video_id: str):
    """
    Poll live processing percentage and status of background AI analysis.
    """
    if video_id in TASK_STATUS:
        return TASK_STATUS[video_id]
    
    # Check if already in DB
    video = DatabaseManager.get_video_by_id(video_id)
    if video:
        incidents = DatabaseManager.get_incidents(video_id=video_id)
        return {
            "video_id": video_id,
            "filename": video["filename"],
            "status": "completed",
            "progress_percent": 100,
            "incidents_count": len(incidents)
        }
    return {"status": "not_found", "progress_percent": 0}

@app.post("/api/videos/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a new warehouse video and trigger background AI analysis.
    """
    video_id = f"vid_{uuid.uuid4().hex[:8]}"
    clean_filename = file.filename.replace(" ", "_")
    save_path = os.path.join("data/raw", f"{clean_filename}")
    
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    TASK_STATUS[video_id] = {
        "video_id": video_id,
        "filename": clean_filename,
        "status": "processing",
        "progress_percent": 5,
        "current_frame": 0,
        "total_frames": 100,
        "incidents_count": 0
    }

    # Execute AI processing in background task
    background_tasks.add_task(
        processor.process_video,
        save_path,
        video_id,
        True,
        2
    )

    return {
        "status": "processing",
        "video_id": video_id,
        "filename": clean_filename,
        "message": f"Video '{file.filename}' uploaded successfully. AI analysis pipeline started."
    }

@app.post("/api/videos/{video_id}/analyze")
def analyze_video(video_id: str, background_tasks: BackgroundTasks):
    video = DatabaseManager.get_video_by_id(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    background_tasks.add_task(
        processor.process_video,
        video["filepath"],
        video_id,
        True,
        2
    )
    return {"status": "processing", "video_id": video_id}

@app.get("/api/incidents")
def list_incidents(
    video_id: Optional[str] = None,
    risk_level: Optional[str] = None,
    behaviour_type: Optional[str] = None,
    limit: int = 100
):
    incidents = DatabaseManager.get_incidents(
        video_id=video_id,
        risk_level=risk_level,
        behaviour_type=behaviour_type,
        limit=limit
    )
    return {
        "count": len(incidents),
        "incidents": incidents
    }

@app.get("/api/incidents/{incident_id}")
def get_incident_details(incident_id: str):
    incident = DatabaseManager.get_incident_by_id(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    return incident

@app.get("/api/analytics")
def get_analytics():
    summary = DatabaseManager.get_analytics_summary()
    return summary

@app.post("/api/assistant/chat")
def chat_with_assistant(req: AssistantQueryRequest):
    """
    Supervisor AI Assistant: grounded in SQLite database incidents.
    """
    answer = AIAssistant.answer_query(query=req.query, video_id=req.video_id)
    return answer
