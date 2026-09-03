"""
VisionGuard FastAPI Backend Service
Provides REST endpoints for video ingestion, incident timelines, evidence retrieval,
analytics summaries, and the grounded AI Supervisor Assistant.
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
from video.processor import VideoProcessor
from assistant.llm import AIAssistant

app = FastAPI(
    title="VisionGuard — AI Warehouse Intelligence API",
    description="Backend API for AI Video Intelligence, Behaviour Understanding, and Damage Prevention",
    version="1.0.0"
)

# CORS configuration for modern frontend integration
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
        # Attach incident counts
        incidents = DatabaseManager.get_incidents(video_id=v["id"])
        v["incident_count"] = len(incidents)
        v["critical_count"] = sum(1 for i in incidents if i["risk_level"] == "CRITICAL")
        v["high_count"] = sum(1 for i in incidents if i["risk_level"] == "HIGH")
    return {"videos": videos}

@app.get("/api/videos/{video_id}")
def get_video_details(video_id: str):
    video = DatabaseManager.get_video_by_id(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    incidents = DatabaseManager.get_incidents(video_id=video_id)
    return {
        "video": video,
        "incidents": incidents
    }

@app.post("/api/videos/upload")
async def upload_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Upload a new warehouse video and trigger AI analysis.
    """
    video_id = f"vid_{uuid.uuid4().hex[:8]}"
    file_ext = os.path.splitext(file.filename)[1] or ".mp4"
    save_path = os.path.join("data/raw", f"{video_id}_{file.filename}")
    
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process video
    result = processor.process_video(
        video_path=save_path,
        video_id=video_id,
        generate_annotated_video=True
    )
    return {
        "message": "Video uploaded and analyzed successfully",
        "result": result
    }

@app.post("/api/videos/{video_id}/analyze")
def analyze_video(video_id: str):
    video = DatabaseManager.get_video_by_id(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    result = processor.process_video(
        video_path=video["filepath"],
        video_id=video_id,
        generate_annotated_video=True
    )
    return result

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
