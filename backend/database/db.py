"""
Database Manager for VisionGuard
Handles SQLite initialization, video records, incident management, and analytical aggregates.
"""

import sqlite3
import os
import json
from typing import List, Dict, Any, Optional

DB_PATH = os.environ.get("DATABASE_URL", "sqlite:///./data/visionguard.db").replace("sqlite:///", "")

def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()
    with get_connection() as conn:
        conn.executescript(schema_sql)
        conn.commit()

class DatabaseManager:
    @staticmethod
    def save_video(
        video_id: str,
        filename: str,
        filepath: str,
        duration: float,
        fps: float,
        frame_count: int,
        width: int,
        height: int,
        status: str = "completed",
        annotated_path: Optional[str] = None
    ):
        with get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO videos 
                (id, filename, filepath, duration_sec, fps, frame_count, width, height, status, annotated_filepath)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (video_id, filename, filepath, duration, fps, frame_count, width, height, status, annotated_path))
            conn.commit()

    @staticmethod
    def save_incident(incident_data: Dict[str, Any]):
        with get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO incidents
                (id, video_id, timestamp_sec, frame_idx, behaviour_type, object_track_id, 
                 operator_track_id, confidence, risk_level, risk_score, evidence_description, 
                 root_cause, recommended_action, bounding_box, evidence_image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                incident_data["id"],
                incident_data["video_id"],
                incident_data["timestamp_sec"],
                incident_data["frame_idx"],
                incident_data["behaviour_type"],
                incident_data.get("object_track_id"),
                incident_data.get("operator_track_id"),
                incident_data["confidence"],
                incident_data["risk_level"],
                incident_data["risk_score"],
                incident_data["evidence_description"],
                incident_data["root_cause"],
                incident_data["recommended_action"],
                json.dumps(incident_data["bounding_box"]),
                incident_data.get("evidence_image_path")
            ))
            conn.commit()

    @staticmethod
    def get_all_videos() -> List[Dict[str, Any]]:
        with get_connection() as conn:
            rows = conn.execute("SELECT * FROM videos ORDER BY processed_at DESC").fetchall()
            return [dict(r) for r in rows]

    @staticmethod
    def get_video_by_id(video_id: str) -> Optional[Dict[str, Any]]:
        with get_connection() as conn:
            row = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
            return dict(row) if row else None

    @staticmethod
    def get_incidents(
        video_id: Optional[str] = None,
        risk_level: Optional[str] = None,
        behaviour_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM incidents WHERE 1=1"
        params = []
        if video_id:
            query += " AND video_id = ?"
            params.append(video_id)
        if risk_level:
            query += " AND risk_level = ?"
            params.append(risk_level.upper())
        if behaviour_type:
            query += " AND behaviour_type = ?"
            params.append(behaviour_type)
        query += " ORDER BY timestamp_sec ASC LIMIT ?"
        params.append(limit)

        with get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["bounding_box"] = json.loads(item["bounding_box"]) if item.get("bounding_box") else []
                results.append(item)
            return results

    @staticmethod
    def get_incident_by_id(incident_id: str) -> Optional[Dict[str, Any]]:
        with get_connection() as conn:
            row = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
            if not row:
                return None
            item = dict(row)
            item["bounding_box"] = json.loads(item["bounding_box"]) if item.get("bounding_box") else []
            return item

    @staticmethod
    def get_analytics_summary() -> Dict[str, Any]:
        with get_connection() as conn:
            total_incidents = conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
            
            risk_counts = {}
            for r in conn.execute("SELECT risk_level, COUNT(*) FROM incidents GROUP BY risk_level").fetchall():
                risk_counts[r[0]] = r[1]

            behaviour_counts = {}
            for b in conn.execute("SELECT behaviour_type, COUNT(*) FROM incidents GROUP BY behaviour_type ORDER BY 2 DESC").fetchall():
                behaviour_counts[b[0]] = b[1]

            videos_count = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]

            # Calculate safe handling discipline index (0-100)
            critical_weight = risk_counts.get("CRITICAL", 0) * 10
            high_weight = risk_counts.get("HIGH", 0) * 5
            med_weight = risk_counts.get("MEDIUM", 0) * 2
            penalty = critical_weight + high_weight + med_weight
            handling_discipline_score = max(35.0, round(100.0 - min(65.0, penalty * 0.8), 1))

            return {
                "total_videos_analyzed": videos_count,
                "total_incidents": total_incidents,
                "risk_breakdown": {
                    "CRITICAL": risk_counts.get("CRITICAL", 0),
                    "HIGH": risk_counts.get("HIGH", 0),
                    "MEDIUM": risk_counts.get("MEDIUM", 0),
                    "LOW": risk_counts.get("LOW", 0)
                },
                "top_behaviours": behaviour_counts,
                "handling_discipline_score": handling_discipline_score,
                "damage_prevention_potential": f"{risk_counts.get('CRITICAL', 0) + risk_counts.get('HIGH', 0)} High-Risk Interventions Enabled"
            }
