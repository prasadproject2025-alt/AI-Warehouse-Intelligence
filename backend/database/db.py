"""
Database Manager for VisionGuard.

SQLite persistence for videos, incidents and the analytics aggregates the
dashboard and the grounded assistant read from. Every aggregate is computed
from real rows; nothing here synthesises numbers.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional

import config

logger = logging.getLogger(__name__)

DB_PATH = config.DATABASE_PATH

# SQLite allows one writer at a time; the pipeline writes from a worker thread
# while the API reads from request threads, so writes are serialised here.
_write_lock = threading.Lock()


def get_connection() -> sqlite3.Connection:
    parent = os.path.dirname(DB_PATH)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Run additive migrations, tolerating statements that already applied."""
    path = os.path.join(os.path.dirname(__file__), "migrations.sql")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    # Strip comments first: a trailing "-- note" after a semicolon would
    # otherwise be prepended to the following statement and hide it.
    cleaned = "\n".join(line.split("--", 1)[0] for line in raw.splitlines())
    for statement in cleaned.split(";"):
        stmt = statement.strip()
        if not stmt:
            continue
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError as exc:
            msg = str(exc).lower()
            if "duplicate column" in msg or "already exists" in msg:
                continue
            logger.warning("Migration statement skipped (%s): %s", exc, stmt[:80])


def init_db() -> None:
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()
    with _write_lock, get_connection() as conn:
        conn.executescript(schema_sql)
        _apply_migrations(conn)
        conn.commit()


def _row_to_incident(row: sqlite3.Row) -> Dict[str, Any]:
    item = dict(row)
    for json_field, default in (
        ("bounding_box", []),
        ("risk_factors", []),
        ("evidence_stages", []),
    ):
        raw = item.get(json_field)
        if raw:
            try:
                item[json_field] = json.loads(raw)
            except (TypeError, ValueError):
                item[json_field] = default
        else:
            item[json_field] = default
    return item


class DatabaseManager:
    # ---------------------------------------------------------------- videos
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
        annotated_path: Optional[str] = None,
        camera_id: str = "CAM-01",
        bay: str = "Unassigned Bay",
        shift: str = "Unassigned Shift",
        error_message: Optional[str] = None,
        processing_seconds: Optional[float] = None,
        detector_backend: Optional[str] = None,
        frames_analysed: Optional[int] = None,
        scene_flags: Optional[Dict[str, Any]] = None,
        batch_id: Optional[str] = None,
    ) -> None:
        with _write_lock, get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO videos
                (id, filename, filepath, duration_sec, fps, frame_count, width, height, status,
                 annotated_filepath, camera_id, bay, shift, error_message, processing_seconds,
                 detector_backend, frames_analysed, scene_flags, batch_id)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    video_id, filename, filepath, duration, fps, frame_count, width, height,
                    status, annotated_path, camera_id, bay, shift, error_message,
                    processing_seconds, detector_backend, frames_analysed,
                    json.dumps(scene_flags or {}), batch_id,
                ),
            )
            conn.commit()

    @staticmethod
    def update_video_status(
        video_id: str, status: str, error_message: Optional[str] = None
    ) -> None:
        with _write_lock, get_connection() as conn:
            conn.execute(
                "UPDATE videos SET status = ?, error_message = ? WHERE id = ?",
                (status, error_message, video_id),
            )
            conn.commit()

    @staticmethod
    def get_all_videos(batch_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with get_connection() as conn:
            if batch_id:
                rows = conn.execute(
                    "SELECT * FROM videos WHERE batch_id = ? ORDER BY processed_at DESC",
                    (batch_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM videos ORDER BY processed_at DESC"
                ).fetchall()
            return [dict(r) for r in rows]

    @staticmethod
    def get_video_by_id(video_id: str) -> Optional[Dict[str, Any]]:
        with get_connection() as conn:
            row = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
            return dict(row) if row else None

    @staticmethod
    def delete_video(video_id: str) -> int:
        with _write_lock, get_connection() as conn:
            conn.execute("DELETE FROM incidents WHERE video_id = ?", (video_id,))
            cur = conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))
            conn.commit()
            return cur.rowcount

    # ------------------------------------------------------------- incidents
    @staticmethod
    def save_incident(incident_data: Dict[str, Any]) -> None:
        with _write_lock, get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO incidents
                (id, video_id, timestamp_sec, frame_idx, behaviour_type, object_track_id,
                 operator_track_id, confidence, risk_level, risk_score, evidence_description,
                 root_cause, recommended_action, bounding_box, evidence_image_path,
                 camera_id, bay, shift, risk_factors, evidence_stages, evidence_clip_path,
                 evidence_tier, review_status, duration_sec, batch_id)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
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
                    incident_data.get("evidence_image_path"),
                    incident_data.get("camera_id"),
                    incident_data.get("bay"),
                    incident_data.get("shift"),
                    json.dumps(incident_data.get("risk_factors", [])),
                    json.dumps(incident_data.get("evidence_stages", [])),
                    incident_data.get("evidence_clip_path"),
                    incident_data.get("evidence_tier", "OBSERVED_BEHAVIOUR"),
                    incident_data.get("review_status", "PENDING_REVIEW"),
                    incident_data.get("duration_sec", 0.0),
                    incident_data.get("batch_id"),
                ),
            )
            conn.commit()

    @staticmethod
    def get_incidents(
        video_id: Optional[str] = None,
        risk_level: Optional[str] = None,
        behaviour_type: Optional[str] = None,
        bay: Optional[str] = None,
        shift: Optional[str] = None,
        batch_id: Optional[str] = None,
        since: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM incidents WHERE 1=1"
        params: List[Any] = []
        if video_id:
            query += " AND video_id = ?"
            params.append(video_id)
        if risk_level:
            query += " AND risk_level = ?"
            params.append(risk_level.upper())
        if behaviour_type:
            query += " AND behaviour_type = ?"
            params.append(behaviour_type)
        if bay:
            query += " AND bay = ?"
            params.append(bay)
        if shift:
            query += " AND shift = ?"
            params.append(shift)
        if batch_id:
            query += " AND batch_id = ?"
            params.append(batch_id)
        if since:
            query += " AND created_at >= ?"
            params.append(since)
        if search:
            query += (
                " AND (behaviour_type LIKE ? OR evidence_description LIKE ?"
                " OR root_cause LIKE ?)"
            )
            like = "%" + search + "%"
            params.extend([like, like, like])
        query += " ORDER BY video_id ASC, timestamp_sec ASC LIMIT ? OFFSET ?"
        params.extend([max(1, min(limit, 1000)), max(0, offset)])

        with get_connection() as conn:
            return [_row_to_incident(r) for r in conn.execute(query, params).fetchall()]

    @staticmethod
    def count_incidents(**filters: Any) -> int:
        clauses, params = ["1=1"], []
        for col in ("video_id", "risk_level", "behaviour_type", "bay", "shift"):
            if filters.get(col):
                clauses.append(col + " = ?")
                params.append(filters[col])
        with get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM incidents WHERE " + " AND ".join(clauses), params
            ).fetchone()
            return int(row[0])

    @staticmethod
    def get_incident_by_id(incident_id: str) -> Optional[Dict[str, Any]]:
        with get_connection() as conn:
            row = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
            return _row_to_incident(row) if row else None

    @staticmethod
    def set_review_status(
        incident_id: str, status: str, note: Optional[str] = None
    ) -> bool:
        with _write_lock, get_connection() as conn:
            cur = conn.execute(
                "UPDATE incidents SET review_status = ?, reviewer_note = ?, "
                "reviewed_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status, note, incident_id),
            )
            conn.commit()
            return cur.rowcount > 0

    # ------------------------------------------------------------- analytics
    @staticmethod
    def get_behaviour_history(behaviour_type: str, bay: Optional[str] = None) -> int:
        """Prior occurrences of a behaviour; the risk engine uses this for recurrence weighting."""
        with get_connection() as conn:
            if bay:
                row = conn.execute(
                    "SELECT COUNT(*) FROM incidents WHERE behaviour_type = ? AND bay = ?",
                    (behaviour_type, bay),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) FROM incidents WHERE behaviour_type = ?",
                    (behaviour_type,),
                ).fetchone()
            return int(row[0])

    @staticmethod
    def get_analytics_summary(batch_id: Optional[str] = None) -> Dict[str, Any]:
        # A batch scope restricts every aggregate to one analysis run, so the
        # dashboard can show exactly the dataset the user just processed rather
        # than everything ever stored.
        iw = " WHERE batch_id = ?" if batch_id else ""
        vw = " AND batch_id = ?" if batch_id else ""
        # Qualified form for the query that joins videos and incidents,
        # where a bare batch_id would be ambiguous.
        vw_q = " AND v.batch_id = ?" if batch_id else ""
        ip = (batch_id,) if batch_id else ()

        with get_connection() as conn:
            total_incidents = conn.execute(
                f"SELECT COUNT(*) FROM incidents{iw}", ip
            ).fetchone()[0]
            videos_count = conn.execute(
                f"SELECT COUNT(*) FROM videos WHERE status='completed'{vw}", ip
            ).fetchone()[0]
            total_minutes = conn.execute(
                f"SELECT COALESCE(SUM(duration_sec),0)/60.0 FROM videos WHERE status='completed'{vw}",
                ip,
            ).fetchone()[0]

            risk_counts = {
                r[0]: r[1]
                for r in conn.execute(
                    f"SELECT risk_level, COUNT(*) FROM incidents{iw} GROUP BY risk_level", ip
                ).fetchall()
            }
            behaviour_counts = {
                b[0]: b[1]
                for b in conn.execute(
                    f"SELECT behaviour_type, COUNT(*) FROM incidents{iw} "
                    "GROUP BY behaviour_type ORDER BY 2 DESC", ip
                ).fetchall()
            }
            by_bay = [
                {"bay": r[0] or "Unassigned Bay", "total": r[1], "high_risk": r[2] or 0}
                for r in conn.execute(
                    "SELECT bay, COUNT(*), "
                    "SUM(CASE WHEN risk_level IN ('HIGH','CRITICAL') THEN 1 ELSE 0 END) "
                    f"FROM incidents{iw} GROUP BY bay ORDER BY 3 DESC, 2 DESC", ip
                ).fetchall()
            ]
            by_shift = [
                {"shift": r[0] or "Unassigned Shift", "total": r[1], "high_risk": r[2] or 0}
                for r in conn.execute(
                    "SELECT shift, COUNT(*), "
                    "SUM(CASE WHEN risk_level IN ('HIGH','CRITICAL') THEN 1 ELSE 0 END) "
                    f"FROM incidents{iw} GROUP BY shift ORDER BY 3 DESC, 2 DESC", ip
                ).fetchall()
            ]
            by_video = [
                {
                    "video_id": r[0],
                    "filename": r[1],
                    "duration_sec": r[2] or 0.0,
                    "total": r[3],
                    "high_risk": r[4] or 0,
                }
                for r in conn.execute(
                    "SELECT v.id, v.filename, v.duration_sec, COUNT(i.id), "
                    "SUM(CASE WHEN i.risk_level IN ('HIGH','CRITICAL') THEN 1 ELSE 0 END) "
                    "FROM videos v LEFT JOIN incidents i ON i.video_id = v.id "
                    f"WHERE v.status='completed'{vw_q} GROUP BY v.id ORDER BY 4 DESC", ip
                ).fetchall()
            ]
            review = {
                r[0] or "PENDING_REVIEW": r[1]
                for r in conn.execute(
                    f"SELECT review_status, COUNT(*) FROM incidents{iw} GROUP BY review_status", ip
                ).fetchall()
            }

        high_and_critical = risk_counts.get("HIGH", 0) + risk_counts.get("CRITICAL", 0)
        # Rate per analysed minute of footage: comparable across videos of very
        # different lengths, unlike a raw count.
        rate = (
            round(high_and_critical / total_minutes, 2)
            if total_minutes and total_minutes > 0
            else 0.0
        )

        return {
            "total_videos_analyzed": videos_count,
            "total_footage_minutes": round(total_minutes or 0.0, 2),
            "total_incidents": total_incidents,
            "risk_breakdown": {
                "CRITICAL": risk_counts.get("CRITICAL", 0),
                "HIGH": risk_counts.get("HIGH", 0),
                "MEDIUM": risk_counts.get("MEDIUM", 0),
                "LOW": risk_counts.get("LOW", 0),
            },
            "top_behaviours": behaviour_counts,
            "by_bay": by_bay,
            "by_shift": by_shift,
            "by_video": by_video,
            "review_breakdown": review,
            "high_risk_events_per_minute": rate,
            "intervention_opportunities": high_and_critical,
        }

    # ----------------------------------------------------------------- batches
    @staticmethod
    def list_batches() -> List[Dict[str, Any]]:
        """Analysis runs, newest first, with what each one produced."""
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT v.batch_id,
                          COUNT(DISTINCT v.id)                       AS videos,
                          MIN(v.processed_at)                        AS started_at,
                          COALESCE(SUM(v.duration_sec), 0) / 60.0    AS minutes,
                          (SELECT COUNT(*) FROM incidents i WHERE i.batch_id = v.batch_id) AS incidents
                   FROM videos v
                   WHERE v.batch_id IS NOT NULL AND v.status = 'completed'
                   GROUP BY v.batch_id
                   ORDER BY started_at DESC"""
            ).fetchall()
            return [
                {
                    "batch_id": r["batch_id"],
                    "videos": r["videos"],
                    "incidents": r["incidents"],
                    "footage_minutes": round(r["minutes"] or 0.0, 2),
                    "started_at": r["started_at"],
                }
                for r in rows
            ]

    @staticmethod
    def clear_analysis(batch_id: Optional[str] = None) -> int:
        """
        Remove analysis rows. Scoped to a batch when given, otherwise every
        non-live video. Live sessions are left alone so a running monitor is
        not silently orphaned.
        """
        with _write_lock, get_connection() as conn:
            if batch_id:
                conn.execute("DELETE FROM incidents WHERE batch_id = ?", (batch_id,))
                cur = conn.execute("DELETE FROM videos WHERE batch_id = ?", (batch_id,))
            else:
                conn.execute(
                    "DELETE FROM incidents WHERE video_id IN "
                    "(SELECT id FROM videos WHERE id NOT LIKE 'live_%')"
                )
                cur = conn.execute("DELETE FROM videos WHERE id NOT LIKE 'live_%'")
            conn.commit()
            return cur.rowcount
