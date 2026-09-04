"""API contract tests: response shapes, status codes, validation and errors."""

import io

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from helpers import seed_incident, seed_video

client = TestClient(app)


# ------------------------------------------------------------------ health
def test_health_reports_the_active_detector_backend():
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "online"
    assert "detector_backend" in body
    assert isinstance(body["open_vocabulary"], bool)


def test_capabilities_are_generated_from_the_code():
    res = client.get("/api/capabilities")
    assert res.status_code == 200
    body = res.json()
    assert body["counts"]["total"] == 12
    statuses = {b["status"] for b in body["behaviours"]}
    # An honest report must contain something other than "everything works".
    assert statuses - {"IMPLEMENTED"}, "coverage must state real limitations"
    for b in body["behaviours"]:
        assert b["label"] and b["method"] and b["limitations"]
        assert isinstance(b["events_recorded"], int)


# ------------------------------------------------------------------ videos
def test_videos_list_is_empty_before_ingest():
    res = client.get("/api/videos")
    assert res.status_code == 200
    assert res.json() == {"count": 0, "videos": []}


def test_video_detail_includes_playback_urls_and_incidents():
    seed_video()
    seed_incident()
    res = client.get("/api/videos/vid_test01")
    assert res.status_code == 200
    body = res.json()
    assert body["video"]["video_url"].startswith("/static/raw/")
    assert len(body["incidents"]) == 1
    assert body["incidents"][0]["evidence_image_url"] == "/static/evidence/evidence_inc_test01.jpg"


def test_unknown_video_returns_404():
    assert client.get("/api/videos/vid_missing").status_code == 404


def test_malformed_video_id_is_rejected():
    res = client.get("/api/videos/..%2F..%2Fetc")
    assert res.status_code in (400, 404)


def test_status_of_unknown_video_is_reported_not_found():
    body = client.get("/api/videos/vid_nope/status").json()
    assert body["status"] == "not_found"


def test_delete_video_removes_its_incidents():
    seed_video()
    seed_incident()
    assert client.delete("/api/videos/vid_test01").status_code == 200
    assert client.get("/api/incidents").json()["count"] == 0
    assert client.delete("/api/videos/vid_test01").status_code == 404


# --------------------------------------------------------------- incidents
def test_incidents_can_be_filtered_and_searched():
    seed_video()
    seed_incident(id="inc_a", behaviour_type="product_drop", risk_level="HIGH")
    seed_incident(id="inc_b", behaviour_type="product_drag", risk_level="LOW",
                  evidence_description="Carton slid along the floor.")

    assert client.get("/api/incidents").json()["count"] == 2
    assert client.get("/api/incidents?risk_level=HIGH").json()["count"] == 1
    assert client.get("/api/incidents?behaviour_type=product_drag").json()["count"] == 1
    assert client.get("/api/incidents?search=slid").json()["count"] == 1
    assert client.get("/api/incidents?bay=Dock%2005").json()["count"] == 2
    assert client.get("/api/incidents?bay=Nowhere").json()["count"] == 0


def test_invalid_filters_return_400():
    assert client.get("/api/incidents?risk_level=EXTREME").status_code == 400
    assert client.get("/api/incidents?limit=0").status_code == 400
    assert client.get("/api/incidents?limit=99999").status_code == 400
    assert client.get("/api/incidents?offset=-5").status_code == 400


def test_incident_detail_exposes_the_risk_breakdown():
    seed_video()
    seed_incident()
    body = client.get("/api/incidents/inc_test01").json()
    assert body["risk_factors"][0]["name"] == "Drop height"
    assert body["evidence_stages"][0]["stage"] == "carried"
    assert body["evidence_tier"] == "POTENTIAL_RISK"


def test_unknown_incident_returns_404():
    assert client.get("/api/incidents/inc_missing").status_code == 404


# ------------------------------------------------------------------ review
def test_human_review_updates_status():
    seed_video()
    seed_incident()
    res = client.patch("/api/incidents/inc_test01/review",
                       json={"status": "DAMAGE_CONFIRMED", "note": "Corner crushed."})
    assert res.status_code == 200
    assert res.json()["review_status"] == "DAMAGE_CONFIRMED"
    assert res.json()["reviewer_note"] == "Corner crushed."


def test_review_rejects_unknown_status():
    seed_video()
    seed_incident()
    assert client.patch("/api/incidents/inc_test01/review",
                        json={"status": "TOTALLY_MADE_UP"}).status_code == 422


def test_review_of_unknown_incident_returns_404():
    assert client.patch("/api/incidents/inc_nope/review",
                        json={"status": "FALSE_POSITIVE"}).status_code == 404


# --------------------------------------------------------------- analytics
def test_analytics_are_zero_with_no_data():
    body = client.get("/api/analytics").json()
    assert body["total_incidents"] == 0
    assert body["risk_breakdown"] == {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    assert body["high_risk_events_per_minute"] == 0.0


def test_analytics_aggregate_real_rows():
    seed_video(duration=60.0)
    seed_incident(id="inc_a", risk_level="HIGH")
    seed_incident(id="inc_b", risk_level="CRITICAL")
    body = client.get("/api/analytics").json()
    assert body["total_incidents"] == 2
    assert body["intervention_opportunities"] == 2
    assert body["total_footage_minutes"] == 1.0
    assert body["high_risk_events_per_minute"] == 2.0
    assert body["by_bay"][0]["bay"] == "Dock 05"


def test_prevention_view_reports_recurrence_and_baseline():
    seed_video(duration=60.0)
    for i in range(3):
        seed_incident(id=f"inc_{i}", behaviour_type="product_drag", risk_level="HIGH")
    body = client.get("/api/prevention").json()
    recurring = {r["behaviour_type"]: r for r in body["recurring_behaviours"]}
    assert recurring["product_drag"]["occurrences"] == 3
    assert recurring["product_drag"]["training_topic"]
    assert body["baseline"]["high_risk_events_per_minute"] == 3.0


# ------------------------------------------------------------------ upload
def test_upload_rejects_non_video_extensions():
    res = client.post("/api/videos/upload",
                      files={"file": ("payload.exe", io.BytesIO(b"MZ"), "application/octet-stream")})
    assert res.status_code == 415


def test_upload_rejects_an_empty_file():
    res = client.post("/api/videos/upload",
                      files={"file": ("empty.mp4", io.BytesIO(b""), "video/mp4")})
    assert res.status_code == 400


def test_upload_rejects_an_invalid_floor_condition():
    res = client.post(
        "/api/videos/upload",
        files={"file": ("clip.mp4", io.BytesIO(b"\x00" * 128), "video/mp4")},
        data={"floor_condition": "damp"},
    )
    assert res.status_code == 400


def test_upload_rejects_a_malformed_staging_zone():
    res = client.post(
        "/api/videos/upload",
        files={"file": ("clip.mp4", io.BytesIO(b"\x00" * 128), "video/mp4")},
        data={"staging_zone": "not-json"},
    )
    assert res.status_code == 400


def test_upload_requires_a_file():
    assert client.post("/api/videos/upload").status_code == 422


# --------------------------------------------------------------- assistant
def test_assistant_rejects_a_blank_query():
    assert client.post("/api/assistant/chat", json={"query": "   "}).status_code == 422
    assert client.post("/api/assistant/chat", json={}).status_code == 422


def test_assistant_returns_grounding_metadata():
    seed_video()
    seed_incident()
    body = client.post("/api/assistant/chat",
                       json={"query": "Show me all high-risk handling events"}).json()
    assert body["grounded"] is True
    assert body["intent"]
    assert body["total_incidents_in_db"] == 1
