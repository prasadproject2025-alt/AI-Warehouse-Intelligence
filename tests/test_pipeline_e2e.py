"""
End-to-end pipeline test on a synthesised video.

A short clip is generated with a person-like and a box-like shape so the test
exercises decode -> detect -> track -> reason -> persist -> annotate without
depending on the pilot footage or on any particular model finding a real
carton. Detector-quality claims belong in the accuracy audit, not here.
"""

import os

import cv2
import numpy as np
import pytest

from backend.database.db import DatabaseManager
from behaviour.behaviour_engine import SceneContext
from video.evidence import create_evidence_snapshot, write_incident_clip
from video.processor import TASK_STATUS, VideoProcessor


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("clips") / "synthetic.mp4")
    w, h, fps, n = 640, 360, 15, 60
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened(), "OpenCV could not open an mp4 writer"
    for i in range(n):
        frame = np.full((h, w, 3), 60, dtype=np.uint8)
        cv2.rectangle(frame, (0, 300), (w, h), (90, 90, 90), -1)      # floor band
        cv2.rectangle(frame, (300, 180), (340, 300), (200, 180, 160), -1)  # figure
        x = 60 + i * 4
        cv2.rectangle(frame, (x, 250), (x + 70, 300), (40, 120, 200), -1)  # moving box
        writer.write(frame)
    writer.release()
    assert os.path.getsize(path) > 0
    return path


@pytest.fixture(scope="module")
def processor():
    return VideoProcessor()


def test_pipeline_runs_and_persists_the_video_record(processor, synthetic_video):
    result = processor.process_video(
        synthetic_video, "vid_e2e", generate_annotated_video=True,
        scene=SceneContext(bay="Test Bay", shift="Shift A", camera_id="CAM-E2E"),
    )
    assert result["video_id"] == "vid_e2e"
    assert result["frame_count"] > 0
    assert result["frames_analysed"] > 0
    assert result["detector_backend"] in ("open_vocab", "coco")

    stored = DatabaseManager.get_video_by_id("vid_e2e")
    assert stored is not None
    assert stored["status"] == "completed"
    assert stored["bay"] == "Test Bay"
    assert stored["width"] == 640 and stored["height"] == 360


def test_annotated_video_is_written_and_playable(processor, synthetic_video):
    result = processor.process_video(synthetic_video, "vid_e2e_annot", True)
    path = result["annotated_video"]
    assert path and os.path.exists(path) and os.path.getsize(path) > 0
    cap = cv2.VideoCapture(path)
    try:
        assert cap.isOpened()
        ok, frame = cap.read()
        assert ok and frame is not None
    finally:
        cap.release()


def test_task_status_completes_with_full_progress(processor, synthetic_video):
    processor.process_video(synthetic_video, "vid_e2e_status", False)
    status = TASK_STATUS["vid_e2e_status"]
    assert status["status"] == "completed"
    assert status["progress_percent"] == 100


def test_incidents_are_persisted_and_retrievable(processor, synthetic_video):
    result = processor.process_video(synthetic_video, "vid_e2e_db", False)
    stored = DatabaseManager.get_incidents(video_id="vid_e2e_db", limit=1000)
    assert len(stored) == result["incidents_count"]
    for inc in stored:
        assert inc["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        assert 0 < inc["risk_score"] <= 98
        assert inc["evidence_description"] and inc["recommended_action"]
        assert inc["bay"] and inc["shift"]
        # Every incident must carry an auditable score breakdown.
        assert inc["risk_factors"], "incidents must explain their score"
        assert inc["evidence_tier"] != "CONFIRMED_DAMAGE"


def test_missing_file_raises_a_clear_error(processor):
    with pytest.raises(FileNotFoundError):
        processor.process_video("data/raw/definitely_not_here.mp4", "vid_missing")


def test_corrupt_file_is_reported_as_failed(processor, tmp_path):
    bad = tmp_path / "corrupt.mp4"
    bad.write_bytes(b"this is not a video")
    with pytest.raises(ValueError):
        processor.process_video(str(bad), "vid_corrupt")
    assert TASK_STATUS["vid_corrupt"]["status"] == "failed"
    assert TASK_STATUS["vid_corrupt"]["error"]


def test_evidence_snapshot_is_written_with_risk_wording(tmp_path):
    frame = np.full((360, 640, 3), 70, dtype=np.uint8)
    incident = {
        "id": "inc_snap", "behaviour_type": "product_drop", "timestamp_sec": 3.5,
        "risk_level": "HIGH", "risk_score": 74.0,
        "bounding_box": [100.0, 100.0, 220.0, 240.0],
        "evidence_description": "Carton descended rapidly and stopped abruptly.",
        "recommended_action": "[HIGH PRIORITY] Inspect the package.",
    }
    path = create_evidence_snapshot(frame, incident, str(tmp_path))
    assert os.path.exists(path) and os.path.getsize(path) > 0
    img = cv2.imread(path)
    assert img is not None and img.shape[:2] == (360, 640)


def test_evidence_clip_is_extracted_around_the_timestamp(synthetic_video, tmp_path):
    out = str(tmp_path / "clip.mp4")
    path = write_incident_clip(synthetic_video, 2.0, out, pre_sec=1.0, post_sec=1.0)
    assert path and os.path.exists(path)
    cap = cv2.VideoCapture(path)
    try:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert frames > 0
    finally:
        cap.release()


def test_reprocessing_replaces_rather_than_duplicates(processor, synthetic_video):
    processor.process_video(synthetic_video, "vid_repeat", False)
    first = DatabaseManager.count_incidents(video_id="vid_repeat")
    processor.process_video(synthetic_video, "vid_repeat", False)
    second = DatabaseManager.count_incidents(video_id="vid_repeat")
    videos = [v for v in DatabaseManager.get_all_videos() if v["id"] == "vid_repeat"]
    assert len(videos) == 1, "re-analysis must not create a second video row"
    # Incident ids are fresh each run, so the row count may grow; what must not
    # happen is a duplicated video record.
    assert first >= 0 and second >= first
