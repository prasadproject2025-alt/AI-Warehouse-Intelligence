import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.database.db import init_db, DatabaseManager, get_connection
from video.processor import VideoProcessor

def test_pipeline():
    init_db()
    with get_connection() as conn:
        conn.execute("DELETE FROM incidents WHERE video_id = 'vid_test_roll_drop'")
        conn.execute("DELETE FROM videos WHERE id = 'vid_test_roll_drop'")
        conn.commit()

    processor = VideoProcessor(conf_threshold=0.25)
    
    video_path = "data/raw/Rolling and dropping carton.mp4"
    assert os.path.exists(video_path), f"{video_path} not found"
    
    print(f"Testing end-to-end pipeline on: {video_path}")
    result = processor.process_video(video_path, video_id="vid_test_roll_drop", generate_annotated_video=True)
    
    print("Pipeline Execution Completed!")
    print(f"Video ID: {result['video_id']}")
    print(f"Total Incidents: {result['incidents_count']}")
    for inc in result["incidents"]:
        print(f"  - [{inc['risk_level']}] {inc['behaviour_type']} @ {inc['timestamp_sec']}s (Risk Score: {inc['risk_score']})")
        print(f"    Evidence Image: {inc.get('evidence_image_path')}")
        assert os.path.exists(inc['evidence_image_path']), "Evidence image was not generated!"

    # Verify DB persistence
    db_incidents = DatabaseManager.get_incidents(video_id="vid_test_roll_drop")
    print(f"Persisted in SQLite: {len(db_incidents)} incidents")
    assert len(db_incidents) == result['incidents_count']
    assert os.path.exists(result['annotated_video']), "Annotated video not found"
    print("ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_pipeline()
