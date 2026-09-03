import requests
import time
import json

def test_async_upload():
    # 1. Health check
    h = requests.get("http://127.0.0.1:8000/api/health")
    assert h.status_code == 200
    print("Server Health: ONLINE")

    # 2. Upload video
    video_path = "data/raw/Rolling and dragging on wet floor.mp4"
    with open(video_path, "rb") as f:
        res = requests.post(
            "http://127.0.0.1:8000/api/videos/upload",
            files={"file": ("test_async_wet_floor.mp4", f, "video/mp4")}
        )
    
    print(f"Upload HTTP Status: {res.status_code}")
    assert res.status_code == 200
    upload_data = res.json()
    print("Upload Response:", upload_data)
    vid_id = upload_data["video_id"]

    # 3. Poll status
    print(f"Polling status for video ID: {vid_id}...")
    for _ in range(30):
        time.sleep(2)
        stat_res = requests.get(f"http://127.0.0.1:8000/api/videos/{vid_id}/status")
        stat_data = stat_res.json()
        print(f" - Progress: {stat_data.get('progress_percent')}% | Status: {stat_data.get('status')} | Incidents: {stat_data.get('incidents_count')}")
        if stat_data.get("status") == "completed":
            print("Video Analysis Completed Successfully!")
            break
    
    assert stat_data.get("status") == "completed"
    print("\nALL ASYNC UPLOAD & BACKGROUND PROCESSING TESTS PASSED 100%!")

if __name__ == "__main__":
    test_async_upload()
