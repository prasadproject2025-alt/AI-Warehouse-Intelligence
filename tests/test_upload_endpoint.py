import requests
import json

def test_video_upload():
    url = "http://127.0.0.1:8000/api/videos/upload"
    video_file = "data/raw/Rolling and dragging on wet floor.mp4"
    
    print(f"Testing upload of: {video_file}")
    with open(video_file, "rb") as f:
        files = {"file": ("test_upload_wet_floor.mp4", f, "video/mp4")}
        res = requests.post(url, files=files)
        
    print(f"Status Code: {res.status_code}")
    assert res.status_code == 200, f"Failed with {res.status_code}: {res.text}"
    
    data = res.json()
    print("Response JSON:")
    print(f" - Message: {data.get('message')}")
    print(f" - Video ID: {data['result']['video_id']}")
    print(f" - Incidents Count: {data['result']['incidents_count']}")
    print("UPLOAD TEST PASSED 100%!")

if __name__ == "__main__":
    test_video_upload()
