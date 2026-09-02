import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from backend.app import app

def test_backend_api():
    client = TestClient(app)
    
    # 1. Health check
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "online"
    print("API Health Check: PASS")

    # 2. Videos list
    res = client.get("/api/videos")
    assert res.status_code == 200
    print(f"API Videos List: PASS (Found {len(res.json()['videos'])} videos)")

    # 3. Incidents list
    res = client.get("/api/incidents")
    assert res.status_code == 200
    print(f"API Incidents List: PASS (Found {res.json()['count']} incidents)")

    # 4. Analytics
    res = client.get("/api/analytics")
    assert res.status_code == 200
    analytics = res.json()
    print("API Analytics: PASS ->", analytics["handling_discipline_score"], "/100")

    # 5. Assistant Chat
    res = client.post("/api/assistant/chat", json={"query": "Show me all high risk events"})
    assert res.status_code == 200
    chat_res = res.json()
    assert "response" in chat_res
    print("API Assistant Chat: PASS")

    print("\nALL API TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_backend_api()
