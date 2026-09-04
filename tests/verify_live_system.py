"""
Live end-to-end verification against a running server.

Exercises the ten workflow steps a demo depends on, including the failure paths.
Start the server first:

    python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
    python tests/verify_live_system.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import quote

# Windows consoles default to cp1252, which cannot encode the report's
# punctuation. Force UTF-8 so verification never fails on its own output.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

BASE = os.environ.get("VISIONGUARD_URL", "http://127.0.0.1:8000")

_passed = 0
_failed = 0


def check(label: str, condition: bool, detail: str = "") -> bool:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  [PASS] {label}" + (f" - {detail}" if detail else ""))
    else:
        _failed += 1
        print(f"  [FAIL] {label}" + (f" - {detail}" if detail else ""))
    return condition


def wait_for_server(timeout_sec: int = 120) -> bool:
    """The server loads model weights on start, so give it time to come up."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            urllib.request.urlopen(BASE + "/api/health", timeout=5)
            return True
        except urllib.error.HTTPError:
            return True  # responding, just not with 200
        except Exception:
            time.sleep(2)
    return False


def request(path, method="GET", data=None, headers=None, raw=False):
    # The API returns percent-encoded static URLs, so only encode a path that
    # still carries raw spaces (which urllib would otherwise reject outright).
    if " " in path:
        path = quote(path, safe="/?&=%")
    req = urllib.request.Request(BASE + path, method=method, data=data)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=60) as res:
            body = res.read()
            return res.status, (body if raw else json.loads(body))
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            return exc.code, json.loads(body)
        except ValueError:
            return exc.code, None


def multipart(fields, filename, content):
    """Build a multipart/form-data body without external dependencies."""
    boundary = "----visionguardverify"
    parts = []
    for k, v in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n")
    body = "".join(parts).encode()
    body += (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{filename}\"\r\nContent-Type: video/mp4\r\n\r\n"
    ).encode()
    body += content + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def main() -> int:
    print("=" * 78)
    print(f"  VISIONGUARD LIVE SYSTEM VERIFICATION — {BASE}")
    print("=" * 78)

    # 1 -------------------------------------------------------------- health
    print("\n[1] Server health and detector backend")
    if not wait_for_server():
        print("  [FAIL] server did not respond within 120s - start it and retry")
        return 1
    status, health = request("/api/health")
    if not check("server responds", status == 200, f"status {status}"):
        print("\nServer unreachable. Start it and retry.")
        return 1
    check("reports a detector backend", bool(health.get("detector_backend")),
          health.get("detector_backend"))
    if not health.get("open_vocabulary"):
        print("  [WARN] running on the COCO fallback — products are not detectable")

    # 2 -------------------------------------------------- honest capabilities
    print("\n[2] Capability report is code-derived and honest")
    status, caps = request("/api/capabilities")
    check("capabilities served", status == 200)
    check("12 behaviours defined", caps["counts"]["total"] == 12, str(caps["counts"]))
    statuses = {b["status"] for b in caps["behaviours"]}
    check("declares real limitations, not all-green", statuses != {"IMPLEMENTED"},
          ", ".join(sorted(statuses)))
    check("every behaviour states its limitation",
          all(b["limitations"] for b in caps["behaviours"]))

    # 3 ---------------------------------------------------- analysed footage
    print("\n[3] Analysed videos and detections")
    status, videos = request("/api/videos")
    check("video list served", status == 200)
    vids = videos["videos"]
    check("pilot footage is analysed", len(vids) > 0, f"{len(vids)} video(s)")
    if not vids:
        return 1
    with_events = [v for v in vids if v["incident_count"] > 0]
    check("at least one video produced detections", len(with_events) > 0,
          f"{len(with_events)}/{len(vids)} videos")
    check("annotated overlay video exists",
          any(v.get("annotated_video_url") for v in vids))
    check("scene context recorded", all(v.get("bay") for v in vids))

    target = with_events[0] if with_events else vids[0]

    # 4 ------------------------------------------------------------ tracking
    print("\n[4] Tracking produces persistent object identities")
    status, detail = request(f"/api/videos/{target['id']}")
    check("video detail served", status == 200)
    incidents = detail["incidents"]
    check("incidents attached to the video", len(incidents) > 0, f"{len(incidents)}")
    track_ids = {i["object_track_id"] for i in incidents if i["object_track_id"]}
    check("incidents reference tracked object ids", len(track_ids) > 0, str(sorted(track_ids)))

    # 5 -------------------------------------------- behaviour + temporal proof
    print("\n[5] Behaviour events carry temporal evidence")
    inc = incidents[0]
    check("behaviour type present", bool(inc["behaviour_type"]), inc["behaviour_type"])
    check("temporal stage chain recorded", len(inc.get("evidence_stages", [])) > 0,
          " -> ".join(s["stage"] for s in inc.get("evidence_stages", [])[:4]))
    check("event has a measured duration", inc.get("duration_sec", 0) >= 0)

    # 6 ------------------------------------------------------ risk generation
    print("\n[6] Risk classification is transparent")
    check("risk level assigned", inc["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL"),
          inc["risk_level"])
    factors = inc.get("risk_factors", [])
    check("score breakdown present", len(factors) > 0, f"{len(factors)} factor(s)")
    total = sum(f["points"] for f in factors)
    clamped = max(5.0, min(98.0, total))
    check("score equals the clamped sum of its factors",
          abs(clamped - inc["risk_score"]) < 0.6, f"{inc['risk_score']} vs {clamped:.1f}")
    check("recommendation is actionable", len(inc["recommended_action"]) > 40)

    # 7 --------------------------------------------------- evidence artefacts
    print("\n[7] Evidence is retrievable")
    if inc.get("evidence_image_url"):
        status, body = request(inc["evidence_image_url"], raw=True)
        check("evidence frame downloads", status == 200 and len(body) > 1000,
              f"{len(body)} bytes")
    else:
        check("evidence frame recorded", False, "no evidence_image_url")
    if inc.get("evidence_clip_url"):
        status, body = request(inc["evidence_clip_url"], raw=True)
        check("replay clip downloads", status == 200 and len(body) > 1000,
              f"{len(body)} bytes")
    status, _ = request(target["video_url"], raw=True)
    check("source video streams", status == 200)

    # 8 ------------------------------------------------- responsible AI tiers
    print("\n[8] Responsible-AI evidence tiers")
    check("no incident claims confirmed damage",
          all(i.get("evidence_tier") != "CONFIRMED_DAMAGE" for i in incidents))
    payload = json.dumps({"status": "FALSE_POSITIVE", "note": "verification run"}).encode()
    status, reviewed = request(f"/api/incidents/{inc['id']}/review", "PATCH", payload,
                               {"Content-Type": "application/json"})
    check("human review recorded", status == 200 and reviewed["review_status"] == "FALSE_POSITIVE")
    payload = json.dumps({"status": "PENDING_REVIEW"}).encode()
    request(f"/api/incidents/{inc['id']}/review", "PATCH", payload,
            {"Content-Type": "application/json"})  # restore

    # 9 ------------------------------------------------------- the assistant
    print("\n[9] Assistant answers from the database")
    questions = [
        ("What were today's high-risk events?", None),
        ("What were the three most common risky behaviours?", "top_behaviours"),
        ("Which loading bay had the highest number of risky events?", "location"),
        ("Why was this event classified as high risk?", "why"),
        ("What corrective action is recommended?", "action"),
    ]
    for q, expected_intent in questions:
        payload = json.dumps({"query": q}).encode()
        status, ans = request("/api/assistant/chat", "POST", payload,
                              {"Content-Type": "application/json"})
        ok = status == 200 and len(ans.get("response", "")) > 30
        if expected_intent:
            ok = ok and ans.get("intent") == expected_intent
        check(f'"{q[:46]}"', ok, f"intent={ans.get('intent')} rows={ans.get('relevant_count')}")

    print("\n    Hallucination resistance:")
    payload = json.dumps({"query": "How many forklift collision events were detected in Bay 99?"}).encode()
    status, ans = request("/api/assistant/chat", "POST", payload,
                          {"Content-Type": "application/json"})
    text = ans.get("response", "").lower()
    check("declines to answer without evidence",
          "no " in text or "not have enough" in text or "zero" in text,
          ans.get("response", "")[:90].replace("\n", " "))

    # 10 --------------------------------------------------- validation paths
    print("\n[10] Invalid input and no-data handling")
    status, _ = request("/api/videos/vid_does_not_exist")
    check("unknown video -> 404", status == 404, f"status {status}")
    status, _ = request("/api/incidents/inc_does_not_exist")
    check("unknown incident -> 404", status == 404, f"status {status}")
    status, _ = request("/api/incidents?risk_level=EXTREME")
    check("invalid risk filter -> 400", status == 400, f"status {status}")
    status, _ = request("/api/incidents?limit=99999")
    check("out-of-range limit -> 400", status == 400, f"status {status}")

    body, ctype = multipart({}, "malware.exe", b"MZ\x00\x00")
    status, _ = request("/api/videos/upload", "POST", body, {"Content-Type": ctype})
    check("non-video upload -> 415", status == 415, f"status {status}")

    body, ctype = multipart({}, "empty.mp4", b"")
    status, _ = request("/api/videos/upload", "POST", body, {"Content-Type": ctype})
    check("empty upload -> 400", status == 400, f"status {status}")

    body, ctype = multipart({"floor_condition": "damp"}, "clip.mp4", b"\x00" * 256)
    status, _ = request("/api/videos/upload", "POST", body, {"Content-Type": ctype})
    check("invalid scene context -> 400", status == 400, f"status {status}")

    payload = json.dumps({"query": "   "}).encode()
    status, _ = request("/api/assistant/chat", "POST", payload,
                        {"Content-Type": "application/json"})
    check("blank assistant query -> 422", status == 422, f"status {status}")

    status, _ = request("/api/videos/vid_nope/status")
    check("status of unknown video handled", status == 200)

    # ------------------------------------------------------------- dashboard
    print("\n[11] Dashboard is served")
    status, body = request("/", raw=True)
    check("dashboard HTML served", status == 200 and b"<div id=\"root\"" in body)

    print("\n" + "=" * 78)
    print(f"  RESULT: {_passed} passed, {_failed} failed")
    print("=" * 78)
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
