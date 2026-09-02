# VisionGuard — Hackathon Demonstration Guide

## Godrej Enterprises Group (GEG) Hackathon Demo Workflow

This step-by-step walkthrough shows how VisionGuard fulfills every requirement of the challenge.

---

### Step 1: Launch the System
In a terminal, run:
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```
Open your browser to:
```
http://localhost:8000
```
*(Or run `npm run dev` in `dashboard/` for hot-reload frontend development).*

---

### Step 2: Operational Dashboard Overview
1. **Notice the Top KPI Bar**:
   - Total Critical & High-Risk events detected across shifts.
   - **Warehouse Handling Discipline Index** (e.g. 78%): Real-time quality index based on incident density.
   - **Damage Prevention Interventions**: Number of times supervisor intervention prevented product loss.
2. **AI Perception HUD**:
   - Shows active YOLOv8 detection and persistent ByteTrack ID tracking at 30 FPS.

---

### Step 3: Interactive Video Replay & Evidence Inspection
1. Select any of the official Godrej warehouse pilot videos using the horizontal switcher bar:
   - `Rolling and dropping carton.mp4`
   - `Throwing Mattresses.mp4`
   - `Dock level, dragging cupboard.mp4`
   - `KD packets dragged, heavy box kept on other packets.mp4`
   - `Stepping on cartons, vertical product kept horizontally, heavy product kept on top.mp4`
   - `Rolling and dragging on wet floor.mp4`
   - `Throwing seating cartons, using strap to hold.mp4`
2. Look at the **Detected Behaviour Timeline**:
   - Filter by `CRITICAL`, `HIGH`, `MEDIUM`, or `LOW`.
3. **Click on any incident card**:
   - The video player immediately seeks to the exact timestamp of the event.
   - The **Incident Evidence Inspector** updates on the right, displaying:
     - High-resolution annotated evidence snapshot with bounding box and telemetry.
     - Observed Physical Metrics (e.g. drop velocity, drag distance).
     - Operational Root Cause.
     - **Prescriptive Actionable Intervention** (e.g., mandate pallet truck, restack base).

---

### Step 4: Shift Behaviour Analytics
1. Click the **Shift Behaviour Analytics** tab.
2. Observe the **Pareto Breakdown**:
   - Identifies which behaviours contribute most heavily to risk (e.g., Improper Stacking, Dragging, Dropping).
3. View the **10 Target Behaviour Taxonomy Coverage Matrix**:
   - Validates that all 10 scenarios defined in the GEG challenge are actively monitored.

---

### Step 5: Conversational AI Warehouse Assistant
1. Click the **AI Warehouse Assistant** tab.
2. Try asking questions or click the quick prompt pills:
   - *"Show me all high-risk handling events"*
   - *"What were the three most common risky behaviours?"*
   - *"How many product drops were detected?"*
   - *"Why was this event classified as high risk?"*
3. **Verify Zero Hallucination**:
   - The assistant references only actual incidents recorded in the SQLite database.
   - Every answer includes timestamp, object ID, risk score, and corrective guidance.

---

### Step 6: Ingest New Warehouse Video
1. Click the **Ingest Warehouse Video** tab.
2. Drag and drop any recorded CCTV or smartphone MP4 video.
3. The AI pipeline processes the video in seconds, populating the database, updating shift analytics, and generating evidence snapshots automatically.
