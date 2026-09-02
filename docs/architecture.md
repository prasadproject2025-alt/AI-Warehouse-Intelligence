# VisionGuard — System Architecture

## Godrej Enterprises Group (GEG) Hackathon
**Challenge**: AI Video Intelligence for Warehouse Handling  
**Paradigm Shift**: From *Retrospective Surveillance* to *Proactive Operational Assistance*

---

## 1. High-Level Architecture Diagram

```
+-----------------------------------------------------------------------------------+
|                            WAREHOUSE PHYSICAL ENVIRONMENT                         |
|      (Loading Bay, Staging Area, Vehicle Bed, Forklifts, Pallets, Operators)      |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼ (CCTV / Smartphone HD Video)
+-----------------------------------------------------------------------------------+
|                           COMPUTER VISION & TRACKING TIER                         |
|  - Video Ingestion & Frame Decoupling (OpenCV @ 30 FPS, 720p HD)                  |
|  - YOLOv8 Object Detection (Operators, Cartons, KD Packets, Cupboards, Mattresses)|
|  - Persistent Multi-Object Tracking (ByteTrack / Kalman Velocity Vectoring)       |
|  - Trajectory & Kinematic State Buffer (vx, vy, ax, ay, ground elevation)         |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼ (Temporal Object Trajectories)
+-----------------------------------------------------------------------------------+
|                        BEHAVIOUR INTELLIGENCE ENGINE (10 MODULAR DETECTORS)       |
|  1. Drop Detector (Peak vertical velocity, floor impact deceleration, stationary) |
|  2. Drag Detector (Sustained floor translation without vertical lifting)          |
|  3. Throw/Push Detector (Release velocity, detachment from operator, ballistic)   |
|  4. Roll/Tumble Detector (Oscillating aspect ratio w/h cycles along ground)        |
|  5. Stacking Inversion Detector (Heavy/larger item placed atop smaller cartons)    |
|  6. Stepping Detector (Operator foot intersection on package top surface)         |
|  7. Strap Pulling Detector (Tensile stress on straps without base support)        |
|  8. Wet Floor Detector (Material dragging/rolling through dock moisture hazard)   |
|  9. Orientation Violation Detector (Vertical goods like cupboards staged flat)    |
| 10. Dock Level Detector (Threshold transition across vehicle gap without bridge)  |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼ (Validated Behaviour Triggers)
+-----------------------------------------------------------------------------------+
|                           TRANSPARENT MULTI-FACTOR RISK ENGINE                    |
|  - Multi-factor Scoring: Drop Height, Velocity, Impact, Object Type, Fragility    |
|  - Classification: LOW | MEDIUM | HIGH | CRITICAL                                 |
|  - Distinction: Observed Behaviour -> Potential Risk -> Confirmed Damage          |
|  - Prescriptive Operational Recommendations (Actionable Supervisor Interventions) |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼ (Incident Records & Evidence Frames)
+-----------------------------------------------------------------------------------+
|                         PERSISTENCE & SERVICE BACKEND TIER                        |
|  - SQLite Storage (videos, incidents, trajectories, shift analytics)              |
|  - Automated Evidence Frame Generation (Annotated bounding box, HUD telemetry)    |
|  - FastAPI REST Endpoints (/api/videos, /api/incidents, /api/analytics)           |
|  - Grounded AI Warehouse Assistant (Zero-hallucination RAG over incident DB)      |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼ (WebSocket / REST JSON & Static Assets)
+-----------------------------------------------------------------------------------+
|                         INDUSTRIAL OPERATOR DASHBOARD UI                          |
|  - Synchronized Video Player with Live HUD Telemetry & Bounding Box Overlay       |
|  - Interactive Incident Timeline with Timestamp Jump-to-Replay                    |
|  - Evidence Frame Inspector with Zoom, Root Cause, and Prescriptive Action        |
|  - Shift Behaviour Pareto Charts & Discipline Compliance Gauge                    |
|  - Conversational AI Assistant Drawer with Suggested Operational Queries          |
|  - Video Upload & Ingestion Dropzone                                              |
+-----------------------------------------------------------------------------------+
```

---

## 2. Component Breakdown

### A. Computer Vision & Persistent Tracking (`detection/`)
- **YOLOv8 Detection Engine**: Identifies operators, parcels, cartons, large appliances, furniture, and vehicles.
- **Persistent Tracker**: Implements Kalman-filter-based tracking with IoU and spatial proximity association to preserve track identity across frames, calculating smoothed velocity $(v_x, v_y)$ and acceleration $(a_x, a_y)$.

### B. Behaviour Reasoning Engine (`behaviour/`)
- Unlike naive single-frame classifiers, VisionGuard maintains a sliding temporal window (15–45 frames) per track.
- Detects the entire chain of activity:
  $$\text{Approach} \rightarrow \text{Interaction} \rightarrow \text{Improper Kinetic Transition} \rightarrow \text{Impact / Resting State}$$

### C. Multi-Factor Risk Engine (`risk/`)
- Quantifies operational damage risks without arbitrary numbers:
  - Drop height in pixels relative to operator height (e.g. $> 1\text{m}$ drop)
  - Impact velocity and abrupt cessation of movement
  - Presence of water or wet dock surface
  - Geometric stacking order (width and mass ratio)

### D. Incident System & Grounded Assistant (`assistant/`)
- All conversational answers generated by the AI assistant are directly queried from the SQLite database. If a behavior was not observed in the video, the assistant explicitly states that no incidents were found, ensuring zero hallucination.
