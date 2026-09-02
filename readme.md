# VisionGuard — AI Video Intelligence for Warehouse Handling
### An AI-Powered Field Intelligence Assistant for Safer, Damage-Free Warehouse Operations

[![Hackathon](https://img.shields.io/badge/Godrej%20Enterprises%20Group-AI%20Warehouse%20Intelligence-blue.svg)](https://forms.cloud.microsoft/r/NLHbJUJ7ru)
[![Python 3.11](https://img.shields.io/badge/python-3.11-brightgreen.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-Vite%20Dashboard-61dafb.svg)](https://vitejs.dev)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-FF5722.svg)](https://ultralytics.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Project Overview & Hackathon Alignment
**VisionGuard** is an AI-powered field intelligence platform designed for warehouse operations. It transforms existing CCTV feeds or mobile video into an intelligent operational assistant that understands material-handling behaviour, detects damage-causing actions, calculates transparent multi-factor risks, and empowers shift supervisors to intervene **before** product damage occurs.

* **Hackathon**: Godrej Enterprises Group (GEG) — AI Video Intelligence for Warehouse Handling
* **Core Paradigm Shift**:
  $$\text{Traditional CCTV: Camera} \rightarrow \text{Recording} \rightarrow \text{Human Review} \rightarrow \text{Incident Discovered} \rightarrow \text{Corrective Action}$$
  $$\text{VisionGuard: Camera} \rightarrow \text{AI Perception} \rightarrow \text{Behaviour Understanding} \rightarrow \text{Risk Detection} \rightarrow \text{Alert} \rightarrow \text{Intervention} \rightarrow \text{Prevention}$$

---

## 2. Problem Statement
Warehouses handle thousands of products daily across loading bays, vehicle transfers, and staging zones. Significant product damage occurs not from equipment failure, but from **inappropriate handling behaviour**: dropping, dragging, tumbling, unstable stacking, stepping on cartons, pulling by straps, and traversing uneven dock gaps. 

Traditional CCTV surveillance only records damage after the fact. VisionGuard moves operations from retrospective post-mortems to **real-time proactive damage prevention**.

---

## 3. Key Capabilities & Features
1. **AI Video Understanding**: Ingests recorded CCTV or smartphone footage (1280x720 HD @ 30 FPS) and detects operators, cartons, KD furniture, cupboards, mattresses, and vehicles.
2. **Persistent Multi-Object Tracking**: Utilizes ByteTrack spatial association to assign persistent Track IDs and calculate velocity vectors $(v_x, v_y)$ and acceleration $(a_x, a_y)$.
3. **10 Modular Behaviour Detectors**: Temporal sequence reasoning over multi-frame kinematic trajectories.
4. **Transparent Multi-Factor Risk Engine**: Classifies events into `LOW`, `MEDIUM`, `HIGH`, or `CRITICAL` based on physical parameters (drop height, velocity, deceleration, floor moisture, stacking order).
5. **Automated Incident Evidence Generation**: Extracts highlighted bounding-box snapshots with operational root causes and supervisor corrective interventions.
6. **Grounded AI Warehouse Supervisor Assistant**: Conversational assistant strictly grounded in SQLite database incidents (zero hallucinations).
7. **Industrial Dark-Mode Dashboard**: Synchronized video player, interactive timeline, evidence inspector drawer, Pareto analytics, and video ingestion dropzone.

---

## 4. End-to-End System Architecture

```
                                  Warehouse Video (CCTV / Mobile)
                                                │
                                                ▼
                                    [Video Processing Pipeline]
                                                │
                                                ▼
                                    [YOLOv8 Object Detection]
                                                │
                                                ▼
                              [Persistent Multi-Object Tracking (ByteTrack)]
                                                │
                                                ▼
                              [Temporal Kinematic State Buffer (vx, vy, ax, ay)]
                                                │
                     ┌──────────────────────────┴──────────────────────────┐
                     ▼                                                     ▼
        [10 Behaviour Detectors]                              [Multi-Factor Risk Engine]
   - Drop (Impact & Deceleration)                        - Evaluates physical severity
   - Dragging (Floor Plane Translation)                  - Computes 0-100 numeric score
   - Throwing / Pushing (Release Velocity)               - Classifies LOW/MED/HIGH/CRITICAL
   - Rolling / Tumbling (Aspect Ratio)                   - Prescribes supervisor action
   - Improper Stacking (Inversion)                                         │
   - Stepping on Cartons (Crush Hazard)                                    ▼
   - Strap Pulling (Tensile Stress)                          [Incident Generation]
   - Wet Floor Dragging (Moisture Hazard)                - Evidence snapshot extraction
   - Orientation (Upright Stored Flat)                   - SQLite database persistence
   - Dock Level (Threshold Gap Shock)                                      │
                     └──────────────────────────┬──────────────────────────┘
                                                ▼
                                       [FastAPI Backend]
                                                │
                     ┌──────────────────────────┴──────────────────────────┐
                     ▼                                                     ▼
     [React + Vite Industrial Dashboard]                  [Grounded AI Supervisor Assistant]
  - Synchronized video & track overlay                  - Zero-hallucination factual chat
  - Interactive clickable incident timeline             - Shift summaries & Pareto reports
  - Evidence frame inspector drawer                     - Root-cause explanation
```

---

## 5. Behaviour Taxonomy & GEG Scenario Coverage

| Scenario # | Behaviour / Bad Practice | Detection Principle & Kinematics | Risk Level | Prescriptive Corrective Intervention |
|:---:|---|---|:---:|---|
| **B1** | **Product Dropping** | Rapid downward velocity ($v_y > 120\text{ px/s}$), floor impact deceleration, stationary resting state | **HIGH / CRITICAL** | Halt handling. Inspect product internally and externally. Review manual lowering technique with worker. |
| **B2** | **Product Dragging** | Sustained horizontal displacement along floor plane ($v_x > 35\text{ px/s}$) with adjacent operator | **MEDIUM / HIGH** | Mandate hydraulic trolley or pallet truck. Stop dragging cartons to prevent bottom abrasion. |
| **B3** | **Throwing / Pushing** | High release velocity ($> 140\text{ px/s}$), spatial detachment from operator hands, ballistic trajectory | **HIGH / CRITICAL** | Intervene immediately. Stop throwing goods. Enforce sequential one-by-one controlled placement. |
| **B4** | **Rolling / Tumbling** | Cyclical aspect ratio inversion ($w/h > 1.15 \leftrightarrow w/h < 0.85$) while translating along floor | **MEDIUM / HIGH** | Deploy hand truck or team lift. Tumbling crushes corners and weakens structural rigidity. |
| **B5** | **Improper Stacking** | Heavy or larger item ($w_{\text{top}} > 1.25 \times w_{\text{bottom}}$) placed atop smaller/lighter packets | **HIGH / CRITICAL** | Restack staging area immediately. Ensure full perimeter support for lighter upper cartons. |
| **B6** | **Stepping on Cartons** | Operator feet/lower body intersecting carton top half ($x_{\text{overlap}} > 30\text{px}$) | **CRITICAL** | Severe safety violation. Strictly prohibit standing on cartons. Clear designated walkways. |
| **B7** | **Using Straps to Pull** | Grip points concentrated strictly on carton upper strapping band without bottom lifting support | **HIGH** | Use carton base lifting handholds. Strapping bands are for closure only and tear box walls. |
| **B8** | **Dragging on Wet Floor** | Material movement across moisture hazard or dock zones with high specular ground reflection | **HIGH / CRITICAL** | Stop movement. Dry dock surface and inspect for water absorption through carton base. |
| **B9** | **Vertical Product Flat** | Upright unit (cupboard, appliance) positioned horizontally ($w/h > 1.25$) violating "This Side Up" | **MEDIUM / HIGH** | Restore vertical stance immediately to prevent hinge strain, component sagging, or panel deflection. |
| **B10**| **Dock Level Hazard** | Transition across vehicle-dock threshold without mechanical dock leveller bridge plate | **HIGH / CRITICAL** | Deploy dock leveller plate before continuing transfer across vehicle-dock transition gap. |

---

## 6. Official Pilot Videos Analyzed
The pipeline has been tested and validated on all 7 official Godrej warehouse pilot videos:
1. `Dock level, dragging cupboard.mp4` (930 frames, 31.0s, 720p HD @ 30 FPS)
2. `KD packets dragged, heavy box kept on other packets.mp4` (1012 frames, 33.7s, 720p HD @ 30 FPS)
3. `Rolling and dragging on wet floor.mp4` (183 frames, 6.1s, 720p HD @ 30 FPS)
4. `Rolling and dropping carton.mp4` (282 frames, 9.4s, 720p HD @ 30 FPS)
5. `Stepping on cartons, vertical product kept horizontally, heavy product kept on top.mp4` (1471 frames, 49.0s, 720p HD @ 30 FPS)
6. `Throwing Mattresses.mp4` (1249 frames, 41.6s, 720p HD @ 30 FPS)
7. `Throwing seating cartons, using strap to hold.mp4` (451 frames, 15.0s, 720p HD @ 30 FPS)

---

## 7. Technology Stack
* **Computer Vision**: Python 3.11, OpenCV 4.11, Ultralytics YOLOv8, ByteTrack Multi-Object Tracking
* **Backend API**: FastAPI 0.110+, Uvicorn 0.28+, Pydantic v2
* **Storage**: SQLite (with schema easily migratable to PostgreSQL)
* **Frontend**: React 18, Vite 5, Vanilla CSS Design System, Lucide React Icons
* **AI Assistant**: Retrieval-Augmented Generation (RAG) over structured SQLite incidents

---

## 8. Installation & Quick Start

### Prerequisites
* Python 3.10+ (Tested on Python 3.11)
* Node.js v18+ & npm
* Git

### Step 1: Clone Repository
```bash
git clone https://github.com/prasadproject2025-alt/AI-Warehouse-Intelligence.git
cd AI-Warehouse-Intelligence
```

### Step 2: Install Backend Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Install & Build Dashboard Frontend
```bash
cd dashboard
npm install
npm run build
cd ..
```

### Step 4: Launch the Full-Stack Application
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```
Open your web browser at:
```
http://localhost:8000
```

*(Optional: For hot-reloading frontend development, run `npm run dev` inside `dashboard/` and view at `http://localhost:5173`).*

---

## 9. Running Tests
Run the comprehensive test suite verifying the detector, tracker, 10 behaviour engines, risk scoring, and backend API:

```bash
# 1. Test Persistent Multi-Object Tracking
python tests/test_tracking.py

# 2. Test Behaviour Detectors & Risk Engine
python tests/test_behaviour_engine.py

# 3. Test Grounded AI Warehouse Assistant
python tests/test_assistant.py

# 4. Test REST API Endpoints
python tests/test_api.py

# 5. Test End-to-End Pipeline Integration
python tests/test_pipeline.py
```

---

## 10. Responsible AI Principles
In compliance with the GEG hackathon guidelines:
* **Focus on Process Improvement**: VisionGuard identifies handling discipline issues to improve training and operational equipment rather than facilitating punitive employee surveillance.
* **Three-Tier Safety Distinction**:
  $$\text{Observed Behaviour} \longrightarrow \text{Potential Risk} \longrightarrow \text{Confirmed Damage}$$
  The system never claims that an item was damaged without physical evidence.
* **Explainable & Actionable Alerts**: Every alert contains human-readable evidence, physical parameters, and supervisor coaching guidance.

---

## 11. Future Roadmap
* **Edge TPU / Jetson Deployment**: Optimize YOLOv8 with TensorRT for zero-latency on-device processing.
* **Multi-Camera Bay Fusion**: Re-identify operators and pallet batches across overlapping dock cameras.
* **ERP / WMS Integration**: Webhook notifications directly to SAP / Manhattan Associates Warehouse Management Systems.

---

## 12. Team & Acknowledgements
Built for the **Godrej Enterprises Group (GEG) AI Video Intelligence for Warehouse Handling Hackathon**.
* **Repository**: [AI-Warehouse-Intelligence](https://github.com/prasadproject2025-alt/AI-Warehouse-Intelligence)
* **Team Contact**: `pudeshi@godrej.com`
