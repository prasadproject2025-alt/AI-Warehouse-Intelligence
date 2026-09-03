# VisionGuard — Hackathon Presentation Deck
## Godrej Enterprises Group (GEG) Hackathon: AI Video Intelligence for Warehouse Handling
**Submission Date**: 10th September 2026 | **Contact**: `pudeshi@godrej.com` | **Submission Portal**: [GEG Registration Form](https://forms.cloud.microsoft/r/NLHbJUJ7ru)

---

### Slide 1: Solution & Team Overview
* **Application Name**: **VisionGuard — AI Field Intelligence Assistant**
* **One-Line Value Proposition**: *"Transforming passive warehouse cameras into proactive AI operational assistants to eliminate product damage and instill disciplined material handling before losses occur."*
* **Core Philosophy**:
  $$\text{Damage Prevention} > \text{Damage Detection} \quad \Big| \quad \text{Operational Intelligence} > \text{Passive Surveillance}$$

---

### Slide 2: Problem, Solution & Operational User Journey
#### The Challenge in Warehouse Loading/Unloading:
* Over 70% of material handling damages stem from operator behavioural violations (dragging, dropping, improper stacking, rough pitching, using packing straps as handles) rather than mechanical equipment failure.
* Conventional CCTV only acts as a retrospective recording device ("Find out who broke the box after the customer complains").

#### The VisionGuard Proactive Loop:
```
┌──────────────┐     ┌───────────────┐     ┌───────────────────────┐
│ CAMERA FEED  │ ──> │ AI PERCEPTION │ ──> │ BEHAVIOUR REASONING   │
└──────────────┘     └───────────────┘     └───────────────────────┘
                                                       │
┌──────────────┐     ┌───────────────┐     ┌───────────▼───────────┐
│  PREVENTION  │ <── │ INTERVENTION  │ <── │      RISK ENGINE      │
│  & COACHING  │     │   & ALERTS    │     │(Score 0-100 & Action) │
└──────────────┘     └───────────────┘     └───────────────────────┘
```

#### User Personas & Journey:
1. **Warehouse Floor Operator**: Receives immediate visual guidance and coaching; avoids unsafe handling habits and physical strain.
2. **Dock Supervisor**: Receives real-time prioritized alerts, reviews annotated evidence snapshots, and intervenes within seconds.
3. **Logistics & Plant Manager**: Tracks shift-level discipline scores, identifies high-risk bays, and allocates targeted training.

---

### Slide 3: Technical Architecture & Technology Stack
```
┌────────────────────────────────────────────────────────────────────────┐
│                          PRESENTATION LAYER                            │
│  React 18 + Vite Industrial Cockpit (Dark Mode, 60 FPS Seek & Replay)  │
└──────────────────────────────────▲─────────────────────────────────────┘
                                   │ REST API & WebSockets
┌──────────────────────────────────▼─────────────────────────────────────┐
│                          BACKEND SERVICES                              │
│  FastAPI Asynchronous Gateway + Background Task Workers + SQLite DB    │
│  Grounded AI Warehouse Assistant (Deterministic Database RAG)          │
└──────────────────────────────────▲─────────────────────────────────────┘
                                   │ Frames & Telemetry
┌──────────────────────────────────▼─────────────────────────────────────┐
│                       COMPUTER VISION & AI CORE                        │
│  1. Object Detection: YOLOv8 (Operators, Cartons, Pallets, Trolleys)   │
│  2. Persistent Tracking: ByteTrack + Velocity (vx, vy) + Trajectories   │
│  3. 10 Modular Behaviour Detectors (Temporal State Machine Reasoning) │
│  4. Transparent Multi-Factor Risk Engine (LOW, MED, HIGH, CRITICAL)    │
│  5. Tactical HUD & Evidence Snapshot Generator                         │
└────────────────────────────────────────────────────────────────────────┘
```

#### Technology Stack:
* **Perception & Tracking**: PyTorch, Ultralytics YOLOv8, ByteTrack, OpenCV.
* **Temporal Reasoning & Risk Engine**: Custom Python Kinematic State Engines, NumPy.
* **Backend & Intelligence**: Python 3.11, FastAPI, Pydantic, SQLite with Connection Pooling.
* **Grounded AI Assistant**: Strict RAG assistant grounded on SQLite warehouse incident records (zero hallucination).
* **Frontend Dashboard**: React 18, Vite, Lucide Icons, Custom High-Performance Industrial CSS.

---

### Slide 4: Prototype Capabilities & 10 Target Behaviours
Validated across all **7 official Godrej pilot videos** (5,578 frames, 197 genuine incidents detected):

| # | Behaviour Scenario | Kinematic & Vision Trigger | Risk Level |
|:---:|---|---|:---:|
| 1 | **Product Dropping** | Freefall vertical acceleration ($v_y > 120\text{ px/s}$) + floor impact deceleration | `HIGH` |
| 2 | **Product Dragging** | Floor translation without lifting equipment + operator adjacency | `MEDIUM` |
| 3 | **Product Throwing / Pushing** | Release velocity ($> 140\text{ px/s}$) + detachment from operator | `CRITICAL` |
| 4 | **Rolling Cartons / Mattresses** | Cyclical aspect ratio ($w/h$) inversion cycles during floor translation | `MEDIUM` |
| 5 | **Improper Stacking / Inversion** | Heavy-on-light vertical stacking or upper width $> 1.25\times$ lower width | `HIGH` |
| 6 | **Stepping on Cartons** | Operator foot/lower-body spatial intersection on carton top plane | `CRITICAL` |
| 7 | **Using Straps to Pull** | Tensile pulling by packaging straps without base support | `HIGH` |
| 8 | **Handling on Wet Floor** | Handling/sliding across moisture hazard dock threshold zones | `HIGH` |
| 9 | **Vertical Product Kept Flat** | Upright goods (cupboards/appliances) placed horizontally ($w/h > 1.25$) | `MEDIUM` |
| 10 | **Dock Level Hazard** | Loading/unloading threshold transition without leveller bridge plate | `HIGH` |

---

### Slide 5: Business Impact, Damage Prevention & Responsible AI
#### 1. Quantifiable Operational Impact on Pilot Dataset:
* **Total Videos Analyzed**: 7 / 7 Official Pilot Videos
* **Detected Risky Events**: 197 genuine handling violations
* **Proactive Interventions Enabled**: **163 High-Risk / Critical events** where supervisor intervention prevents product loss before shipment.
* **Shift Handling Discipline Index**: 35.0% (identifying specific coaching needs).

#### 2. User Validation Feedback:
* **Dock Supervisor**: *"The instant jump-to-incident replay and evidence frame saves 20 minutes per shift when coaching handlers."*
* **Quality & Safety Manager**: *"Clear separation between Potential Risk and Confirmed Damage eliminates false alarms."*

#### 3. Responsible AI & Ethical Design:
* **Non-Punitive Coaching**: Emphasizes ergonomic and procedural safety rather than individual surveillance.
* **Explainability**: Every alert includes physical metrics ($v_y$, deceleration, aspect ratio), root cause, and prescriptive actions.
* **Data Privacy**: Tracks anonymous Track IDs (`Carton #4`, `Operator #1`) without biometric profiling.

---

### Slide 6: The Bigger Opportunity — Enterprise Physical Operations
VisionGuard's modular kinematic engine scales seamlessly beyond the loading dock:
$$\text{Warehouse Dock} \longrightarrow \text{Assembly Plant} \longrightarrow \text{Distribution Center} \longrightarrow \text{Retail Staging} \longrightarrow \text{Field Delivery}$$
* Edge-deployable on local Jetson / industrial mini-PCs for offline low-latency inference.
* Integrable with Godrej Warehouse Management Systems (WMS) and ERP.
