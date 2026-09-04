# VisionGuard — AI Video Intelligence for Warehouse Handling

**An AI field-intelligence assistant for safer, damage-free warehouse loading and unloading.**

Built for the Godrej Enterprises Group (GEG) challenge *AI Video Intelligence for Warehouse Handling*.

[![Python 3.11](https://img.shields.io/badge/python-3.11-brightgreen.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-Vite-61dafb.svg)](https://vitejs.dev)
[![YOLO-World](https://img.shields.io/badge/Ultralytics-YOLO--World-FF5722.svg)](https://ultralytics.com)

---

## 1. What this is

Traditional CCTV records what happened. VisionGuard **understands what is happening** and turns it
into an intervention a supervisor can act on before product is damaged:

```
Camera → AI perception → Object tracking → Temporal behaviour reasoning
       → Risk classification → Evidence + explanation → Supervisor intervention → Prevention
```

The system ingests recorded or live warehouse footage, detects operators and products, tracks them
across frames, reasons over **sequences of motion states** (not single frames), classifies damage
risk with a fully auditable score, and answers supervisor questions strictly from what it recorded.

### The honesty principle

This README states measured capability, not aspiration. Where something does not work reliably on
the pilot footage, it says so and says why. **Section 7** is the honest capability matrix, and the
dashboard's *Detection Coverage* tab renders the same statuses generated directly from the detector
code, so the UI cannot claim more than the implementation delivers.

---

## 2. Quick start

```bash
git clone <repo> && cd AI-Warehouse-Intelligence
python -m venv .venv && .venv\Scripts\activate       # Windows
pip install -r requirements.txt
cp .env.example .env
```

```bash
cd dashboard && npm install && npm run build && cd ..
```

```bash
python process_all_pilot_videos.py
```

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000>.

For frontend development with hot reload, run the API as above and in a second terminal:

```bash
cd dashboard && npm run dev
```

---

## 3. Architecture

```
                        Warehouse video (CCTV / smartphone / upload)
                                          │
                          ┌───────────────▼───────────────┐
                          │  Perception  (detection/)     │
                          │  YOLO-World, prompted with    │
                          │  warehouse nouns:             │
                          │  person · box · carton ·      │
                          │  package · mattress · pallet  │
                          │  · trolley · forklift · truck │
                          └───────────────┬───────────────┘
                                          │  Detections mapped to a strict
                                          │  taxonomy — unknown classes are
                                          │  dropped, never guessed.
                          ┌───────────────▼───────────────┐
                          │  Tracking  (detection/tracker)│
                          │  · persistent track IDs       │
                          │  · velocity in frame-heights/s│
                          │  · motion state machine       │
                          │  · fitted ground plane        │
                          └───────────────┬───────────────┘
                                          │  Sequences of states, not frames
                          ┌───────────────▼───────────────┐
                          │  Behaviour engine (behaviour/)│
                          │  12 detectors over temporal   │
                          │  state transitions + scene    │
                          │  context (bay/shift/floor)    │
                          └───────────────┬───────────────┘
                                          │
                          ┌───────────────▼───────────────┐
                          │  Risk engine (risk/)          │
                          │  base weight + named, signed  │
                          │  factors → 0–100 → LOW/MED/   │
                          │  HIGH/CRITICAL, fully audited │
                          └───────────────┬───────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              ▼                           ▼                           ▼
   Evidence (video/)            SQLite (backend/database)    FastAPI (backend/app)
   · annotated stream           · videos, incidents          · REST + validation
   · still evidence frame       · risk factor breakdown      · static media
   · replay clip                · temporal stage chain       · SPA hosting
   · optional face blurring     · human review status
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              ▼                                                       ▼
   React dashboard (dashboard/)                    Grounded assistant (assistant/)
   · AI-overlay / original toggle                  · intent → SQL retrieval → answer
   · incident timeline + scrubber                  · says "not enough evidence" when true
   · evidence inspector + score audit              · optional LLM may only reword
   · analytics, prevention, coverage
```

### Why an open-vocabulary detector

COCO — what stock YOLOv8 is trained on — has **no class for a cardboard carton, pallet or trolley**.
The first implementation mapped every unrecognised COCO class onto "carton", so kites, umbrellas,
skis and books became warehouse products, and their erratic frame-to-frame jitter produced hundreds
of phantom "throwing" and "stepping" incidents.

VisionGuard uses **YOLO-World** prompted with warehouse nouns, which detects real cartons, and a
**strict taxonomy with no catch-all fallback**: a detection that maps to nothing meaningful is
discarded. If the open-vocabulary weights or CLIP text encoder are unavailable the system falls back
to COCO YOLOv8 and says so in `/api/health` — in that mode products are not detectable and only
person-based reasoning works.

---

## 4. Temporal reasoning — the core of the challenge

The challenge asks for **object detection + tracking + action recognition + temporal reasoning +
risk classification**, and explicitly not "person + box detected".

Every track carries a motion state — `STATIONARY`, `CARRIED`, `SLIDING`, `FALLING`, `SETTLED` — and
detectors consume **transitions between them**. A drop is only reported when the whole chain is
observed:

```
operator contact → sustained descent over consecutive samples
                 → abrupt velocity collapse (impact)
                 → object remains where it landed
```

Break any link and no event is emitted: a steady descent that never stops is a controlled lowering;
a fast object nobody was holding is not a throw. Each stored incident carries its
`evidence_stages` chain, which the dashboard renders as
`carried@4.1s → falling@4.4s → settled@4.7s`, so a supervisor can see the reasoning rather than
trust a label.

All kinematic thresholds are expressed in **frame-heights per second**, not pixels, so they hold
across camera resolutions and zoom levels.

---

## 5. Risk scoring — transparent by construction

Risk is never random and never a bare constant. Each score is a base weight for the behaviour class
plus **named, signed contributions** from measured quantities, scene context and history:

```
Behaviour class baseline        +62   product drop carries inherent handling-damage risk
Drop height                     +18   fall of approximately 1.1 m (estimated against operator stature)
Impact velocity                 +10   peak descent 1.04 frame-heights/s before floor contact
Abrupt deceleration              +8   velocity collapsed within one analysis interval
Uncontrolled landing             +4   product came to rest where it landed
Product sensitivity             +15   knock-down furniture package (panel/hinge sensitive)
Moderate perception confidence   -6   detection confidence 0.54
                                ────
                                 98   → CRITICAL
```

The breakdown is persisted with every incident, returned by the API and rendered in the evidence
inspector. Drop height in metres is a **monocular estimate** scaled against observed operator
stature — the README, the UI and the assistant all describe it as approximate.

Contributing factors: behaviour type, movement kinematics, estimated drop height, impact indication,
duration, stacking geometry, product fragility, recurrence count, bay, and detection confidence
(which *reduces* the score when perception evidence is weak).

---

## 6. Responsible AI

The challenge asks for a defensible distinction, and the system enforces it in the data model:

```
OBSERVED_BEHAVIOUR  →  POTENTIAL_RISK  →  CONFIRMED_DAMAGE
```

* The vision pipeline can only ever emit the first two tiers. `CONFIRMED_DAMAGE` is reachable
  **exclusively** through `PATCH /api/incidents/{id}/review` — a human decision.
* Evidence frames are captioned **"POTENTIAL DAMAGE RISK"**, never "damaged", and carry the line
  *"Requires human review before any corrective decision."*
* **Faces are blurred by default** in stored evidence (`BLUR_FACES_IN_EVIDENCE=true`). Incidents are
  keyed by anonymous track IDs; no identity, name or worker ID is stored anywhere.
* Every recommendation targets the **process** — equipment, sequence, coaching, floor condition —
  and a test asserts that no recommendation contains punitive language.
* Low detection confidence explicitly *reduces* the risk score and is surfaced as a factor, so weak
  evidence cannot produce a confident accusation.
* Impact is framed as prevention: *"N high-risk events identified, each an opportunity to intervene
  before damage occurred"*, never *"N products damaged"*.
* Retention is configurable (`EVIDENCE_RETENTION_DAYS`).

---

## 7. Capability matrix — measured, not claimed

Statuses below are generated from the detector classes and mirrored at `/api/capabilities` and in
the dashboard's *Detection Coverage* tab.

| # | Behaviour (GEG scope) | Detection method | Status | Principal limitation |
|---|---|---|---|---|
| 1 | Product dropped | descent → impact → at-rest state chain | **Implemented** | Drop height is a monocular estimate |
| 2 | Product dragged | sustained floor-plane sliding with operator contact | **Implemented** | Cannot separate pushing from pulling without pose |
| 3 | Product thrown / pushed | release velocity + unsupported flight phase | **Implemented** | Fast low-contrast throws can break the track mid-flight |
| 4 | Rolling / tumbling | cyclical aspect-ratio inversion on the floor | **Partial** | Axis-symmetric items rotate without aspect change |
| 5 | Improper / unstable stacking | persistent overhang or heavy-on-light geometry | **Implemented** | Weight inferred from size and class, not measured |
| 6 | Stepping on cartons | feet above the ground plane *at the operator's own depth* | **Implemented** | Assumes a single continuous ground plane |
| 7 | Handled without required equipment | large item carried with no trolley in scene | **Implemented** | Absence in frame ≠ proof of unavailability |
| 8 | Handling on wet floor | declared floor condition + observed floor movement | **Partial** | Wet condition is **not** sensed from video (see below) |
| 9 | Upright product kept flat | observed upright→flat transition of one tracked item | **Partial** | Handling arrows are not read; already-flat items are not judged |
| 10 | Dock level / transition hazard | declared dock + unaided heavy slide | **Partial** | Gap height is not measured |
| 11 | Product outside designated area | settled outside a configured staging polygon | **Requires zone configuration** | Cannot know where goods belong without a zone |
| 12 | Unsafe loading sequence | concurrent uncontrolled handling at one transfer point | **Partial** | Detects congestion, not adherence to a specific plan |

This satisfies the challenge's requirement to demonstrate at least 10 predefined behaviours, with
12 defined and their real state declared.

### What is deliberately *not* automated

**Wet-floor sensing.** A specular-reflection classifier (bright, low-saturation floor pixels) was
implemented and evaluated across all seven pilot clips. It did not separate wet from dry — the
wet-floor clip scored *lower* than two dry clips. Rather than ship a detector that would fabricate
hazards, floor condition is a declared scene input (ingest form, supervisor report or floor sensor),
and the detector reliably does the remaining half: deciding whether goods were actually moved
through the affected area. The previous implementation switched this on by matching the word "wet"
in the **filename**, which meant results depended on what a file was called.

**Strap-pulling.** Present in the pilot footage and in the GEG parameter table, but distinguishing
"gripping the strap" from "gripping the carton" requires hand/pose estimation at a resolution this
footage does not carry. The earlier heuristic (operator torso near a carton's top edge while it
moves) fired on 59 events, almost none of which were strap pulls. **It has been removed rather than
left in as a decorative capability.** Reinstating it needs a pose model plus labelled examples.

---

## 8. Measured performance on the pilot footage

Run the audits yourself:

```bash
python tests/audit_perception.py
```

```bash
python tests/audit_accuracy.py
```

### Measured perception coverage (30 frames sampled per clip)

| Clip | Frames with operator | Frames with product | Mean product conf |
|---|---|---|---|
| Rolling and dropping carton | 100 % | **83 %** | 0.41 |
| KD packets dragged, heavy box on other packets | 97 % | **80 %** | 0.39 |
| Dock level, dragging cupboard | 93 % | **47 %** | 0.34 |
| Throwing seating cartons, using strap | 100 % | **27 %** | 0.23 |
| Stepping on cartons, vertical kept horizontal | 100 % | **20 %** | 0.16 |
| Rolling and dragging on wet floor | 97 % | **7 %** | 0.15 |
| Throwing Mattresses | 97 % | **0 %** | 0.00 |
| **Mean** | **97 %** | **39 %** | 0.25 |

**Perception is the binding constraint, and it is measured.** Operator detection is reliable across
every clip (97 %). Product detection is not: the pilot videos are phone recordings *of a CCTV
monitor*, including the NVMS application chrome, at low effective resolution, with editorial
captions and arrows burnt over the action. Low-contrast tan cartons on wooden pallets and
plastic-wrapped mattresses are frequently not detected at any confidence threshold or inference
resolution — both were tested (0.03–0.25 confidence; 640/960/1280 px) and neither changed the
outcome.

The consequence is stated plainly: **behaviour recall is limited by product detection, not by the
behaviour logic.** The three clips with 0–20 % product detection are exactly the three that yielded
no events. A dropped carton that was never detected cannot produce a drop event. The behaviour
reasoning itself is verified independently of the detector by 40 synthetic-track tests.

### Ground-truth result on the pilot set

| Metric | Value |
|---|---|
| Ground-truth behaviours across the 7 clips | 14 |
| Detected | 3 |
| **Behaviour-level recall** | **21.4 %** |
| Events recorded in total | 13 across 3.1 minutes of footage |
| Behaviours observed | `product_drag` ×8, `unsupported_handling` ×3, `dock_level_hazard` ×2 |

Correctly matched: dragging and the dock-transition hazard on *Dock level, dragging cupboard*, and
dragging on *KD packets dragged*. Closing the remaining gap requires a detector fine-tuned on
warehouse packaging (a few hundred labelled frames from the target bays would be sufficient) — a
data problem, not a threshold problem.

### False-positive reduction achieved

Identical footage, identical ground truth:

| | Before | After |
|---|---|---|
| Incidents across the 7 pilot clips | **372** | **13** |
| Rated CRITICAL | 214 | 0 |
| `stepping_on_carton` (all rated CRITICAL) | 114 | 0 — eliminated as a systemic false positive |
| `strap_pulling` | 59 | removed — not reliably detectable |
| `product_throw` | 90 | 0 — the previous events were detector noise |
| Root cause of the bulk | every unknown COCO class mapped to "carton"; a fixed floor line flagged every *distant* worker as standing on a package | strict taxonomy with no fallback; depth-aware ground plane fitted from operator stature |

The earlier build reported 372 events on this footage and 306 of them as HIGH or CRITICAL. Almost
none were real: it detected kites, umbrellas and skis as cartons, and flagged background workers as
standing on packages. Both root causes are fixed and covered by regression tests.

---

## 9. AI operations assistant

Retrieval-first, so hallucination is structurally difficult:

```
question → intent classification + filter extraction → SQL retrieval → answer rendered from rows
```

Every number, timestamp, bay and behaviour in an answer is copied from a retrieved row, and the
response returns the rows it used so the dashboard can show the evidence. With no matching data it
returns *"The system does not have enough detected evidence to answer this."*

Supported questions include:

* "Show me all high-risk handling events"
* "What were the three most common risky behaviours?"
* "Which loading bay had the highest number of risky events?"
* "Why was this event classified as high risk?" — returns the full score breakdown and state chain
* "What corrective action is recommended?"
* "How many product drops were detected?"

An LLM is **optional and off by default**. When enabled it receives the already-retrieved answer and
may only reword it; it is never the source of facts.

---

## 10. API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Status and the **active detector backend** |
| `GET` | `/api/capabilities` | Code-derived capability matrix with real event counts |
| `GET` | `/api/videos` | Analysed videos with incident counts and playback URLs |
| `GET` | `/api/videos/{id}` | One video plus its incidents |
| `DELETE` | `/api/videos/{id}` | Remove a video and its incidents |
| `GET` | `/api/videos/{id}/status` | Live analysis progress, stage and errors |
| `POST` | `/api/videos/upload` | Ingest with scene context (bay, shift, floor, dock, zone) |
| `POST` | `/api/videos/{id}/analyze` | Re-analyse an existing video |
| `GET` | `/api/incidents` | Filter by video, risk, behaviour, bay, shift, free-text search |
| `GET` | `/api/incidents/{id}` | Full incident with risk factors and stage chain |
| `PATCH` | `/api/incidents/{id}/review` | **Human review** — the only route to confirmed damage |
| `GET` | `/api/analytics` | Risk mix, behaviour Pareto, by bay / shift / video |
| `GET` | `/api/prevention` | Recurring behaviours, hotspots, training topics, baseline |
| `POST` | `/api/assistant/chat` | Grounded supervisor assistant |

Interactive docs at `/docs`.

---

## 11. Testing

```bash
python -m pytest
```

110 tests covering the risk engine (determinism, factor attribution, thresholds, bounds, no confirmed
damage, no punitive language), behaviour detectors (positive chains **and** the negative cases that
previously produced false incidents), tracker kinematics and the ground-plane fit, database
operations and migrations, all API endpoints including validation and error paths, assistant routing
and hallucination resistance, and an end-to-end pipeline run on a synthesised clip.

The two audit scripts (`tests/audit_perception.py`, `tests/audit_accuracy.py`) are reporting tools,
not pass/fail gates — a gate on recall could be satisfied by loosening detectors, which is precisely
what this project is trying not to do.

---

## 12. Security & configuration

* All configuration is environment-driven via `config.py`; `.env` is git-ignored and
  `.env.example` documents every setting. No secrets in source.
* CORS origins come from config. Credentials are automatically disabled when a wildcard origin is
  used, because browsers reject that combination.
* Uploads: extension allow-list, size cap enforced **while streaming**, empty-file rejection,
  filename sanitisation (directory components stripped), and a video-id prefix so re-uploading the
  same filename cannot overwrite an earlier recording.
* Path traversal is blocked: IDs are validated against a strict pattern and served files are
  resolved by basename inside their configured directory.
* Background analysis failures are caught, logged and recorded against the video row, so a crashed
  job surfaces as an error in the UI instead of a task stuck at 99%.

---

## 13. Repository layout

```
config.py                     Central env-driven configuration
backend/
  app.py                      FastAPI: routes, validation, error handling, SPA hosting
  database/                   schema.sql, migrations.sql, db.py
detection/
  object_classes.py           Warehouse taxonomy, prompts, strict mapping
  detector.py                 YOLO-World / COCO backends, per-class thresholds
  tracker.py                  Persistent tracking, normalised kinematics, ground plane
behaviour/
  base.py                     Event schema, evidence tiers, implementation status
  kinematic_detectors.py      drop, throw, drag, roll
  spatial_detectors.py        stacking, stepping, orientation, equipment, zone, sequence
  scene_detectors.py          wet floor, dock level
  behaviour_engine.py         Orchestration, scene context, coverage report
risk/risk_engine.py           Transparent multi-factor scoring
video/
  processor.py                End-to-end pipeline, progress, error capture
  evidence.py                 Overlays, evidence frames, replay clips, face blurring
assistant/llm.py              Retrieval-grounded supervisor assistant
dashboard/src/                React dashboard
tests/                        pytest suite + audit scripts
docs/                         Architecture, behaviour taxonomy, demo script, deck outline
process_all_pilot_videos.py   Reproducible pilot analysis
```

---

## 14. Known limitations

1. **Product detection recall on the pilot footage** (Section 8) — the dominant constraint.
2. **Wet-floor and dock conditions are declared, not sensed** (Section 7).
3. **Designated-area detection needs a staging polygon** per camera; unconfigured, it stays silent.
4. **Strap-pulling is not implemented** — needs pose estimation.
5. **Single-camera only** — no cross-camera re-identification.
6. **Near-real-time, not real-time** — roughly 0.3× realtime on CPU at stride 3. A GPU or a smaller
   model raises this; the code already supports frame striding and inference downscaling.
7. **Improvement tracking needs a second batch** — one pilot batch establishes the baseline only.
