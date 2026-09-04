# VisionGuard — Technical Architecture

## 1. Pipeline

```
Video (recorded / uploaded / live-capable)
   │
   ├─ decode ──────────────── OpenCV VideoCapture; invalid dimensions and undecodable
   │                          files fail fast with a clear error recorded on the video row
   │
   ├─ stride ──────────────── every Nth frame reaches the detector (DETECTION_FRAME_STRIDE);
   │                          the annotated writer still receives every frame
   │
   ├─ downscale ───────────── one resize before inference (MAX_INFERENCE_WIDTH); boxes are
   │                          mapped back to full resolution afterwards
   │
   ├─ DETECT ──────────────── YOLO-World prompted with warehouse nouns
   │                          → strict taxonomy mapping, no catch-all
   │                          → per-class confidence gates
   │                          → cross-class NMS (open-vocab prompts overlap)
   │
   ├─ TRACK ───────────────── greedy IoU + proximity association
   │                          → persistent track IDs
   │                          → velocity in frame-heights/s
   │                          → motion state machine
   │                          → operator-contact assignment
   │                          → fitted ground plane
   │
   ├─ REASON ──────────────── 12 behaviour detectors over state transitions
   │                          + declared scene context + recurrence history
   │
   ├─ SCORE ───────────────── base weight + named signed factors → 0–100 → risk level
   │
   ├─ EVIDENCE ────────────── annotated frame, replay clip, optional face blurring
   │
   └─ PERSIST ─────────────── SQLite: incident + factor breakdown + stage chain
```

## 2. Why open-vocabulary detection

COCO contains no class for a carton, pallet or trolley. Mapping unrecognised COCO classes onto
"carton" — the first implementation's fallback — turned kites, umbrellas, skis and books into
warehouse products whose frame-to-frame jitter generated hundreds of phantom incidents.

`detection/object_classes.py` defines the taxonomy and two mapping tables. `map_label()` returns
`None` for anything without a defensible warehouse meaning, and `detector.py` drops those
detections. The open-vocabulary backend (`yolov8s-worldv2` + CLIP text encoder) is prompted with
`person, box, carton, package, mattress, pallet, trolley, forklift, truck`. If the weights or CLIP
are unavailable the system degrades to COCO YOLOv8, logs a warning, and reports the active backend
at `/api/health` — in that mode products are not detectable and this is stated rather than hidden.

Products score materially lower than people under an open-vocabulary head, so thresholds are
per-class (`PERSON_CONF`, `PRODUCT_CONF`, `EQUIPMENT_CONF`) rather than global. Because several
prompts fire on the same physical object ("box"/"carton"/"package"), a cross-class NMS pass keeps
the highest-confidence detection per object while never merging an operator with a product.

## 3. Normalised kinematics

Velocities are expressed in **frame-heights per second**, not pixels per second:

```python
inst_vx = (new_center[0] - self.center[0]) / self.frame_height / dt
```

Every downstream threshold is therefore independent of camera resolution and zoom. The test
`test_velocity_is_resolution_independent` asserts that identical physical motion at 720p and 1440p
yields the same velocity.

## 4. Motion state machine

Each non-operator track carries one of:

| State | Condition |
|---|---|
| `STATIONARY` | speed below threshold, no recent motion |
| `CARRIED` | moving with an operator inside their reach envelope |
| `SLIDING` | horizontal motion with the base in the floor band |
| `FALLING` | sustained downward velocity dominating horizontal |
| `SETTLED` | at rest **after** a period of motion |

The `STATIONARY`/`SETTLED` split matters: "came to rest after moving" is evidence of an impact,
whereas "never moved" is not. Operator reach scales with the operator's apparent size, which is a
depth proxy, so contact assignment does not use a fixed pixel radius.

## 5. Ground-plane fit

A single global floor line fails on a receding ground plane: distant workers legitimately have feet
high in the frame, which produced 114 CRITICAL false "stepping on carton" events on the pilot set.

For anyone standing on the floor, foot position varies almost linearly with apparent stature (a
depth cue), so `PersistentTracker.expected_floor_y()` least-squares fits `foot_y ≈ a·height + b`
over observed operators and returns the floor height **at that operator's depth**. Elevation is the
residual above it, and must exceed 2.5× the fit's own standard deviation. Where the sample lacks
depth spread the fit returns `None` and the detector stays silent.

## 6. Temporal reasoning

Detectors consume state *transitions*, not frames. `DropDetector._find_fall_then_impact` scans
recent history for a sustained descent immediately followed by a velocity collapse, then separately
requires prior operator contact (or clear initial elevation) and post-impact rest. Every detector
also enforces a minimum dwell or duration, minimum track maturity (4–8 hits), and a per
`(behaviour, track)` cooldown so one continuous action yields one incident.

Each incident stores its `evidence_stages` chain, which the API returns and the dashboard renders.

## 7. Risk scoring

`RiskEngine.evaluate()` returns a `RiskAssessment` carrying `RiskFactor(name, points, detail)`
entries. The score is the clamped sum; the level follows published thresholds
(CRITICAL ≥82, HIGH ≥64, MEDIUM ≥42). Factor groups: behaviour baseline, measured physical
quantities, product fragility, recurrence, and a **negative** contribution for low detection
confidence.

Nothing is random — asserted by `test_scoring_is_deterministic`.

## 8. Evidence tiers

```
OBSERVED_BEHAVIOUR → POTENTIAL_RISK → CONFIRMED_DAMAGE
```

The pipeline can emit only the first two. `CONFIRMED_DAMAGE` is set exclusively via
`PATCH /api/incidents/{id}/review`, and `test_never_asserts_confirmed_damage` asserts the engine
cannot produce it for any behaviour at any parameters.

## 9. Scene context

Bay, shift, camera, floor condition, dock status and staging zone are supplied at ingest and stored
on the video and every incident. They drive the wet-floor, dock and designated-area detectors, and
the per-bay and per-shift analytics.

The previous build derived wet/dock status by matching substrings in the **filename**, so results
depended on what a file was called. Scene context is now site knowledge, provided explicitly.

## 10. Persistence

SQLite with WAL. `schema.sql` holds the base tables; `migrations.sql` holds additive `ALTER TABLE`
statements applied idempotently at start-up, with duplicate-column errors ignored so the same
database upgrades cleanly. Comment lines are stripped before splitting on `;` — a trailing `--`
comment would otherwise swallow the following statement.

Writes are serialised through a module-level lock because the pipeline writes from a worker thread
while the API reads from request threads.

## 11. API layer

FastAPI. Pydantic models with validators for request bodies, explicit range checks on query
parameters, strict ID patterns, and a catch-all exception handler that logs the traceback and
returns a generic 500 without leaking internals. CORS origins come from config, and credentials are
disabled automatically when a wildcard origin is configured.

Uploads stream to disk with the size cap enforced during the write, sanitise the filename to a
basename with unsafe characters replaced, and prefix the stored name with the video id so
re-uploading the same filename cannot overwrite an earlier recording. Background analysis is wrapped
so failures are logged and written to the video row rather than vanishing.

## 12. Assistant

Retrieval-first: intent classification and filter extraction, then a parameterised SQL query, then
an answer rendered from the returned rows. The response includes the rows used, so grounding is
visible. Intent ordering places specific phrasings first — "most common risky behaviours" contains
the token "risk" and previously routed to the high-risk event list.

An LLM is optional, off by default, and receives the already-retrieved answer with instructions to
reword only.

## 13. Frontend

React + Vite. Vite proxies `/api` and `/static` to the backend in development; in production FastAPI
serves the built SPA. State is fetched from the API with explicit loading, empty and error states at
every level, and a dismissible error banner for API failures.

The *Detection Coverage* tab renders `/api/capabilities`, which is generated from the detector
classes, so the UI cannot claim a capability the code does not implement.

## 14. Performance

| Lever | Setting | Effect |
|---|---|---|
| Frame stride | `DETECTION_FRAME_STRIDE=3` | ~3× fewer inferences; annotated video still full-rate |
| Inference downscale | `MAX_INFERENCE_WIDTH=960` | one resize per analysed frame |
| Inference size | `INFERENCE_IMGSZ=640` | 640 vs 1280 measured ~3× faster for comparable recall |
| Model load | one shared `VideoProcessor` | weights loaded once per process, not per request |
| History buffer | 60 records/track | bounded memory regardless of video length |
| Evidence clips | second pass, `GENERATE_EVIDENCE_CLIPS` | can be disabled for throughput |

Measured on the pilot set: roughly 0.3× realtime on CPU.
