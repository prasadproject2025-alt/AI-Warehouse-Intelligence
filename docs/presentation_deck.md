# VisionGuard — Presentation Deck Outline

GEG requirement: minimum 5, maximum 6 slides. This is 6.

Fill the bracketed figures from a live run:

```bash
python process_all_pilot_videos.py
python tests/audit_accuracy.py
python tests/audit_perception.py
python -m pytest
```

---

## Slide 1 — Solution & Team

**VisionGuard**
*AI Video Intelligence for Warehouse Handling*

**Team:** [team name] · [3–5 members with roles]

**Value proposition (one line):**
> Turns existing warehouse cameras into a field-intelligence assistant that understands handling
> behaviour and flags damage risk while there is still time to intervene.

Visual: dashboard hero screenshot with the KPI strip and an incident selected.

---

## Slide 2 — Problem, Solution & User Journey

**The gap**

```
Traditional CCTV:  Camera → Recording → Human review → Damage discovered → Corrective action
VisionGuard:       Camera → AI perception → Behaviour understanding → Risk detection
                          → Alert → Intervention → Prevention
```

**The supervisor's journey**

1. Camera covers a loading bay; footage is ingested with its scene context (bay, shift, floor
   condition, dock).
2. AI detects operators and products, tracks them, and reasons over motion **sequences**.
3. A risky handling sequence raises an alert with evidence, an explanation and a recommended action.
4. Supervisor reviews the evidence frame and clip, and records the outcome — including whether
   damage was actually found.
5. Recurring behaviours become named coaching topics; the elevated-risk rate becomes the metric that
   proves the coaching worked.

Visual: the flow diagram plus the evidence inspector showing the temporal stage chain.

---

## Slide 3 — Technical Architecture & Stack

```
Video → Open-vocabulary detection → Persistent tracking → Temporal state machine
      → 12 behaviour detectors → Transparent risk engine → Evidence + SQLite
      → FastAPI → React dashboard + Grounded assistant
```

| Layer | Technology | Why |
|---|---|---|
| Computer vision | **YOLO-World** (open-vocabulary) | COCO has **no carton, pallet or trolley class** — this is what makes product-level reasoning possible |
| Tracking | Custom IoU + proximity tracker | Persistent IDs, velocity in frame-heights/s (resolution-independent), motion state machine |
| Action recognition | 12 temporal detectors | Consume state *transitions*, never single frames |
| Risk | Transparent multi-factor engine | Every point attributable to a named factor |
| LLM | Retrieval-grounded assistant (LLM optional) | Retrieval-first, so hallucination is structurally hard |
| Video processing | OpenCV | Decode, overlays, evidence frames, replay clips |
| Backend | FastAPI + SQLite (WAL) | Validated REST, background analysis, migrations |
| Frontend | React + Vite | Overlay playback, timeline, evidence audit, analytics |
| Edge/cloud | CPU-capable; frame striding + inference downscaling | Deployable on bay-side hardware |

Highlight the challenge's own formula and where each part lives:
**Object detection + Object tracking + Action recognition + Temporal reasoning + Risk classification.**

---

## Slide 4 — Prototype & Demo

Screenshots (four panels):

1. **AI overlay** — track IDs, entity classes, motion states, velocity vectors.
2. **Incident timeline + evidence inspector** — the temporal stage chain
   `carried → falling → settled` with timestamps.
3. **Risk score breakdown** — named factors with their point contributions, including a *negative*
   factor for low detection confidence.
4. **Detection Coverage tab** — the honest capability matrix, generated from the detector code.

Demo video (3–5 scenarios): dragging at the dock, dock-transition hazard, handling without
equipment, an assistant question, and a live ingest with scene context.

Caption to include verbatim:
> Every figure shown is produced by the pipeline from real footage. No detections, statistics or
> demo data are seeded.

---

## Slide 5 — Impact, Damage Prevention & Responsible AI

**The framing shift**

> Not *"we detected 25 damaged products"* — that is a post-mortem.
> Instead: **"we identified [N] high-risk handling events, each an opportunity to intervene before
> damage occurred."**

**Metrics the system produces**

| Metric | Value |
|---|---|
| Footage analysed | [N] minutes across [N] clips |
| Risk events detected | [N] |
| Intervention opportunities (high + critical) | [N] |
| Baseline elevated-risk rate | [N] per minute — the number to re-measure after coaching |
| False positives eliminated vs first build | 372 → [N] events on identical footage |

**Responsible AI, built into the data model**

* `OBSERVED_BEHAVIOUR → POTENTIAL_RISK → CONFIRMED_DAMAGE` — the pipeline can emit only the first
  two; confirmed damage requires a human review action.
* Faces blurred by default; incidents keyed to anonymous track IDs; no identity stored.
* Recommendations target the process, not the person — asserted by an automated test.
* Low perception confidence *reduces* the risk score and is shown as a factor.

**Users to validate with:** warehouse supervisor, loading/unloading operator, logistics manager,
quality professional, safety professional. Record what each observed and what changed as a result.

---

## Slide 6 — Honest Status & Roadmap

**Working today**

* 12 behaviours defined; [N] implemented, [N] partial, 1 requiring zone configuration
* Full temporal reasoning with auditable evidence chains
* Transparent risk scoring; grounded assistant; human-review workflow
* 110 automated tests, plus two measurement audits

**Measured limits — stated, not hidden**

* **Product detection recall is the binding constraint.** The pilot clips are phone recordings *of a
  CCTV monitor* with application chrome and burnt-in captions; low-contrast cartons are frequently
  not detected. Measured per clip by `audit_perception.py`. Behaviour logic is verified
  independently by 40 synthetic-track tests.
* **Wet-floor sensing is not automated.** A specular classifier was built and tested; it could not
  separate wet from dry on this footage, so condition is a declared input rather than a fabricated
  detection.
* **Strap-pulling was removed**, not left in — the heuristic fired 59 times and zero of those were on
  the clip that actually shows strap use. It needs pose estimation.

**Next**

1. Fine-tune the detector on a few hundred labelled frames from the target bays — the single highest
   -leverage change.
2. Second footage batch to convert the baseline rate into a measured improvement claim.
3. Pose estimation for grip-level behaviours; multi-camera tracking; live RTSP ingest.

Closing line:
> We would rather show a prototype that knows exactly where its limits are than one that hides them.
