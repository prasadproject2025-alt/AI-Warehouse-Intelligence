# VisionGuard — Demo Script

A 6–8 minute walkthrough that shows real detections, states the system's limits honestly, and lands
the damage-prevention message. Judges reward a working prototype that knows its own boundaries more
than a polished one that overclaims.

---

## 0. Before you start (10 minutes ahead)

```bash
python process_all_pilot_videos.py
```

```bash
cd dashboard && npm run build && cd ..
```

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000> and confirm:

- [ ] Header shows **Backend online** and **Perception: YOLO-World open-vocabulary**
      (if it says *COCO fallback*, CLIP is missing — reinstall from `requirements.txt`)
- [ ] KPI strip shows non-zero counts
- [ ] The Operations tab plays video with **AI overlay** selected
- [ ] The Detection Coverage tab renders

Have a spare short clip ready on the desktop for the live-ingest step.

---

## 1. Frame the problem (45 s)

> "Warehouses lose product to *handling behaviour* — dropping, dragging, bad stacking — not to
> equipment failure. Traditional CCTV records the damage. It tells you what already went wrong.
> VisionGuard is built to catch the behaviour that *causes* damage, while there is still time to
> intervene."

Point at the KPI strip.

> "Every number here comes from footage the system actually analysed. Nothing is seeded."

---

## 2. Perception and tracking (60 s)

Operations tab, video playing with **AI overlay** on.

> "Persistent track IDs, entity class, and each object's current motion state — carried, sliding,
> falling, settled. The arrows are velocity vectors."

Toggle to **Original**, then back.

> "Same footage, and here is what the AI sees on top of it."

Say the important technical point:

> "Stock YOLO is trained on COCO, which has no class for a cardboard carton or a pallet. So we run
> an open-vocabulary detector prompted with warehouse nouns — carton, pallet, trolley, mattress.
> That is what makes product-level reasoning possible at all."

---

## 3. Temporal reasoning — the heart of it (90 s)

Click a **product drag** event on the timeline. The video seeks to it; the inspector opens.

> "This is not 'a person and a box were detected'. Read the temporal sequence."

Point at the stage chain, e.g. `carried@12.4s → sliding@12.9s → sliding@14.1s`.

> "The system required the carton to be in a sliding state, on the floor plane, with an operator in
> contact range, continuously for over a second, before it would call this a drag. A single fast
> frame is not a drag. If any link in that chain is missing, no event is raised."

---

## 4. Transparent risk (60 s)

Scroll to **Risk score breakdown** in the same inspector.

> "No black box, and no random number. The score is a base weight for the behaviour class plus named
> contributions we can each defend: drag distance, duration, whether handling equipment was present.
> Note this one" — point to a negative factor — "low detection confidence *reduces* the score.
> Weak evidence cannot produce a confident accusation."

---

## 5. Responsible AI (45 s) — do not skip this

Point at the amber banner in the inspector.

> "It says **Potential damage risk**, not 'damaged'. The pipeline can only ever report observed
> behaviour and potential risk. Confirmed damage is reachable only through this human review
> control" — point at the review buttons — "which is a person's decision after physical inspection."

> "Faces are blurred in stored evidence by default. Incidents are keyed to anonymous track IDs — no
> names, no worker IDs. Every recommendation targets the process: equipment, sequence, coaching.
> This is a damage-prevention tool, not a surveillance tool."

---

## 6. Honest coverage (45 s) — this is a strength, present it as one

Open the **Detection Coverage** tab.

> "This table is generated from the detector code itself, so the UI cannot claim a capability the
> implementation does not have. Twelve behaviours defined, statuses declared honestly."

Point at wet floor.

> "Wet-floor handling is marked partial, and here is why. We built a specular-reflection classifier
> and tested it on all seven clips. It could not separate wet from dry — the wet clip actually scored
> lower than two dry ones. So rather than fabricate hazards, floor condition is a declared input,
> and the detector does the half it can do reliably: was product actually moved through that area."

> "We also removed a strap-pulling detector that looked good in a demo and was measurably wrong —
> it fired 59 times, and zero of those were on the clip that actually shows strap use."

---

## 7. Analytics and prevention (60 s)

**Shift Analytics** tab.

> "Behaviour Pareto, risk by bay, per-video rates."

**Prevention & Learning** tab.

> "Recurring behaviours mapped to specific coaching topics, high-risk locations, and the baseline
> rate — elevated-risk events per minute of footage. That is the number you re-measure after
> coaching to prove the intervention worked. One batch establishes the baseline; the improvement
> claim needs the second batch, and we say so."

---

## 8. AI assistant (60 s)

**AI Assistant** tab. Use the quick prompts.

1. *"What were the three most common risky behaviours?"* — ranked from real counts.
2. *"Which loading bay had the highest number of risky events?"* — real bay names.
3. *"Why was this event classified as high risk?"* — full score breakdown and state chain.

Then the honesty test — type a behaviour that is **not** in the data:

> *"How many mattress throwing events were detected?"*

> "It says it has no such events rather than inventing a number. The assistant retrieves rows first
> and renders the answer from them, so every figure traces back to a database row. An LLM is
> optional and can only reword an answer that was already retrieved."

Point at the grounding footer under the answer.

---

## 9. Live ingest (60 s)

**Ingest Video** tab.

> "Scene context first — bay, shift, camera, floor condition, whether this camera covers a dock.
> That is site knowledge a camera cannot infer, and it is what the wet-floor and dock detectors
> reason against. The previous version guessed it from the filename."

Upload the spare clip. Let the progress bar run.

> "Real progress from the backend — frames analysed, events found so far."

---

## 10. Close (30 s)

> "The framing matters. Not 'we detected 25 damaged products' — that is a post-mortem. Instead:
> we identified N high-risk handling events, each one a chance to intervene before damage occurred.
> That is the shift from damage detection to damage prevention."

> "And we know exactly what stands between this and production: a detector fine-tuned on warehouse
> packaging. A few hundred labelled frames from the target bays. We measured that gap rather than
> hiding it."

---

## Q&A preparation

**"Why do some clips show no events?"**
> Product detection is the binding constraint. These clips are phone recordings of a CCTV monitor,
> including the application chrome, with captions burnt over the action. Low-contrast cartons on
> wooden pallets are frequently not detected. Run `python -m tests.audit_perception` — we measured
> the product detection rate per clip rather than guessing. Behaviour logic is verified independently
> by 35 synthetic-track tests.

**"How do I know the risk scores are not arbitrary?"**
> Open any incident: the score is the sum of named factors, each shown with its point contribution
> and justification. `test_score_equals_sum_of_published_factors` asserts that the score is exactly
> the sum, and `test_scoring_is_deterministic` asserts it never varies between runs.

**"What was wrong with the earlier version?"**
> It reported 372 incidents on this footage; most were false. Every unrecognised COCO class was
> mapped to "carton", and a fixed floor line flagged every distant worker as standing on a package —
> 114 CRITICAL false positives alone. Both root causes are fixed and covered by regression tests.

**"Can it run in real time?"**
> Roughly 0.3× realtime on CPU at stride 3. Frame striding and inference downscaling are already in
> the config; a GPU or a smaller backbone closes the gap. Nothing in the architecture assumes offline
> processing.

**"Is this surveillance?"**
> No identity is stored anywhere — incidents are keyed to anonymous track IDs and faces are blurred
> by default. The output is process recommendations, and a test asserts that no recommendation
> contains punitive language.
