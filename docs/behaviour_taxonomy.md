# VisionGuard — Behaviour Taxonomy & Detection Reference

Maps the GEG challenge's good/bad-practice parameters onto what VisionGuard implements, with the
detection method, evidence requirements and honest status for each.

Statuses here mirror the values declared on the detector classes themselves and served by
`GET /api/capabilities`, so this document cannot drift from the code.

---

## Status vocabulary

| Status | Meaning |
|---|---|
| **IMPLEMENTED** | Works from video alone with the stated evidence requirements met |
| **PARTIALLY_IMPLEMENTED** | Works, with a named structural limitation on when it can fire |
| **REQUIRES_ZONE_CONFIGURATION** | Logic complete; needs per-camera configuration to produce output |
| **REMOVED** | Was implemented, measured as unreliable, and deliberately taken out |

---

## 1. Product dropped — `product_drop` · IMPLEMENTED

* **GEG practice:** *Lift and place products gently. Never throw or drop a package.*
* **Temporal chain required:** operator contact (or item clearly elevated) → sustained descent over
  consecutive analysis samples → abrupt velocity collapse (impact) → item remains where it landed.
  Any missing link suppresses the event.
* **Measured:** peak descent speed, net descent, estimated drop height in metres (scaled against
  observed operator stature), impact deceleration, post-impact rest.
* **Risk drivers:** drop height (+18 above ~1 m), impact velocity, abrupt deceleration,
  uncontrolled landing, product fragility.
* **Limitation:** drop height is a monocular estimate, described as approximate everywhere.

## 2. Product dragged — `product_drag` · IMPLEMENTED

* **GEG practice:** *Use a trolley, pallet truck or suitable handling equipment instead of dragging.*
* **Requires:** ≥1.0 s of continuous `SLIDING` state, ≥0.10 frame-heights of travel, and an operator
  within contact range for ≥50 % of the window. Movement with nobody nearby is not a drag.
* **Risk drivers:** drag distance, duration, wet-floor contact (+16), absence of handling equipment.

## 3. Product thrown / pushed — `product_throw` · IMPLEMENTED

* **GEG practice:** *Place products one at a time in the correct position.*
* **Requires:** operator contact before release, no contact at peak speed, peak speed ≥0.70
  frame-heights/s, and ≥0.10 frame-heights of horizontal travel. Pure vertical motion is classified
  as a drop, not a throw.
* **Risk drivers:** release velocity, confirmed unsupported flight, landing on other goods.
* **Limitation:** fast low-contrast throws can break the track mid-flight; those are missed rather
  than reported.

## 4. Rolling / tumbling — `rolling_product` · PARTIALLY_IMPLEMENTED

* **GEG practice:** *Carry or move products using appropriate handling equipment; do not roll.*
* **Requires:** ≥2 aspect-ratio inversion cycles within 3.5 s, floor contact, and translation.
* **Limitation:** aspect-ratio inversion is a proxy for rotation. Axis-symmetric items (rolled
  mattresses, drums) rotate without changing aspect ratio and are under-detected. A rotated-box or
  segmentation model would be required.

## 5. Improper / unstable stacking — `improper_stacking` · IMPLEMENTED

* **GEG practice:** *Stack larger and heavier packets at the bottom; ensure the complete packet is
  supported.*
* **Requires:** vertical adjacency (base of upper meets top of lower), ≥50 % horizontal overlap,
  both items at rest, configuration held ≥1.5 s, and either >1.25× width inversion or a rigid
  package on a lighter carton.
* **Risk drivers:** heavy-on-light (+18), severe overhang (+14), persistence.
* **Limitation:** weight is inferred from apparent size and class, not measured. A small dense
  package on a large light one is not distinguishable from video.

## 6. Stepping / standing on cartons — `stepping_on_carton` · IMPLEMENTED

* **GEG practice:** *Never step, stand or walk on packages. Keep a clear working path.*
* **Requires:** operator's feet above the **ground plane at their own depth** by more than 2.5× the
  fit residual, horizontally inside the package footprint, held ≥0.6 s.
* **Why the ground plane matters:** on a receding floor a distant worker's feet are legitimately
  high in the frame. Testing against a single global floor line produced 114 CRITICAL false
  positives on the pilot set. The plane is fitted by least squares from operator apparent stature
  versus foot position; where the fit is unsupported the detector stays silent.
* **Limitation:** assumes one continuous ground plane; ambiguous on ramps and split-level docks.

## 7. Handled without required equipment — `unsupported_handling` · IMPLEMENTED

* **GEG practice:** *Use the correct equipment for movement — trolley or pallet truck.*
* **Requires:** item ≥0.22 frame-heights across, moved with operator contact for ≥1.5 s, and **no**
  trolley, pallet or forklift detected anywhere in the scene.
* **Framing:** reported as an opportunity to check equipment provisioning, not as a violation —
  absence from frame is not proof of unavailability.

## 8. Handling on a wet floor — `wet_floor_hazard` · PARTIALLY_IMPLEMENTED

* **GEG practice:** *Keep the loading area dry. Stop movement until unsafe floor conditions are
  corrected.*
* **Requires:** floor condition declared `wet` for the camera **and** goods observed moving in the
  floor band for ≥1.0 s.
* **Why the condition is declared:** a specular-reflection classifier was implemented and tested
  across all seven pilot clips; it did not separate wet from dry (the wet clip scored lower than two
  dry clips). Shipping it would have fabricated hazards. In deployment the condition comes from the
  ingest form, a supervisor report or a floor sensor.
* **Superseded approach:** the earlier build enabled this by matching `"wet"` in the *filename*.

## 9. Upright product kept flat — `orientation_violation` · PARTIALLY_IMPLEMENTED

* **GEG practice:** *Follow the specified product orientation and handling labels/arrows.*
* **Requires:** the same tracked item observed upright (aspect <0.80) and subsequently flat
  (aspect >1.30) for ≥1.5 s.
* **Why a transition is required:** from a ceiling camera most packages are wider than tall, so
  flagging every wide box (the earlier behaviour) is meaningless.
* **Limitation:** handling arrows are not read; an item already flat on entry cannot be judged and
  is not reported.

## 10. Dock level / transition hazard — `dock_level_hazard` · PARTIALLY_IMPLEMENTED

* **GEG practice:** *Use a proper dock leveller or bridge and ensure a safe, level transition.*
* **Requires:** camera declared as covering a dock transition, item ≥0.20 frame-heights, sliding
  ≥1.0 s, with leveller/trolley presence checked from the scene.
* **Limitation:** the gap height itself is not measured.

## 11. Product outside designated area — `outside_designated_area` · REQUIRES_ZONE_CONFIGURATION

* **GEG practice:** *Stage products systematically according to the vehicle-loading plan.*
* **Requires:** a staging polygon in normalised coordinates for the camera, plus a product settled
  outside it for ≥3 s.
* **Behaviour without configuration:** emits nothing and reports its status. The system does not
  guess where goods belong.

## 12. Unsafe loading / unloading sequence — `unsafe_loading_sequence` · PARTIALLY_IMPLEMENTED

* **GEG practice:** *Load products in a stable and planned sequence.*
* **Requires:** ≥3 products simultaneously in `CARRIED`/`SLIDING`/`FALLING` state at one transfer
  point, sustained ≥1.5 s.
* **Limitation:** detects concurrency and congestion, not adherence to a specific documented loading
  plan — verifying a plan needs the plan as an input.

---

## Removed: strap-pulling

*GEG practice: "Handle the carton using proper lifting points. Do not use packaging straps as
handles."*

Implemented in the first build as "operator torso near a carton's top edge while the carton moves".
On the pilot footage this produced 59 events, including 6 on a clip containing no strap use at all
and **zero** on the clip actually labelled as strap handling — an inverted result.

Distinguishing "gripping the strap" from "gripping the carton" requires hand or pose estimation at a
resolution the pilot footage does not carry. The heuristic was **removed rather than left in as a
decorative capability**. Reinstating it needs a pose model plus labelled examples of both grips.

---

## Cross-cutting design rules

1. **Normalised units.** All kinematic thresholds are in frame-heights per second, so they hold
   across resolutions and zoom levels.
2. **Temporal gating.** Every detector requires either a state chain or a minimum dwell/duration. No
   detector fires on a single frame.
3. **Track maturity.** Tracks must have ≥4–8 hits before behaviour reasoning trusts them, which
   suppresses one-frame detector blips.
4. **Cooldowns.** Per `(behaviour, track)` cooldowns ensure one continuous real-world action
   produces one incident, not a burst.
5. **Confidence propagation.** Weak detection confidence reduces the risk score and is shown as a
   named negative factor.
6. **Scene context is declared, never inferred from filenames.** Bay, shift, camera, floor condition
   and dock status are supplied at ingest.
