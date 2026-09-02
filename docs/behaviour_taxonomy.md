# VisionGuard — Behaviour Taxonomy & Operational Parameters

This document defines the 10 core warehouse material-handling behaviors implemented in VisionGuard, mapping directly to the Godrej Enterprises Group (GEG) Challenge Parameters.

---

| # | Behaviour / Bad Practice | Expected Good Handling Practice | Detection Heuristic & Kinematics | Risk Range | Actionable Supervisor Intervention |
|---|--------------------------|---------------------------------|-----------------------------------|------------|-----------------------------------|
| **1** | **Product Dropping** | Lift and place products gently. Never drop or throw cartons. | Rapid downward vertical velocity ($v_y > 120\text{ px/s}$), vertical drop $\ge 40\text{px}$, floor impact deceleration, followed by stationary resting state. | **HIGH to CRITICAL** (80–98) | Halt handling. Inspect product internally and externally. Review manual lowering technique with worker. |
| **2** | **Product Dragging** | Use trolley, pallet truck, or mechanical equipment. Do not drag on floor. | Object bottom within lower 60% of vertical scene, sustained horizontal motion ($v_x > 35\text{ px/s}$, $v_y \approx 0$), operator adjacent. | **MEDIUM to HIGH** (45–65) | Mandate immediate deployment of hydraulic pallet truck or trolley. Prevent bottom abrasion. |
| **3** | **Product Throwing / Pushing** | Two-handed controlled placement. Never pitch or throw goods. | High release velocity ($> 140\text{ px/s}$), spatial detachment from operator hands, ballistic trajectory into vehicle/bay. | **HIGH to CRITICAL** (80–95) | Intervene immediately. Stop throwing mattresses/cartons. Enforce sequential one-by-one staging. |
| **4** | **Rolling Products / Mattresses** | Carry or transport using handling equipment; do not roll or tumble. | Cyclical aspect ratio inversion ($w/h > 1.15 \leftrightarrow w/h < 0.85$) while translating along floor plane. | **MEDIUM to HIGH** (60–75) | Deploy hand truck or team lift. Rotational tumbling crushes corners and compromises structural integrity. |
| **5** | **Improper Stacking / Inversion** | Stack heavier/larger items at base; light/small packets on top. | Vertical containment where upper object width exceeds bottom object width by $> 25\%$, or heavy furniture placed over carton. | **HIGH to CRITICAL** (70–90) | Restack staging area immediately. Ensure full perimeter support for lighter upper cartons. |
| **6** | **Stepping / Standing on Cartons** | Keep clear walkway around material. Never step, stand, or walk on packages. | Operator lower bounding box (feet/legs) overlapping carton top half ($x_{\text{overlap}} > 30\text{px}, y_{\text{overlap}} > 15\text{px}$). | **CRITICAL** (85–95) | Severe safety and quality violation. Strictly prohibit standing on cartons. Clear designated walkways. |
| **7** | **Using Packaging Straps as Handles** | Handle cartons using base lifting points or mechanical devices. | Operator grip point concentrated strictly on upper carton edge with horizontal pulling motion without base support. | **MEDIUM to HIGH** (65–75) | Instruct worker that straps are for sealing only. Use carton bottom handholds or team lifting. |
| **8** | **Handling / Dragging on Wet Floor** | Keep loading area dry and clean. Do not move goods over wet floors. | Material dragging or sliding in moisture hazard zone or dock areas with high specular ground reflections. | **HIGH to CRITICAL** (75–95) | Stop movement immediately. Divert traffic, dry dock surface, and inspect for carton water absorption. |
| **9** | **Vertical Product Kept Flat** | Follow orientation markings ("This Side Up") throughout storage. | Product classified as vertical unit (cupboard, tall appliance) positioned with width exceeding height ($w/h > 1.25$). | **MEDIUM to HIGH** (55–75) | Restore upright orientation immediately. Prevents hinge strain, internal shelf collapse, or panel bowing. |
| **10**| **Dock Level / Transition Hazard** | Use proper dock leveller bridge; ensure level transition. | Heavy item traversed across vehicle bed and warehouse dock threshold without mechanical ramp or trolley. | **HIGH to CRITICAL** (80–90) | Deploy dock leveller plate before continuing transfer across vehicle-dock transition gap. |

---

## Responsible AI & Human-Centric Principles

1. **Process Improvement Over Surveillance**:
   VisionGuard is calibrated as an operational assistant for supervisors, not a tool for employee punitive discipline.
2. **Tri-Tier Distinction**:
   $$\text{Observed Behaviour} \longrightarrow \text{Potential Risk} \longrightarrow \text{Confirmed Damage}$$
   The system never claims that an item was damaged unless physical inspection confirms it.
3. **Actionable Coaching**:
   Every alert delivers an immediate operational recommendation and root-cause explanation to facilitate on-the-spot operator coaching.
