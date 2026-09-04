"""
Grounded AI Warehouse Operations Assistant.

Answers supervisor questions using only rows retrieved from the incident
database. The design goal is hallucination resistance, so the architecture is
retrieval-first:

    question -> intent + filters -> SQL retrieval -> answer rendered from rows

Every number, timestamp, behaviour and bay in an answer is copied from a
retrieved row. When retrieval returns nothing, the assistant says so instead of
composing a plausible-sounding answer. The response also carries the exact rows
used, so the dashboard can show the evidence behind the text.

An optional LLM can be enabled to rephrase the answer (see ``_maybe_polish``);
it is given the retrieved facts and is never the source of them. With no API key
configured the deterministic renderer is used, which is the default.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from backend.database.db import DatabaseManager

logger = logging.getLogger(__name__)

NO_DATA = (
    "The system does not have enough detected evidence to answer this. "
    "No matching events are recorded in the incident database."
)

BEHAVIOUR_SYNONYMS: Dict[str, List[str]] = {
    "product_drop": ["drop", "dropped", "dropping", "fell", "fall", "falling"],
    "product_drag": ["drag", "dragged", "dragging", "pulled along", "sliding"],
    "product_throw": ["throw", "thrown", "throwing", "toss", "tossed", "pitched", "push"],
    "rolling_product": ["roll", "rolled", "rolling", "tumble", "tumbling"],
    "improper_stacking": ["stack", "stacked", "stacking", "overhang", "on top"],
    "stepping_on_carton": ["step", "stepped", "stepping", "standing on", "stood on", "walk on"],
    "unsupported_handling": ["without equipment", "no trolley", "manual handling", "unsupported"],
    "wet_floor_hazard": ["wet floor", "wet", "moisture", "water"],
    "orientation_violation": ["orientation", "upright", "flat", "horizontal", "this side up"],
    "dock_level_hazard": ["dock", "leveller", "leveler", "bridge plate", "threshold"],
    "outside_designated_area": ["designated area", "outside zone", "staging area", "wrong place"],
    "unsafe_loading_sequence": ["sequence", "loading order", "concurrent", "congestion"],
}


def _pretty(behaviour: str) -> str:
    return behaviour.replace("_", " ").title()


class AIAssistant:
    """Retrieval-grounded conversational assistant for shift supervisors."""

    SYSTEM_ROLE = (
        "VisionGuard Assistant supports Godrej warehouse supervisors with damage "
        "prevention. It reports only what the computer-vision pipeline actually "
        "recorded, distinguishes observed behaviour from potential risk and from "
        "confirmed damage, and frames findings as process improvements rather than "
        "individual blame."
    )

    # ------------------------------------------------------------------ entry
    @classmethod
    def answer_query(cls, query: str, video_id: Optional[str] = None) -> Dict[str, Any]:
        q = (query or "").strip()
        ql = q.lower()

        intent = cls._classify_intent(ql)
        filters = cls._extract_filters(ql)

        incidents = DatabaseManager.get_incidents(
            video_id=video_id,
            risk_level=filters.get("risk_level"),
            behaviour_type=filters.get("behaviour_type"),
            bay=filters.get("bay"),
            limit=500,
        )
        summary = DatabaseManager.get_analytics_summary()

        handler = {
            "high_risk": cls._answer_high_risk,
            "why": cls._answer_why,
            "top_behaviours": cls._answer_top_behaviours,
            "location": cls._answer_location,
            "shift": cls._answer_shift,
            "behaviour": cls._answer_behaviour,
            "count": cls._answer_count,
            "action": cls._answer_action,
            "prevention": cls._answer_prevention,
            "overview": cls._answer_overview,
        }[intent]

        text, used = handler(ql, filters, incidents, summary)
        text = cls._maybe_polish(q, text)

        return {
            "query": q,
            "intent": intent,
            "filters": {k: v for k, v in filters.items() if v},
            "response": text,
            "grounded": True,
            "relevant_count": len(used),
            "relevant_incidents": used[:8],
            "total_incidents_in_db": summary["total_incidents"],
        }

    # --------------------------------------------------------------- routing
    @staticmethod
    def _classify_intent(ql: str) -> str:
        """
        Order matters: the most specific question shapes are tested first.

        The earlier implementation tested for the substring "risk" before
        "most common", so "the three most common risky behaviours" was answered
        with a high-risk event list. Specific phrasings are matched first here.
        """
        if re.search(r"\bwhy\b|classif|root cause|reason", ql):
            return "why"
        if re.search(r"most common|top \d|top three|top 3|frequent|which behaviour", ql):
            return "top_behaviours"
        if re.search(r"which (loading )?bay|which dock|which location|where |per bay|by bay", ql):
            return "location"
        if re.search(r"shift|morning|afternoon|night", ql):
            return "shift"
        if re.search(r"what (should|action|corrective)|recommend|corrective|fix|prevent this", ql):
            return "action"
        if re.search(r"prevent|training|improve|coach|recurring", ql):
            return "prevention"
        if re.search(r"how many|count|number of|total", ql):
            return "count"
        if re.search(r"high[- ]risk|critical|severe|urgent", ql):
            return "high_risk"
        for terms in BEHAVIOUR_SYNONYMS.values():
            if any(t in ql for t in terms):
                return "behaviour"
        if re.search(r"summar|overview|shift report|how are we", ql):
            return "overview"
        return "overview"

    @staticmethod
    def _extract_filters(ql: str) -> Dict[str, Optional[str]]:
        filters: Dict[str, Optional[str]] = {
            "risk_level": None,
            "behaviour_type": None,
            "bay": None,
        }
        if "critical" in ql:
            filters["risk_level"] = "CRITICAL"
        elif re.search(r"\bhigh[- ]risk\b|\bhigh risk\b", ql):
            filters["risk_level"] = "HIGH"
        elif re.search(r"\bmedium\b", ql):
            filters["risk_level"] = "MEDIUM"
        elif re.search(r"\blow risk\b", ql):
            filters["risk_level"] = "LOW"

        best: Optional[Tuple[str, int]] = None
        for behaviour, terms in BEHAVIOUR_SYNONYMS.items():
            for term in terms:
                if term in ql and (best is None or len(term) > best[1]):
                    best = (behaviour, len(term))
        if best:
            filters["behaviour_type"] = best[0]
        return filters

    # -------------------------------------------------------------- handlers
    @staticmethod
    def _fmt_incident(inc: Dict[str, Any], idx: Optional[int] = None) -> str:
        head = f"{idx}. " if idx else "- "
        return (
            f"**{head}[{inc['risk_level']}] {_pretty(inc['behaviour_type'])}** "
            f"at `{inc['timestamp_sec']:.2f}s` in {inc.get('bay') or 'an unassigned bay'} "
            f"(score {inc['risk_score']:.0f}/100)\n"
            f"   - Observed: {inc['evidence_description']}\n"
            f"   - Recommended: {inc['recommended_action']}\n"
        )

    @classmethod
    def _answer_high_risk(cls, ql, filters, incidents, summary):
        rows = [i for i in incidents if i["risk_level"] in ("HIGH", "CRITICAL")]
        if filters.get("risk_level"):
            rows = [i for i in incidents if i["risk_level"] == filters["risk_level"]]
        if not rows:
            return (
                "No high or critical risk handling events are recorded for the current selection. "
                f"The database holds {summary['total_incidents']} event(s) in total.",
                [],
            )
        lines = [
            f"### {len(rows)} elevated-risk handling event(s) recorded",
            "These are **potential** damage risks from observed handling behaviour. "
            "Damage is only confirmed by physical inspection.\n",
        ]
        for n, inc in enumerate(rows[:5], 1):
            lines.append(cls._fmt_incident(inc, n))
        if len(rows) > 5:
            lines.append(f"_{len(rows) - 5} further event(s) available in the incident timeline._")
        return "\n".join(lines), rows

    @classmethod
    def _answer_why(cls, ql, filters, incidents, summary):
        if not incidents:
            return NO_DATA, []
        inc = max(incidents, key=lambda i: i["risk_score"])
        lines = [
            f"### Why event `{inc['id']}` is rated {inc['risk_level']}",
            f"- **Behaviour:** {_pretty(inc['behaviour_type'])}",
            f"- **Score:** {inc['risk_score']:.0f}/100 "
            f"(LOW <42, MEDIUM 42-63, HIGH 64-81, CRITICAL >=82)",
            f"- **Observed at:** {inc['timestamp_sec']:.2f}s, {inc.get('bay') or 'unassigned bay'}",
            f"- **What was seen:** {inc['evidence_description']}",
            f"- **Operational root cause:** {inc['root_cause']}",
        ]
        factors = inc.get("risk_factors") or []
        if factors:
            lines.append("\n**Score breakdown - every point is attributable:**")
            for f in factors:
                lines.append(f"- `{f['points']:+.0f}` {f['name']} - {f['detail']}")
        stages = inc.get("evidence_stages") or []
        if stages:
            chain = " -> ".join(f"{s['stage']}@{s['at_sec']}s" for s in stages)
            lines.append(f"\n**Observed motion sequence:** {chain}")
        lines.append(f"\n**Recommended action:** {inc['recommended_action']}")
        lines.append(
            f"\n_Evidence tier: {inc.get('evidence_tier', 'OBSERVED_BEHAVIOUR')}. "
            f"Review status: {inc.get('review_status', 'PENDING_REVIEW')}._"
        )
        return "\n".join(lines), [inc]

    @classmethod
    def _answer_top_behaviours(cls, ql, filters, incidents, summary):
        top = summary.get("top_behaviours") or {}
        if not top:
            return NO_DATA, []
        want = 3
        m = re.search(r"top (\d+)|(\d+) most", ql)
        if m:
            want = int(m.group(1) or m.group(2))
        elif "three" in ql:
            want = 3
        want = max(1, min(want, len(top)))

        total = max(1, summary["total_incidents"])
        lines = [f"### {want} most frequently detected risky behaviour(s)"]
        for n, (b, c) in enumerate(list(top.items())[:want], 1):
            lines.append(f"{n}. **{_pretty(b)}** - {c} event(s) ({c / total * 100:.1f}% of all detections)")
        lines.append(
            f"\nBased on {summary['total_incidents']} recorded event(s) across "
            f"{summary['total_videos_analyzed']} analysed video(s) "
            f"({summary['total_footage_minutes']:.1f} minutes of footage)."
        )
        used = [i for i in incidents if i["behaviour_type"] in list(top)[:want]]
        return "\n".join(lines), used

    @classmethod
    def _answer_location(cls, ql, filters, incidents, summary):
        by_bay = [b for b in summary.get("by_bay", []) if b["total"] > 0]
        if not by_bay:
            return NO_DATA, []
        if len(by_bay) == 1 and by_bay[0]["bay"] == "Unassigned Bay":
            return (
                "All recorded events are attributed to 'Unassigned Bay', so a per-bay "
                "comparison is not meaningful yet. Set the bay on the ingest form (or "
                "per camera) and re-analyse to enable location analytics.",
                [],
            )
        top = by_bay[0]
        lines = [
            f"### Risk events by loading bay",
            f"**{top['bay']}** has the highest count: {top['total']} event(s), "
            f"of which {top['high_risk']} are HIGH or CRITICAL.\n",
        ]
        for b in by_bay[:8]:
            lines.append(f"- **{b['bay']}** - {b['total']} event(s), {b['high_risk']} elevated risk")
        used = [i for i in incidents if i.get("bay") == top["bay"]]
        return "\n".join(lines), used

    @classmethod
    def _answer_shift(cls, ql, filters, incidents, summary):
        by_shift = [s for s in summary.get("by_shift", []) if s["total"] > 0]
        if not by_shift:
            return NO_DATA, []
        lines = ["### Risk events by shift"]
        for s in by_shift:
            lines.append(f"- **{s['shift']}** - {s['total']} event(s), {s['high_risk']} elevated risk")
        lines.append(
            f"\nAcross all shifts: {summary['intervention_opportunities']} intervention "
            f"opportunit(ies) identified in {summary['total_footage_minutes']:.1f} minutes of footage "
            f"({summary['high_risk_events_per_minute']:.2f} elevated-risk events per minute)."
        )
        return "\n".join(lines), incidents[:8]

    @classmethod
    def _answer_behaviour(cls, ql, filters, incidents, summary):
        b = filters.get("behaviour_type")
        if not b:
            return cls._answer_overview(ql, filters, incidents, summary)
        rows = [i for i in incidents if i["behaviour_type"] == b]
        if not rows:
            return (
                f"No **{_pretty(b)}** events are recorded for the current selection. "
                "The system reports only what it actually detected.",
                [],
            )
        lines = [f"### {len(rows)} {_pretty(b)} event(s) recorded\n"]
        for inc in rows[:5]:
            lines.append(cls._fmt_incident(inc))
        if len(rows) > 5:
            lines.append(f"_{len(rows) - 5} further event(s) in the timeline._")
        return "\n".join(lines), rows

    @classmethod
    def _answer_count(cls, ql, filters, incidents, summary):
        b = filters.get("behaviour_type")
        r = filters.get("risk_level")
        rows = incidents
        if b:
            rows = [i for i in rows if i["behaviour_type"] == b]
        if r:
            rows = [i for i in rows if i["risk_level"] == r]
        label = " ".join(x for x in [r, _pretty(b) if b else None] if x) or "handling"
        if not rows:
            return (
                f"Zero **{label}** events are recorded. "
                f"The database holds {summary['total_incidents']} event(s) in total.",
                [],
            )
        by_bay = Counter(i.get("bay") or "Unassigned Bay" for i in rows)
        lines = [
            f"**{len(rows)}** {label} event(s) are recorded.",
            "",
            "Distribution by bay:",
        ]
        for bay, c in by_bay.most_common():
            lines.append(f"- {bay}: {c}")
        first, last = rows[0], rows[-1]
        lines.append(
            f"\nFirst at `{first['timestamp_sec']:.2f}s`, most recent at `{last['timestamp_sec']:.2f}s` "
            "within their source recordings."
        )
        return "\n".join(lines), rows

    @classmethod
    def _answer_action(cls, ql, filters, incidents, summary):
        rows = incidents
        if filters.get("behaviour_type"):
            rows = [i for i in rows if i["behaviour_type"] == filters["behaviour_type"]]
        if not rows:
            return NO_DATA, []
        top = sorted(rows, key=lambda i: i["risk_score"], reverse=True)[:3]
        lines = ["### Recommended corrective actions (highest risk first)\n"]
        for n, inc in enumerate(top, 1):
            lines.append(
                f"{n}. **{_pretty(inc['behaviour_type'])}** at `{inc['timestamp_sec']:.2f}s` "
                f"({inc['risk_level']}, {inc.get('bay') or 'unassigned bay'})\n"
                f"   - {inc['recommended_action']}\n"
            )
        lines.append(
            "_These are prevention actions on the handling process. "
            "No damage is asserted; each item should be physically inspected before dispatch._"
        )
        return "\n".join(lines), top

    @classmethod
    def _answer_prevention(cls, ql, filters, incidents, summary):
        top = summary.get("top_behaviours") or {}
        if not top:
            return NO_DATA, []
        training = {
            "product_drop": "controlled lowering and team lifting",
            "product_drag": "trolley and pallet-truck usage",
            "product_throw": "lift-and-place discipline at transfer points",
            "rolling_product": "why rolling damages corners, and the alternatives",
            "improper_stacking": "stacking order and base support",
            "stepping_on_carton": "walkway discipline and height access equipment",
            "unsupported_handling": "manual handling limits and equipment provisioning",
            "wet_floor_hazard": "floor condition reporting and stop-work authority",
            "orientation_violation": "handling arrows and orientation labels",
            "dock_level_hazard": "dock leveller procedure",
            "outside_designated_area": "staging zone discipline",
            "unsafe_loading_sequence": "planned one-at-a-time loading sequence",
        }
        lines = ["### Prevention priorities from the recorded evidence\n"]
        for b, c in list(top.items())[:4]:
            lines.append(
                f"- **{_pretty(b)}** ({c} event(s)) -> coach on {training.get(b, 'the site handling SOP')}"
            )
        lines.append(
            f"\nBaseline for improvement tracking: "
            f"**{summary['high_risk_events_per_minute']:.2f}** elevated-risk events per minute of "
            f"footage across {summary['total_footage_minutes']:.1f} minutes analysed. "
            "Compare this rate after the coaching intervention to measure whether behaviour improved."
        )
        return "\n".join(lines), incidents[:8]

    @classmethod
    def _answer_overview(cls, ql, filters, incidents, summary):
        if summary["total_incidents"] == 0:
            return (
                "No handling events have been recorded yet. Ingest a warehouse video "
                "from the **Ingest** tab and the analysis results will appear here.",
                [],
            )
        rb = summary["risk_breakdown"]
        lines = [
            "### Shift overview",
            f"- **Footage analysed:** {summary['total_videos_analyzed']} video(s), "
            f"{summary['total_footage_minutes']:.1f} minutes",
            f"- **Handling events detected:** {summary['total_incidents']}",
            f"- **Critical:** {rb['CRITICAL']} | **High:** {rb['HIGH']} | "
            f"**Medium:** {rb['MEDIUM']} | **Low:** {rb['LOW']}",
            f"- **Intervention opportunities:** {summary['intervention_opportunities']} "
            f"({summary['high_risk_events_per_minute']:.2f} per minute of footage)",
        ]
        top = list((summary.get("top_behaviours") or {}).items())[:3]
        if top:
            lines.append("\n**Most frequent behaviours:**")
            for b, c in top:
                lines.append(f"- {_pretty(b)}: {c}")
        lines.append(
            "\nYou can ask: *'Which loading bay had the most risky events?'*, "
            "*'Why was this event classified as high risk?'*, "
            "*'What were the three most common risky behaviours?'*, "
            "*'What corrective action is recommended?'*"
        )
        return "\n".join(lines), incidents[:8]

    # ------------------------------------------------------------ optional LLM
    @staticmethod
    def _maybe_polish(question: str, grounded_answer: str) -> str:
        """
        Optionally rephrase the grounded answer with an LLM.

        The model receives the already-retrieved answer as the only permitted
        source, so it can reword but cannot introduce facts. Any failure returns
        the deterministic answer unchanged. Disabled unless an API key is set.
        """
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key or os.environ.get("ASSISTANT_LLM_POLISH", "false").lower() != "true":
            return grounded_answer
        try:  # pragma: no cover - optional path, off by default
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = (
                f"{AIAssistant.SYSTEM_ROLE}\n\n"
                "Rewrite the ANSWER below more fluently for a warehouse supervisor. "
                "Do not add, remove or alter any number, timestamp, bay name or behaviour. "
                "If a fact is not in the ANSWER, it must not appear in your output.\n\n"
                f"QUESTION: {question}\n\nANSWER:\n{grounded_answer}"
            )
            out = model.generate_content(prompt).text
            return out.strip() or grounded_answer
        except Exception:  # noqa: BLE001
            logger.warning("LLM polish unavailable; returning grounded answer", exc_info=True)
            return grounded_answer
