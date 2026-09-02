"""
Grounded AI Warehouse Assistant
Answers supervisor inquiries using structured event data from SQLite.
Adheres strictly to factual grounding: NEVER invents incidents, statistics, or fake detections.
"""

from typing import Dict, Any, List, Optional
import re
from backend.database.db import DatabaseManager

class AIAssistant:
    """
    Conversational AI operational assistant for warehouse shift supervisors.
    """

    SYSTEM_ROLE = """You are VisionGuard Assistant, an AI Field Intelligence Assistant for Godrej Warehouse Operations.
Your primary mandate is proactive damage prevention and safe material handling.
You analyze detected computer vision events (dropping, dragging, improper stacking, rough handling, etc.).
RULES:
1. Always base your answers ONLY on the actual detected incidents provided in the context.
2. If the user asks about a specific incident, explain the root cause, risk level, and recommended corrective action.
3. If no matching incidents are found in the database, explicitly state that no such events were recorded.
4. Maintain a professional, safety-oriented, and constructive tone.
5. Emphasize operational improvement, damage prevention, and operator coaching, not punitive discipline.
"""

    @classmethod
    def answer_query(cls, query: str, video_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process user question with intent classification, database retrieval, and grounded response.
        """
        query_lower = query.lower()
        
        # 1. Retrieve all incidents and summary stats
        incidents = DatabaseManager.get_incidents(video_id=video_id, limit=200)
        summary = DatabaseManager.get_analytics_summary()
        
        # 2. Extract query filters
        target_risk = None
        for risk in ["critical", "high", "medium", "low"]:
            if risk in query_lower:
                target_risk = risk.upper()
                break

        target_behaviour = None
        behaviour_synonyms = {
            "drop": "product_drop",
            "fall": "product_drop",
            "drag": "product_drag",
            "pull": "product_drag",
            "throw": "product_throw",
            "toss": "product_throw",
            "roll": "rolling_product",
            "tumble": "rolling_product",
            "stack": "improper_stacking",
            "step": "stepping_on_carton",
            "stand": "stepping_on_carton",
            "walk": "stepping_on_carton",
            "strap": "strap_pulling",
            "wet": "wet_floor_hazard",
            "water": "wet_floor_hazard",
            "orient": "orientation_violation",
            "vertical": "orientation_violation",
            "horizontal": "orientation_violation",
            "dock": "dock_level_hazard"
        }
        for keyword, b_type in behaviour_synonyms.items():
            if keyword in query_lower:
                target_behaviour = b_type
                break

        # Filter relevant incidents
        relevant_incidents = incidents
        if target_risk:
            relevant_incidents = [i for i in relevant_incidents if i["risk_level"] == target_risk]
        if target_behaviour:
            relevant_incidents = [i for i in relevant_incidents if i["behaviour_type"] == target_behaviour]

        # 3. Generate grounded deterministic answer
        response_text = cls._generate_grounded_text(
            query=query,
            query_lower=query_lower,
            target_risk=target_risk,
            target_behaviour=target_behaviour,
            relevant_incidents=relevant_incidents,
            all_incidents=incidents,
            summary=summary
        )

        return {
            "query": query,
            "response": response_text,
            "relevant_count": len(relevant_incidents),
            "relevant_incidents": relevant_incidents[:8],
            "total_incidents_in_db": summary["total_incidents"]
        }

    @classmethod
    def _generate_grounded_text(
        cls,
        query: str,
        query_lower: str,
        target_risk: Optional[str],
        target_behaviour: Optional[str],
        relevant_incidents: List[Dict[str, Any]],
        all_incidents: List[Dict[str, Any]],
        summary: Dict[str, Any]
    ) -> str:
        # Scenario A: Inquiries about why an event was classified or root causes
        if "why" in query_lower or "classify" in query_lower or "cause" in query_lower or "reason" in query_lower:
            if relevant_incidents:
                inc = relevant_incidents[0]
            elif all_incidents:
                inc = all_incidents[0]
            else:
                return "No incidents are currently available for causal analysis."

            return (
                f"### Incident Risk Classification Analysis (ID: {inc['id']})\n"
                f"- **Behaviour:** {inc['behaviour_type'].replace('_', ' ').title()}\n"
                f"- **Risk Classification:** `{inc['risk_level']}` (Score: {inc['risk_score']}/100)\n"
                f"- **Physical Factors:** {inc['evidence_description']}\n"
                f"- **Operational Root Cause:** {inc['root_cause']}\n"
                f"- **Prescriptive Action:** {inc['recommended_action']}\n\n"
                f"*Classification Methodology:* The risk engine evaluates impact deceleration, drop height, duration of floor dragging, and stacking geometry to assign non-arbitrary risk tiers."
            )

        # Scenario B: Inquiries about high-risk or critical events
        if "high" in query_lower or "critical" in query_lower or ("risk" in query_lower and "score" not in query_lower):
            high_and_crit = [i for i in all_incidents if i["risk_level"] in ["HIGH", "CRITICAL"]]
            if not high_and_crit:
                return "Good news! No high or critical risk handling events were detected in the recorded sessions."
            
            lines = [
                f"### High-Risk Handling Analysis ({len(high_and_crit)} Events Identified)",
                f"VisionGuard detected **{len(high_and_crit)}** elevated risk events that require supervisor intervention to prevent product damage:\n"
            ]
            for idx, inc in enumerate(high_and_crit[:4], 1):
                lines.append(
                    f"**{idx}. [{inc['risk_level']}] {inc['behaviour_type'].replace('_', ' ').title()}** at `{inc['timestamp_sec']}s` (Risk Score: {inc['risk_score']}/100)\n"
                    f"- **Observed Finding:** {inc['evidence_description']}\n"
                    f"- **Recommended Intervention:** {inc['recommended_action']}\n"
                )
            lines.append("💡 *Preventive Recommendation:* Conduct a 5-minute pre-shift briefing emphasizing safe mechanical handling and proper pallet stacking.")
            return "\n".join(lines)

        # Scenario B: Inquiries about most common / top behaviours
        if "most common" in query_lower or "top" in query_lower or "frequent" in query_lower or "summary" in query_lower:
            top = summary.get("top_behaviours", {})
            if not top:
                return "There are currently no detected incidents logged in the system."
            
            lines = [
                "### Shift Behaviour Intelligence Summary",
                f"- **Total Videos Analyzed:** {summary['total_videos_analyzed']}",
                f"- **Total Incidents Identified:** {summary['total_incidents']}",
                f"- **Warehouse Discipline Index:** **{summary['handling_discipline_score']}/100**",
                f"- **Proactive Prevention:** {summary['damage_prevention_potential']}\n",
                "**Top Detected Handling Behaviours:**"
            ]
            for b_name, count in list(top.items())[:5]:
                pct = (count / max(1, summary['total_incidents'])) * 100
                lines.append(f"- **{b_name.replace('_', ' ').title()}:** {count} occurrences ({pct:.1f}%)")
            return "\n".join(lines)

        # Scenario C: Inquiries about specific behaviour (e.g. drop, drag, stack)
        if target_behaviour:
            b_name = target_behaviour.replace('_', ' ').title()
            if not relevant_incidents:
                return f"No **{b_name}** events were observed in the analyzed video footage."
            
            lines = [
                f"### Detected {b_name} Events ({len(relevant_incidents)} Logged)",
                f"VisionGuard detected **{len(relevant_incidents)}** instances of {b_name}:\n"
            ]
            for inc in relevant_incidents[:5]:
                lines.append(
                    f"- **Timestamp {inc['timestamp_sec']:.2f}s** | Risk: `{inc['risk_level']}` | Object #{inc.get('object_track_id', 'N/A')}\n"
                    f"  - *Finding:* {inc['evidence_description']}\n"
                    f"  - *Corrective Action:* {inc['recommended_action']}\n"
                )
            return "\n".join(lines)

        # Scenario D: Inquiries about why an event was classified as high risk
        if "why" in query_lower or "classify" in query_lower or "cause" in query_lower:
            if relevant_incidents:
                inc = relevant_incidents[0]
            elif all_incidents:
                inc = all_incidents[0]
            else:
                return "No incidents are currently available for causal analysis."

            return (
                f"### Incident Risk Classification Analysis (ID: {inc['id']})\n"
                f"- **Behaviour:** {inc['behaviour_type'].replace('_', ' ').title()}\n"
                f"- **Risk Classification:** `{inc['risk_level']}` (Score: {inc['risk_score']}/100)\n"
                f"- **Physical Factors:** {inc['evidence_description']}\n"
                f"- **Operational Root Cause:** {inc['root_cause']}\n"
                f"- **Prescriptive Action:** {inc['recommended_action']}\n\n"
                f"*Classification Methodology:* The risk engine evaluates impact deceleration, drop height, duration of floor dragging, and stacking geometry to assign non-arbitrary risk tiers."
            )

        # Scenario E: Default general summary grounded in DB
        return (
            f"VisionGuard has analyzed **{summary['total_videos_analyzed']} warehouse sessions** with a total of "
            f"**{summary['total_incidents']} handling incidents** logged.\n\n"
            f"- **Critical Risk:** {summary['risk_breakdown']['CRITICAL']}\n"
            f"- **High Risk:** {summary['risk_breakdown']['HIGH']}\n"
            f"- **Medium Risk:** {summary['risk_breakdown']['MEDIUM']}\n"
            f"- **Low Risk:** {summary['risk_breakdown']['LOW']}\n\n"
            f"You can ask me specific questions such as:\n"
            f"- *'Show all high-risk handling events'* \n"
            f"- *'What were the most common risky behaviours?'* \n"
            f"- *'How many product drops were detected?'* \n"
            f"- *'Why was this event classified as high risk?'*"
        )
