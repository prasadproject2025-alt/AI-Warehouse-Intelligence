"""
Assistant grounding tests.

The assistant must answer from retrieved rows only. These tests check that it
routes questions correctly, reports real numbers, and — most importantly —
refuses to invent anything when the database has no matching evidence.
"""

import re

from assistant.llm import NO_DATA, AIAssistant
from helpers import seed_incident, seed_video


def _numbers(text):
    return set(re.findall(r"\b\d+\b", text))


# ------------------------------------------------------- no-data behaviour
def test_empty_database_produces_an_explicit_no_data_answer():
    res = AIAssistant.answer_query("What were today's high-risk events?")
    assert res["relevant_count"] == 0
    assert res["total_incidents_in_db"] == 0
    low = res["response"].lower()
    assert "no " in low or "not have enough" in low


def test_does_not_invent_events_for_a_behaviour_that_never_happened():
    seed_video()
    seed_incident(behaviour_type="product_drag")
    res = AIAssistant.answer_query("How many product drops were detected?")
    assert res["relevant_count"] == 0
    assert "zero" in res["response"].lower() or "no " in res["response"].lower()
    # It must not quote a drop count it does not have.
    assert "product_drop" not in res["response"] or "0" in res["response"]


def test_unanswerable_location_question_says_so():
    seed_video(bay="Unassigned Bay")
    seed_incident(bay="Unassigned Bay")
    res = AIAssistant.answer_query("Which loading bay had the highest number of risky events?")
    assert "unassigned" in res["response"].lower()
    assert "not meaningful" in res["response"].lower() or "set the bay" in res["response"].lower()


# ------------------------------------------------------------------ routing
def test_most_common_question_is_not_answered_as_a_high_risk_list():
    """
    Regression: 'most common risky behaviours' contains the token 'risk', which
    previously routed the question to the high-risk event list instead of the
    frequency ranking.
    """
    seed_video()
    for i in range(3):
        seed_incident(id=f"inc_drag{i}", behaviour_type="product_drag", risk_level="MEDIUM")
    seed_incident(id="inc_drop1", behaviour_type="product_drop", risk_level="HIGH")

    res = AIAssistant.answer_query("What were the three most common risky behaviours?")
    assert res["intent"] == "top_behaviours"
    assert "Product Drag" in res["response"]
    assert "3 event" in res["response"]


def test_why_question_returns_the_score_breakdown():
    seed_video()
    seed_incident()
    res = AIAssistant.answer_query("Why was this event classified as high risk?")
    assert res["intent"] == "why"
    assert "Drop height" in res["response"]
    assert "+18" in res["response"] or "18" in res["response"]
    assert "carried" in res["response"]  # temporal sequence is shown


def test_location_question_ranks_real_bays():
    seed_video(video_id="v1", bay="Dock 09")
    seed_video(video_id="v2", bay="Dock 03")
    seed_incident(id="i1", video_id="v1", bay="Dock 09", risk_level="CRITICAL")
    seed_incident(id="i2", video_id="v1", bay="Dock 09", risk_level="HIGH")
    seed_incident(id="i3", video_id="v2", bay="Dock 03", risk_level="LOW")

    res = AIAssistant.answer_query("Which loading bay had the highest number of risky events?")
    assert res["intent"] == "location"
    assert "Dock 09" in res["response"]
    assert res["response"].index("Dock 09") < res["response"].index("Dock 03")


def test_action_question_returns_stored_recommendations():
    seed_video()
    seed_incident()
    res = AIAssistant.answer_query("What corrective action is recommended?")
    assert res["intent"] == "action"
    assert "Inspect the package" in res["response"]


def test_behaviour_specific_question_lists_only_that_behaviour():
    seed_video()
    seed_incident(id="i1", behaviour_type="product_drop")
    seed_incident(id="i2", behaviour_type="product_drag")
    res = AIAssistant.answer_query("Show me the dragging events")
    assert res["filters"]["behaviour_type"] == "product_drag"
    assert all(i["behaviour_type"] == "product_drag" for i in res["relevant_incidents"])


def test_shift_question_reports_by_shift():
    seed_video(shift="Shift B")
    seed_incident(shift="Shift B")
    res = AIAssistant.answer_query("How did the night shift perform?")
    assert res["intent"] == "shift"
    assert "Shift B" in res["response"]


# ------------------------------------------------- factual grounding checks
def test_every_number_in_a_count_answer_comes_from_the_database():
    seed_video()
    for i in range(4):
        seed_incident(id=f"inc_{i}", behaviour_type="product_drag")
    res = AIAssistant.answer_query("How many dragging events were detected?")
    assert "4" in res["response"]
    assert res["relevant_count"] == 4


def test_answers_cite_the_rows_they_used():
    seed_video()
    seed_incident()
    res = AIAssistant.answer_query("Show me all high-risk handling events")
    assert res["relevant_count"] >= 1
    assert res["relevant_incidents"][0]["id"] == "inc_test01"


def test_timestamps_are_never_fabricated():
    seed_video()
    seed_incident(timestamp_sec=12.34)
    res = AIAssistant.answer_query("Show me all high-risk handling events")
    assert "12.34" in res["response"]
    # No other timestamp-shaped value should appear.
    stamps = set(re.findall(r"\d+\.\d\ds", res["response"]))
    assert stamps <= {"12.34s"}


def test_never_claims_confirmed_damage_from_video_alone():
    seed_video()
    seed_incident(risk_level="CRITICAL", risk_score=95.0)
    for question in ["Show me all high-risk handling events",
                     "What corrective action is recommended?",
                     "Summarise the shift"]:
        text = AIAssistant.answer_query(question)["response"].lower()
        assert "products were damaged" not in text
        assert "product damaged" not in text


def test_scoping_to_a_video_limits_retrieval():
    seed_video(video_id="v1")
    seed_video(video_id="v2")
    seed_incident(id="i1", video_id="v1")
    seed_incident(id="i2", video_id="v2")
    res = AIAssistant.answer_query("Show me all high-risk handling events", video_id="v1")
    assert {i["id"] for i in res["relevant_incidents"]} == {"i1"}


# --------------------------------------- untracked subjects must be refused
def test_refuses_to_count_a_behaviour_it_does_not_track():
    """
    Regression: 'how many forklift collision events' previously fell through to
    the unfiltered total, implying the system had detected 13 forklift
    collisions. It must say it does not track that instead.
    """
    seed_video()
    for i in range(4):
        seed_incident(id=f"inc_{i}", behaviour_type="product_drag")

    res = AIAssistant.answer_query("How many forklift collision events were detected in Bay 99?")
    assert res["relevant_count"] == 0
    assert "does not detect" in res["response"]
    assert "forklift collision" in res["response"]
    assert "**4**" not in res["response"], "must not report an unrelated count"


def test_lists_what_it_does_track_when_refusing():
    seed_video()
    seed_incident()
    res = AIAssistant.answer_query("How many PPE violations were there?")
    assert "does not detect" in res["response"]
    assert "Product Drop" in res["response"] and "Product Drag" in res["response"]


def test_generic_count_questions_still_return_the_total():
    """'How many events in total' is a legitimate request for the overall count."""
    seed_video()
    for i in range(5):
        seed_incident(id=f"inc_{i}")
    for q in ["How many events were detected in total?", "How many incidents are there?"]:
        res = AIAssistant.answer_query(q)
        assert res["relevant_count"] == 5, q
        assert "5" in res["response"]


def test_known_behaviour_with_zero_rows_reports_zero_not_a_refusal():
    seed_video()
    seed_incident(behaviour_type="product_drag")
    res = AIAssistant.answer_query("How many product drops were detected?")
    assert "zero" in res["response"].lower()
    assert "does not detect" not in res["response"]


def test_bay_filter_only_triggers_on_a_real_bay_identifier():
    """
    Regression: 'which loading bay had the most events' matched the word after
    'bay' and filtered on the non-existent bay "Bay Had", silently returning
    zero rows for a question about every bay.
    """
    seed_video(bay="Dock 09")
    seed_incident(id="i1", bay="Dock 09", risk_level="HIGH")
    seed_incident(id="i2", bay="Dock 09", risk_level="CRITICAL")

    res = AIAssistant.answer_query("Which loading bay had the highest number of risky events?")
    assert res["filters"].get("bay") is None
    assert res["relevant_count"] > 0
    assert "Dock 09" in res["response"]


def test_bay_filter_applies_when_a_bay_number_is_named():
    seed_video(video_id="v1", bay="Dock 09")
    seed_video(video_id="v2", bay="Dock 03")
    seed_incident(id="i1", video_id="v1", bay="Dock 09")
    seed_incident(id="i2", video_id="v2", bay="Dock 03")
    res = AIAssistant.answer_query("How many dragging events in dock 09?")
    assert res["filters"].get("bay") == "Dock 09"
