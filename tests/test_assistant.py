import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from assistant.llm import AIAssistant
from backend.database.db import DatabaseManager

def test_assistant():
    # Test high-risk inquiry
    res1 = AIAssistant.answer_query("Show me all high risk events from today's unloading")
    print("RES 1 (High Risk Query):")
    print(res1["response"][:200])
    assert res1["relevant_count"] >= 1
    
    # Test specific behaviour query
    res2 = AIAssistant.answer_query("How many product drops were detected?")
    print("\nRES 2 (Drop Query):")
    print(res2["response"][:200])
    assert "drop" in res2["response"].lower()

    # Test why classified query
    res3 = AIAssistant.answer_query("Why was this event classified as high risk?")
    print("\nRES 3 (Why Query):")
    print(res3["response"][:200])
    assert "risk" in res3["response"].lower()
    
    print("\nALL ASSISTANT TESTS PASSED!")

if __name__ == "__main__":
    test_assistant()
