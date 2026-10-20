import pytest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.workflow import LangGraphMineSafetyEngine

@pytest.fixture(scope="module")
def engine():
    eng = LangGraphMineSafetyEngine()
    yield eng
    if hasattr(eng, "retriever") and hasattr(eng.retriever, "close"):
        eng.retriever.close()


def test_langgraph_engine_initialization(engine):
    assert engine.graph is not None


def test_emergency_action_gating(engine):
    res = engine.run_safety_query("emergency stop directive site alpha", human_approved=False)
    
    assert res.get("requires_human_approval") is True
    assert "HIGH-IMPACT EMERGENCY ACTION" in res["answer"] or "DIRECTIVE" in res["answer"]
