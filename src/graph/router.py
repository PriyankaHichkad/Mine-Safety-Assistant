import re
from typing import Dict, Any, Literal

class MineSafetyRouter:
    """
    Layer 1 Model & Intent Router Node.
    Routes queries to:
    - Tier 1 (Factual Lookup): Direct low-latency lookup / cached answer.
    - Tier 2 (Complex Hazard Scenario): Full hybrid retrieval + LangGraph workflow.
    - Tier 3 (High-Impact Emergency Directive): Gated tool execution requiring human approval.
    """
    def __init__(self):
        self.emergency_keywords = ["emergency stop", "halt mine", "shutdown bench", "evacuate seam", "danger stop"]
        self.factual_keywords = ["what is", "define", "regulation number", "clause", "minimum depth", "berm height", "loto code"]

    def classify_intent(self, query: str) -> Dict[str, Any]:
        q_lower = query.lower().strip()
        
        # Tier 3: Emergency Directive / High-Impact Action
        if any(kw in q_lower for kw in self.emergency_keywords):
            return {
                "tier": "Tier 3",
                "intent": "emergency_directive",
                "requires_human_approval": True,
                "target_tool": "issue_emergency_stop_directive"
            }

        # Tier 1: Quick Factual / Regulation Lookup
        if any(kw in q_lower for kw in self.factual_keywords) and len(q_lower.split()) <= 10:
            return {
                "tier": "Tier 1",
                "intent": "factual_lookup",
                "requires_human_approval": False,
                "target_tool": "lookup_mine_regulation"
            }

        # Tier 2: Complex Scenario Analysis
        return {
            "tier": "Tier 2",
            "intent": "complex_hazard_scenario",
            "requires_human_approval": False,
            "target_tool": None
        }
