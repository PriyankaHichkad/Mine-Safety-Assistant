import re
from typing import Dict, Any, List

class SafetyGuardrails:
    """
    Layer 3 Real-Time Input and Output Guardrails.
    - Input Guardrail: Detects prompt injection, jailbreaks, or malicious instruction overrides.
    - Output Guardrail: Verifies that factual answers contain valid grounded citation tags.
    """
    def __init__(self):
        self.injection_patterns = [
            r"ignore (all )?previous instructions",
            r"system prompt",
            r"you are now (an? )?unrestricted",
            r"override safety",
            r"jailbreak",
            r"bypass guardrails"
        ]

    def check_input(self, query: str) -> Dict[str, Any]:
        """Input Guardrail: Prompt Injection Defense."""
        q_lower = query.lower()
        for pattern in self.injection_patterns:
            if re.search(pattern, q_lower):
                return {
                    "passed": False,
                    "reason": "Prompt injection / policy bypass attempt detected.",
                    "action": "refuse"
                }
        return {"passed": True}

    def check_output_grounding(self, response_text: str, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Output Guardrail: Enforces grounded citation tags in output."""
        if not citations:
            return {"passed": True, "grounding_score": 1.0}
            
        # Check if response references citations
        has_citation_ref = "[" in response_text and "]" in response_text or "Book:" in response_text or "OSHA" in response_text or "DGMS" in response_text or "MSHA" in response_text
        
        if not has_citation_ref:
            return {
                "passed": False,
                "reason": "Generated response lacks grounded citation references.",
                "grounding_score": 0.0
            }
            
        return {"passed": True, "grounding_score": 1.0}
