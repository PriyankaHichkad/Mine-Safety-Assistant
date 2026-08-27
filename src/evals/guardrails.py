import re
from typing import Dict, Any, List

class SafetyGuardrails:
    """
    Layer 3 Real-Time Input and Output Guardrails.
    - Input Guardrail: Prompt Injection Defense & Strict Domain Control (Mining Safety & Regulations Only).
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
        
        # Core domain keywords for mining safety, accidents, rules, and engineering
        self.domain_keywords = [
            "mine", "mining", "safety", "accident", "report", "fatal", "fatality", "fire", "heating", 
            "spontaneous", "combustion", "explosion", "methane", "gas", "roof", "fall", "strata", 
            "bolting", "pillar", "haulage", "shuttle", "dumper", "truck", "berm", "parapet", 
            "electric", "shock", "cable", "loto", "lockout", "tagout", "hazwoper", "ventilation", 
            "blast", "flyrock", "msha", "dgms", "osha", "cmr", "omr", "rule", "rules", "regulation", 
            "hazard", "cause", "probability", "prevention", "dust", "inundation", "flood", "water", 
            "coal", "seam", "bench", "pit", "underground", "opencast", "surface", "equipment", 
            "machinery", "conveyor", "excavator", "shovel", "quarry", "worker", "miner", "injury"
        ]

    def check_input(self, query: str) -> Dict[str, Any]:
        """Input Guardrail: Prompt Injection Defense & Domain Restriction."""
        q_lower = query.lower().strip()
        
        # 1. Prompt Injection Check
        for pattern in self.injection_patterns:
            if re.search(pattern, q_lower):
                return {
                    "passed": False,
                    "reason": "Prompt injection / policy bypass attempt detected.",
                    "action": "refuse"
                }

        # 2. Strict Domain Relevance Check (Domain Scope Restriction)
        # Check if query contains at least one mining/safety domain concept
        is_domain_query = any(kw in q_lower for kw in self.domain_keywords)
        
        if not is_domain_query:
            return {
                "passed": False,
                "reason": "I am only able to answer questions related to mine safety, accident investigation reports, and mining regulations. Please ask a safety-related question.",
                "action": "out_of_domain"
            }

        return {"passed": True}

    def check_output_grounding(self, response_text: str, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Output Guardrail: Enforces grounded citation tags in output."""
        if not citations:
            return {"passed": True, "grounding_score": 1.0}
            
        has_citation_ref = "[" in response_text and "]" in response_text or "Book:" in response_text or "OSHA" in response_text or "DGMS" in response_text or "MSHA" in response_text or "Source:" in response_text
        
        if not has_citation_ref:
            return {
                "passed": False,
                "reason": "Generated response lacks grounded citation references.",
                "grounding_score": 0.0
            }
            
        return {"passed": True, "grounding_score": 1.0}
