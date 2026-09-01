import re
from typing import Dict, Any, List

class SafetyGuardrails:
    """Layer 3 Real-Time Input and Output Guardrails."""
    def __init__(self):
        self.injection_patterns = [
            r"ignore (all )?previous instructions",
            r"system prompt",
            r"you are now (an? )?unrestricted",
            r"override safety", r"jailbreak", r"bypass guardrails"
        ]
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
        q_lower = query.lower().strip()
        for pattern in self.injection_patterns:
            if re.search(pattern, q_lower):
                return {"passed": False, "reason": "Prompt injection / policy bypass attempt detected.", "action": "refuse"}

        if not any(kw in q_lower for kw in self.domain_keywords):
            return {
                "passed": False,
                "reason": "I am only able to answer questions related to mine safety, accident investigation reports, and mining regulations. Please ask a safety-related question.",
                "action": "out_of_domain"
            }
        return {"passed": True}

    def check_output_grounding(self, response_text: str, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not citations:
            return {"passed": True, "grounding_score": 1.0}
        has_ref = "[" in response_text and "]" in response_text or "Book:" in response_text or "OSHA" in response_text or "DGMS" in response_text or "MSHA" in response_text or "Source:" in response_text
        return {"passed": has_ref, "grounding_score": 1.0 if has_ref else 0.0}
