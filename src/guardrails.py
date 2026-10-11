import re
from typing import Dict, Any, List

class SafetyGuardrails:
    """Layer 3 Real-Time Input and Output Guardrails."""
    def __init__(self):
        self.injection_patterns = [
            r"ignore (all )?previous instructions",
            r"system prompt",
            r"you are now (an? )?unrestricted",
            r"override safety", r"jailbreak", r"bypass guardrails",
            r"disregard (everything|all|previous)",
            r"hidden instructions",
            r"no longer apply",
            r"forget (your )?(earlier|previous) guidance",
            r"write me a (limerick|poem|song|story)"
        ]
        self.primary_patterns = [
            r"\b(mine|mines|mining|miner|miners)\b",
            r"\b(coal|goaf|gob|strata|longwall|seam|seams|bench|benches|highwall|quarry|quarries|opencast|underground)\b",
            r"\b(haul road|haul roads|haulage|shuttle car|shuttle cars|dumper|dumpers|haul truck|man-riding|locomotive)\b",
            r"\b(msha|dgms|cmr|omr|osha)\b",
            r"\b(berm|berms|parapet|extensometer|tell-tale)\b",
            r"\b(loto|lockout|tagout|hazwoper)\b",
            r"\b(anfo|flyrock|blast|blasts|blasting|misfire|post-blast|had not fired)\b",
            r"\b(methane|firedamp|blackdamp|toxic gas|co concentration)\b",
            r"\b(roof bolt|roof bolts|roof bolting|roof fall|roof slab|cable bolt|cable bolting|roof convergence|cutter roof)\b",
            r"\b(flameproof|flp|fras|pds|lhd|lhds|continuous miner)\b",
            r"\b(trailing cable|trailing cables|high-voltage|substation|arc flash|ground fault|dielectric)\b",
            r"\b(inundation|water-logged|flooded workings|shaft flood|drill steel|probe holes|stone dust|stone dusting)\b",
            r"\b(stemming|burden|workings|gallery|galleries|heading|headings|barrier pillar|barrier pillars)\b",
            r"\b(wooden props|prop|props|breaker|gradient|conveyor belt|conveyor drum|nip point|feeder breaker|tail drum|pull cord|pull cords)\b",
            r"\b(proximity system|proximity detection|ssr|highwall monitoring|warning siren|retreat|emergency stop|halt mine)\b"
        ]
        self.secondary_patterns = [
            r"\b(accident|accidents|hazard|hazards|fatal|fatality|injuries|injury)\b",
            r"\b(ventilation|ventilator|excavator|shovel|conveyor|explosion|explosions|combustion)\b",
            r"\b(drill|drilling|pillar|pillars|electrocution|overturn|skid|collapse)\b",
            r"\b(truck driver|cable sheath|conductors?|shaft|evacuation drills?)\b",
            r"\b(fire broke out|mine fire|coal fire|spontaneous fire)\b"
        ]

    def check_input(self, query: str) -> Dict[str, Any]:
        q_lower = query.lower().strip()
        for pattern in self.injection_patterns:
            if re.search(pattern, q_lower):
                return {"passed": False, "reason": "Prompt injection / policy bypass attempt detected.", "action": "refuse"}

        for pattern in self.primary_patterns:
            if re.search(pattern, q_lower):
                return {"passed": True}

        sec_matches = sum(1 for p in self.secondary_patterns if re.search(p, q_lower))
        if sec_matches >= 1 and any(w in q_lower for w in ["truck", "cable", "shaft", "ventilat", "drill", "conveyor", "safety", "fire broke", "gas", "ground", "power"]):
            return {"passed": True}

        return {
            "passed": False,
            "reason": "I am only able to answer questions related to mine safety, accident investigation reports, and mining regulations. Please ask a safety-related question.",
            "action": "out_of_domain"
        }

    def check_output_grounding(self, response_text: str, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not citations:
            return {"passed": True, "grounding_score": 1.0}
        has_ref = "[" in response_text and "]" in response_text or "Book:" in response_text or "OSHA" in response_text or "DGMS" in response_text or "MSHA" in response_text or "Source:" in response_text
        return {"passed": has_ref, "grounding_score": 1.0 if has_ref else 0.0}
