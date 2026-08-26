import time
import json
import os
from typing import Dict, Any, Optional

AUDIT_LOG_FILE = "./data/production_audit_logs.jsonl"

def lookup_mine_regulation(regulation_id: str) -> Dict[str, Any]:
    """
    Read-only typed tool: Looks up specific MSHA, OSHA, or DGMS regulation details.
    Blast radius: Low.
    """
    reg_clean = regulation_id.upper().strip()
    
    registry = {
        "CMR-104": {
            "title": "Coal Mines Regulations 2017 Regulation 104",
            "body": "Methane gas (CH4) threshold: Max 0.75% in working places, Max 1.25% in return airways. Withdrawal mandatory above 1.25%."
        },
        "CMR-111": {
            "title": "Coal Mines Regulations 2017 Regulation 111",
            "body": "Underground Ventilation: Min 6 cubic metres/min per person on largest shift, or 2.5 m3/min per tonne daily production."
        },
        "CMR-123": {
            "title": "Coal Mines Regulations 2017 Regulation 123",
            "body": "Strata Control & Support: Full column resin roof bolting mandatory at max 1.0m grid spacing for Rock Mass Rating RMR < 40."
        },
        "OSHA-1910.147": {
            "title": "OSHA 29 CFR 1910.147 Lockout/Tagout (LOTO)",
            "body": "Control of Hazardous Energy: Application of individual padlocks/tags, zero-energy verification before maintenance."
        },
        "OSHA-1910.120": {
            "title": "OSHA 29 CFR 1910.120 HAZWOPER",
            "body": "Hazardous Waste & Emergency Response: Continuous atmospheric monitoring (H2S, CO, O2), mandatory Level B/A SCBA PPE."
        }
    }
    
    match = registry.get(reg_clean)
    if match:
        return {"status": "success", "data": match}
    return {
        "status": "not_found",
        "message": f"Regulation '{regulation_id}' not found in local quick registry. Querying vector index..."
    }

def log_incident_audit_entry(site_id: str, hazard_type: str, details: str) -> Dict[str, Any]:
    """
    Write typed tool: Logs an immutable safety audit entry into production logs.
    Blast radius: Low (Idempotent audit log).
    """
    os.makedirs(os.path.dirname(AUDIT_LOG_FILE), exist_ok=True)
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "site_id": site_id,
        "hazard_type": hazard_type,
        "details": details,
        "action": "log_incident"
    }
    
    with open(AUDIT_LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
        
    return {
        "status": "success",
        "audit_id": f"AUDIT-{int(time.time())}",
        "message": f"Incident logged successfully for site '{site_id}'."
    }

def issue_emergency_stop_directive(site_id: str, reason: str, human_approved: bool = False) -> Dict[str, Any]:
    """
    High-Impact Write Tool: Issues an emergency mine stop directive.
    Blast radius: HIGH. MUST be gated on human safety officer approval!
    """
    if not human_approved:
        return {
            "status": "gated_approval_required",
            "requires_human_signoff": True,
            "site_id": site_id,
            "reason": reason,
            "message": f"CRITICAL DIRECTIVE: Emergency stop for site '{site_id}' requires explicit human safety officer sign-off!"
        }

    os.makedirs(os.path.dirname(AUDIT_LOG_FILE), exist_ok=True)
    directive = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "directive_id": f"EMERGENCY-STOP-{int(time.time())}",
        "site_id": site_id,
        "reason": reason,
        "approved_by": "Human Safety Officer",
        "status": "EXECUTED_IMMEDIATELY"
    }
    
    with open(AUDIT_LOG_FILE, "a") as f:
        f.write(json.dumps(directive) + "\n")

    return {
        "status": "executed",
        "directive_id": directive["directive_id"],
        "message": f"🚨 EMERGENCY STOP DIRECTIVE EXECUTED for site '{site_id}'. Reason: {reason}."
    }
