import json
import os
import time
from typing import Dict, Any, List

PRODUCTION_TRACE_FILE = "./data/production_audit_logs.jsonl"
GOLDEN_DATASET_FILE = "./data/mining_golden_dataset.json"

class ProductionDiscoveryLoop:
    """
    Figure 1 Step 4-7 Production Monitoring & Discovery Loop:
    - Step 4: Smart Log Filtering
    - Step 5: Select & Deploy Production Metrics (Latency, Rerank Scores, Containment)
    - Step 6: Guardrails & Improvement Loops
    - Step 7: Build Emerging Issue Discovery (Appends low-confidence / unhandled traces to Golden Dataset)
    """
    def __init__(self):
        os.makedirs(os.path.dirname(PRODUCTION_TRACE_FILE), exist_ok=True)

    def log_trace(self, query: str, response: Dict[str, Any], user_rating: str = "neutral"):
        """Logs execution span telemetry into production trace store."""
        trace = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "latency_ms": response.get("telemetry", {}).get("total_latency_ms", 0),
            "num_contexts": response.get("telemetry", {}).get("num_contexts_used", 0),
            "llm_used": response.get("telemetry", {}).get("llm_used", "Unknown"),
            "num_citations": len(response.get("citations", [])),
            "user_rating": user_rating
        }
        
        with open(PRODUCTION_TRACE_FILE, "a") as f:
            f.write(json.dumps(trace) + "\n")

    def get_production_metrics(self) -> Dict[str, Any]:
        """Calculates production telemetry metrics across logged traces."""
        if not os.path.exists(PRODUCTION_TRACE_FILE):
            return {"total_requests": 0, "avg_latency_ms": 0.0, "containment_rate": 100.0}

        traces = []
        with open(PRODUCTION_TRACE_FILE, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        traces.append(json.loads(line.strip()))
                    except Exception:
                        pass

        if not traces:
            return {"total_requests": 0, "avg_latency_ms": 0.0, "containment_rate": 100.0}

        total = len(traces)
        avg_lat = sum(t.get("latency_ms", 0) for t in traces) / total
        contained = sum(1 for t in traces if t.get("num_citations", 0) > 0)
        containment_rate = (contained / total) * 100.0

        return {
            "total_requests": total,
            "avg_latency_ms": round(avg_lat, 2),
            "containment_rate": round(containment_rate, 2),
            "recent_traces": traces[-10:]
        }

    def capture_emerging_edge_case(self, query: str, expected_domain: str = "MSHA/OSHA/DGMS Safety"):
        """Step 7: Captures unhandled / low-confidence edge case and appends to Golden Reference Dataset."""
        if not os.path.exists(GOLDEN_DATASET_FILE):
            return

        with open(GOLDEN_DATASET_FILE, "r") as f:
            dataset = json.load(f)

        # Check if already present
        existing_queries = [item["query"] for item in dataset]
        if query in existing_queries:
            return

        new_entry = {
            "query": query,
            "expected_keywords": [expected_domain.split()[0]],
            "expected_regulation": expected_domain
        }
        dataset.append(new_entry)

        with open(GOLDEN_DATASET_FILE, "w") as f:
            json.dump(dataset, f, indent=2)

        print(f"[Discovery Loop]: Appended new edge case to Golden Reference Dataset ({len(dataset)} items total).")
