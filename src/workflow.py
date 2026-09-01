import os
import time
from typing import Dict, Any, List, TypedDict, Optional
from langgraph.graph import StateGraph, END

from src.engine import MineMindHybridRetriever, MineMindRAGEngine
from src.guardrails import SafetyGuardrails
from src.tools.safety_tools import issue_emergency_stop_directive
from src.optimization.cache import SemanticQueryCache
from src.evals.discovery_loop import ProductionDiscoveryLoop

class MineSafetyState(TypedDict):
    query: str
    tier: str
    intent: str
    requires_human_approval: bool
    target_tool: Optional[str]
    mine_type: str
    depth_m: int
    hazard_category: str
    retrieved_docs: List[Dict[str, Any]]
    prelevance_passed: bool
    prevention_plan: str
    citations: List[Dict[str, Any]]
    telemetry: Dict[str, Any]
    model_override: Optional[str]
    human_approved: bool

class LangGraphMineSafetyEngine:
    """Enterprise-Grade LangGraph Mine Accident Prevention & Hazard Copilot."""
    def __init__(self, db_path: str = "./data/qdrant_db", bm25_path: str = "./data/bm25_index.pkl"):
        self.retriever = MineMindHybridRetriever(db_path=db_path, bm25_path=bm25_path)
        self.rag_engine = MineMindRAGEngine(self.retriever)
        self.cache = SemanticQueryCache()
        self.guardrails = SafetyGuardrails()
        self.discovery_loop = ProductionDiscoveryLoop()
        self.graph = self._build_langgraph_workflow()

    def _build_langgraph_workflow(self) -> StateGraph:
        workflow = StateGraph(MineSafetyState)
        workflow.add_node("parse_scenario", self._node_parse_scenario)
        workflow.add_node("hybrid_retrieve", self._node_hybrid_retrieve)
        workflow.add_node("safety_guardrail", self._node_safety_guardrail)
        workflow.add_node("generate_plan", self._node_generate_plan)

        workflow.set_entry_point("parse_scenario")
        workflow.add_edge("parse_scenario", "hybrid_retrieve")
        workflow.add_edge("hybrid_retrieve", "safety_guardrail")
        workflow.add_conditional_edges("safety_guardrail", lambda s: "generate_plan" if s["prelevance_passed"] else END)
        workflow.add_edge("generate_plan", END)
        return workflow.compile()

    def _node_parse_scenario(self, state: MineSafetyState) -> Dict[str, Any]:
        q = state["query"].lower()
        mtype = "Underground Mining" if any(w in q for w in ["underground", "pillar", "shaft", "seam", "longwall"]) else "Opencast / Surface Mining"
        cat = "General Mine Safety"
        if any(w in q for w in ["roof", "fall", "strata"]): cat = "Roof Fall & Strata Failure"
        elif any(w in q for w in ["haulage", "dumper", "shuttle"]): cat = "Powered Haulage & Overturn"
        elif any(w in q for w in ["fire", "heating", "spontaneous"]): cat = "Mine Fire & Combustion"
        return {"mine_type": mtype, "hazard_category": cat}

    def _node_hybrid_retrieve(self, state: MineSafetyState) -> Dict[str, Any]:
        start = time.time()
        results = self.retriever.search(query=state["query"], top_k=15, final_top_m=3)
        return {"retrieved_docs": results, "telemetry": {"retrieval_latency_ms": round((time.time() - start) * 1000, 2)}}

    def _node_safety_guardrail(self, state: MineSafetyState) -> Dict[str, Any]:
        docs = state.get("retrieved_docs", [])
        passed = (docs[0].get("rerank_score", 0.0) >= 0.01 if docs else False) or len(docs) > 0
        if not passed:
            self.discovery_loop.capture_emerging_edge_case(state["query"])
            return {"prelevance_passed": False, "prevention_plan": "I could not find relevant mining safety regulations to answer your query.", "citations": []}
        return {"prelevance_passed": True}

    def _node_generate_plan(self, state: MineSafetyState) -> Dict[str, Any]:
        res = self.rag_engine.answer_query(state["query"], model_override=state.get("model_override"))
        telemetry = state.get("telemetry", {})
        telemetry.update(res["telemetry"])
        return {"prevention_plan": res["answer"], "citations": res["citations"], "telemetry": telemetry}

    def run_safety_query(self, query: str, model_override: Optional[str] = None, human_approved: bool = False) -> Dict[str, Any]:
        guard_res = self.guardrails.check_input(query)
        if not guard_res["passed"]:
            return {"query": query, "answer": guard_res["reason"], "citations": [], "telemetry": {"total_latency_ms": 1.0, "llm_used": "Input Guardrail Refusal"}}

        cached = self.cache.get(query)
        if cached:
            return cached

        if "emergency stop" in query.lower() or "halt mine" in query.lower():
            tool_res = issue_emergency_stop_directive("SITE-ALPHA-01", reason=query, human_approved=human_approved)
            if tool_res["status"] == "gated_approval_required":
                return {"query": query, "answer": f"🚨 **HIGH-IMPACT EMERGENCY ACTION DETECTED**\n\n{tool_res['message']}\n\n*Please click 'Approve & Execute Emergency Stop' in the UI below to authorize this directive.*", "citations": [], "requires_human_approval": True, "telemetry": {"total_latency_ms": 5.0, "llm_used": "Emergency Router"}}
            return {"query": query, "answer": f"🚨 **EMERGENCY DIRECTIVE EXECUTED**\n\n{tool_res['message']}", "citations": [], "requires_human_approval": False, "telemetry": {"total_latency_ms": 12.0, "llm_used": "Emergency Tool"}}

        initial_state: MineSafetyState = {
            "query": query, "tier": "", "intent": "", "requires_human_approval": False, "target_tool": None,
            "mine_type": "", "depth_m": 0, "hazard_category": "", "retrieved_docs": [], "prelevance_passed": False,
            "prevention_plan": "", "citations": [], "telemetry": {}, "model_override": model_override, "human_approved": human_approved
        }
        
        final_state = self.graph.invoke(initial_state)
        response = {
            "query": query, "mine_type": final_state.get("mine_type"), "hazard_category": final_state.get("hazard_category"),
            "answer": final_state.get("prevention_plan"), "citations": final_state.get("citations", []), "telemetry": final_state.get("telemetry", {})
        }
        if final_state.get("prelevance_passed", False):
            self.cache.set(query, response)
        self.discovery_loop.log_trace(query, response)
        return response
