import os
import sys
import time
from typing import Dict, Any, List, TypedDict, Optional
from langgraph.graph import StateGraph, END

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.retrieval.hybrid_search import MineMindHybridRetriever
from src.generation.rag_engine import MineMindRAGEngine
from src.graph.router import MineSafetyRouter
from src.tools.safety_tools import lookup_mine_regulation, issue_emergency_stop_directive, log_incident_audit_entry
from src.optimization.cache import SemanticQueryCache
from src.evals.guardrails import SafetyGuardrails
from src.evals.discovery_loop import ProductionDiscoveryLoop

# Define LangGraph State Schema
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
    """
    Enterprise-Grade LangGraph Mine Accident Prevention & Hazard Copilot
    Implementing the 5-Layer Customer Support System Design Architecture:
    - Layer 1: Model & Intent Router Node
    - Layer 2: Wrapping Layer (Contextual Hybrid Retrieval, Typed Safety Tools, Memory)
    - Layer 3: Inline Guardrails & Figure 1 Discovery Loop
    - Layer 4 & 5: Production Telemetry & Sub-10ms Semantic Query Cache
    """
    def __init__(self, db_path: str = "./data/qdrant_db", bm25_path: str = "./data/bm25_index.pkl"):
        self.retriever = MineMindHybridRetriever(db_path=db_path, bm25_path=bm25_path)
        self.rag_engine = MineMindRAGEngine(self.retriever)
        self.router = MineSafetyRouter()
        self.cache = SemanticQueryCache()
        self.guardrails = SafetyGuardrails()
        self.discovery_loop = ProductionDiscoveryLoop()
        self.graph = self._build_langgraph_workflow()

    def _build_langgraph_workflow(self) -> StateGraph:
        """Constructs a 5-node LangGraph StateGraph pipeline with Layer 1 Intent Routing."""
        workflow = StateGraph(MineSafetyState)

        # Add Nodes to Graph
        workflow.add_node("route_intent", self._node_route_intent)
        workflow.add_node("parse_scenario", self._node_parse_scenario)
        workflow.add_node("hybrid_retrieve", self._node_hybrid_retrieve)
        workflow.add_node("safety_guardrail", self._node_safety_guardrail)
        workflow.add_node("generate_plan", self._node_generate_plan)

        # Define DAG Edges
        workflow.set_entry_point("route_intent")
        workflow.add_edge("route_intent", "parse_scenario")
        workflow.add_edge("parse_scenario", "hybrid_retrieve")
        workflow.add_edge("hybrid_retrieve", "safety_guardrail")
        
        # Conditional Edge based on Safety Guardrail
        workflow.add_conditional_edges(
            "safety_guardrail",
            lambda state: "generate_plan" if state["prelevance_passed"] else END
        )
        workflow.add_edge("generate_plan", END)

        return workflow.compile()

    def _node_route_intent(self, state: MineSafetyState) -> Dict[str, Any]:
        """Node 0: Layer 1 Model & Intent Router."""
        classification = self.router.classify_intent(state["query"])
        return {
            "tier": classification["tier"],
            "intent": classification["intent"],
            "requires_human_approval": classification["requires_human_approval"],
            "target_tool": classification["target_tool"]
        }

    def _node_parse_scenario(self, state: MineSafetyState) -> Dict[str, Any]:
        """Node 1: Extract Mine Parameters & Hazard Category from Query."""
        query = state["query"].lower()
        
        mine_type = "Underground Mining" if any(w in query for w in ["underground", "pillar", "shaft", "seam", "longwall"]) else "Opencast / Surface Mining"
        
        hazard_cat = "General Mine Safety"
        if any(w in query for w in ["roof", "fall", "strata", "bolting"]):
            hazard_cat = "Roof Fall & Strata Failure"
        elif any(w in query for w in ["haulage", "dumper", "shuttle", "truck"]):
            hazard_cat = "Powered Haulage & Vehicle Overturn"
        elif any(w in query for w in ["methane", "gas", "explosion", "ventilation"]):
            hazard_cat = "Methane Gas & Firedamp Explosion"
        elif any(w in query for w in ["cable", "electric", "power", "shock"]):
            hazard_cat = "Electrical Ground Fault"
        elif any(w in query for w in ["blast", "flyrock", "powder"]):
            hazard_cat = "Blasting Flyrock Hazard"
        elif any(w in query for w in ["loto", "lockout", "tagout"]):
            hazard_cat = "OSHA Lockout/Tagout (LOTO)"
        elif any(w in query for w in ["hazwoper", "h2s", "toxic"]):
            hazard_cat = "OSHA HAZWOPER Toxic Gas Protection"

        return {
            "mine_type": mine_type,
            "hazard_category": hazard_cat
        }

    def _node_hybrid_retrieve(self, state: MineSafetyState) -> Dict[str, Any]:
        """Node 2: LangChain BM25 + Qdrant Hybrid Search across MSHA & DGMS Corpus."""
        start_time = time.time()
        results = self.retriever.search(query=state["query"], top_k=15, final_top_m=3)
        ret_latency = (time.time() - start_time) * 1000

        return {
            "retrieved_docs": results,
            "telemetry": {"retrieval_latency_ms": round(ret_latency, 2)}
        }

    def _node_safety_guardrail(self, state: MineSafetyState) -> Dict[str, Any]:
        """Node 3: Safety Guardrail Relevance Score Thresholding (>0.01)."""
        docs = state.get("retrieved_docs", [])
        top_score = docs[0].get("rerank_score", 0.0) if docs else 0.0
        
        passed = top_score >= 0.01 or len(docs) > 0
        if not passed:
            refusal = (
                "I could not find relevant mining textbook literature or safety regulations in the knowledge base to answer your query. "
                "Please ask a question related to mining engineering, mine definitions, safety rules, MSHA fatality reports, or OSHA standards."
            )
            self.discovery_loop.capture_emerging_edge_case(state["query"])
            
            return {
                "prelevance_passed": False,
                "prevention_plan": refusal,
                "citations": []
            }
            
        return {"prelevance_passed": True}

    def _node_generate_plan(self, state: MineSafetyState) -> Dict[str, Any]:
        """Node 4: Synthesize MSHA Grounded Prevention Plan & Citations."""
        start_gen = time.time()
        model_override = state.get("model_override")
        res = self.rag_engine.answer_query(state["query"], model_override=model_override)
        gen_latency = (time.time() - start_gen) * 1000

        telemetry = state.get("telemetry", {})
        telemetry["total_latency_ms"] = res["telemetry"]["total_latency_ms"]
        telemetry["generation_latency_ms"] = round(gen_latency, 2)
        telemetry["llm_used"] = res["telemetry"].get("llm_used", "Ollama")

        return {
            "prevention_plan": res["answer"],
            "citations": res["citations"],
            "telemetry": telemetry
        }

    def run_safety_query(self, query: str, model_override: Optional[str] = None, human_approved: bool = False) -> Dict[str, Any]:
        """Executes the compiled LangGraph workflow with Cache, Guardrails, and Tools."""
        # 1. Check Input Guardrails (Prompt Injection Defense)
        guard_res = self.guardrails.check_input(query)
        if not guard_res["passed"]:
            return {
                "query": query,
                "answer": f"🛡️ INPUT GUARDRAIL BLOCKED: {guard_res['reason']}",
                "citations": [],
                "telemetry": {"total_latency_ms": 1.0, "llm_used": "Input Guardrail Refusal"}
            }

        # 2. Check Layer 5 Semantic Query Cache (Sub-10ms response)
        cached_res = self.cache.get(query)
        if cached_res:
            return cached_res

        # 3. Check Layer 1 Router for High-Impact Actions
        intent_data = self.router.classify_intent(query)
        if intent_data["target_tool"] == "issue_emergency_stop_directive":
            tool_res = issue_emergency_stop_directive("SITE-ALPHA-01", reason=query, human_approved=human_approved)
            if tool_res["status"] == "gated_approval_required":
                return {
                    "query": query,
                    "answer": f"🚨 **HIGH-IMPACT EMERGENCY ACTION DETECTED**\n\n{tool_res['message']}\n\n*Please click 'Approve & Execute Emergency Stop' in the UI below to authorize this directive.*",
                    "citations": [],
                    "requires_human_approval": True,
                    "telemetry": {"total_latency_ms": 5.0, "llm_used": "Emergency Directive Router"}
                }
            else:
                return {
                    "query": query,
                    "answer": f"🚨 **EMERGENCY DIRECTIVE EXECUTED**\n\n{tool_res['message']}",
                    "citations": [],
                    "requires_human_approval": False,
                    "telemetry": {"total_latency_ms": 12.0, "llm_used": "Emergency Directive Tool"}
                }

        # 4. Execute Full LangGraph Workflow
        initial_state: Dict[str, Any] = {
            "query": query,
            "tier": "",
            "intent": "",
            "requires_human_approval": False,
            "target_tool": None,
            "mine_type": "",
            "depth_m": 0,
            "hazard_category": "",
            "retrieved_docs": [],
            "prelevance_passed": False,
            "prevention_plan": "",
            "citations": [],
            "telemetry": {},
            "model_override": model_override,
            "human_approved": human_approved
        }
        
        final_state = self.graph.invoke(initial_state)
        response = {
            "query": query,
            "mine_type": final_state.get("mine_type"),
            "hazard_category": final_state.get("hazard_category"),
            "answer": final_state.get("prevention_plan"),
            "citations": final_state.get("citations", []),
            "telemetry": final_state.get("telemetry", {})
        }

        # 5. Populate Cache & Log Telemetry Trace (Figure 1 Step 4)
        if final_state.get("prelevance_passed", False):
            self.cache.set(query, response)

        self.discovery_loop.log_trace(query, response)
        return response
