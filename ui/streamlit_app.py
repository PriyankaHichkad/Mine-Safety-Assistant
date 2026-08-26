import os
import sys
import time
import streamlit as st

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph.safety_graph import LangGraphMineSafetyEngine

# Page Configuration
st.set_page_config(
    page_title="Mine Safety Assistant | MSHA & OSHA Compliance Guide",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Readability Light Slate Theme CSS
st.markdown("""
<style>
    /* Clean Light Background */
    .stApp, .main, [data-testid="stHeader"] {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }
    
    /* Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #f1f5f9 !important;
        border-right: 1px solid #cbd5e1 !important;
    }
    
    /* General Paragraph & Text */
    p, span, label, li, div {
        color: #0f172a !important;
    }
    
    /* Select & Input styling */
    .stSelectbox label, .stSlider label, .stNumberInput label, .stTextInput label {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        font-weight: 600 !important;
    }
    
    /* Citation Badges */
    .citation-badge {
        background-color: #fef3c7 !important;
        color: #78350f !important;
        border: 1px solid #f59e0b !important;
        padding: 0.3rem 0.7rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.88rem;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #0f172a !important;
        font-weight: 800 !important;
    }
    
    /* Tab Styling */
    button[data-baseweb="tab"] {
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }
    button[aria-selected="true"] {
        color: #2563eb !important;
        border-bottom-color: #2563eb !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_emergency_query" not in st.session_state:
    st.session_state.pending_emergency_query = None

# Auto-detect Groq API key silently from environment or Streamlit Secrets
groq_key_secret = os.getenv("GROQ_API_KEY", "")
if not groq_key_secret and hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
    groq_key_secret = st.secrets["GROQ_API_KEY"]

# LangGraph Safety Engine Initialization with Silent Backend Logging
def load_safety_engine(groq_key: str):
    db_path = "./data/qdrant_db"
    bm25_path = "./data/bm25_index.pkl"
    
    if not os.path.exists(db_path) or not os.path.exists(bm25_path):
        from scripts.scrape_msha_accidents import generate_msha_dataset
        from scripts.ingest_docs import main as ingest_main
        generate_msha_dataset()
        ingest_main()

    engine = LangGraphMineSafetyEngine(db_path=db_path, bm25_path=bm25_path)
    if groq_key and hasattr(engine.rag_engine, "ollama"):
        engine.rag_engine.ollama.api_key = groq_key
        try:
            import groq
            engine.rag_engine.ollama.groq_client = groq.Groq(api_key=groq_key)
        except Exception:
            pass
    return engine

engine = load_safety_engine(groq_key_secret)

# Sidebar: User-Centric Simple Navigation
with st.sidebar:
    st.title("🛡️ Mine Safety Assistant")
    st.caption("Official MSHA & OSHA Compliance Guide")
    st.divider()

    st.success("🟢 Assistant Online & Ready")

    st.subheader("💡 Common Safety Questions")
    st.markdown("""
    - What causes shuttle car accidents?
    - What is mandatory dumper berm height?
    - Roof bolting rules for soft rock RMR < 40
    - Electrical safety for trailing cables
    - OSHA Lockout/Tagout (LOTO) rules
    """)

    st.divider()
    st.info("ℹ️ All answers are grounded in official MSHA fatality reports, OSHA standards, and DGMS circulars.")

# Main Title Banner
st.title("🛡️ Mine Safety & Hazard Assistant")
st.caption("A practical safety guide for mine workers, safety officers, and mining engineering students.")

tabs = st.tabs(["💬 Ask Safety Assistant", "📋 Mine Safety Plan Generator", "📖 Official Safety Rules & Manuals"])

# TAB 1: Chat Assistant (Default Tab for Miners & Students)
with tabs[0]:
    st.subheader("Ask a Question About Mine Safety, Regulations, or Hazards")
    
    # Sample Query Buttons
    st.caption("Click a sample question to get started:")
    btn_cols = st.columns(4)
    sample_q = None
    if btn_cols[0].button("🚜 Shuttle Car Accidents"):
        sample_q = "What causes shuttle car crush injuries during underground pillar extraction in MSHA fatality reports?"
    if btn_cols[1].button("🚛 Dumper Edge Overturn"):
        sample_q = "What parapet wall height is required to prevent opencast dump truck rollbacks under DGMS circulars?"
    if btn_cols[2].button("⚡ Trailing Cable Shock"):
        sample_q = "What electrical ground continuity safety precautions are mandatory for heavy excavator trailing cables?"
    if btn_cols[3].button("🚨 Emergency Stop"):
        sample_q = "Emergency stop halt mine operations immediately on Bench-04 due to highwall slope movement."

    # Handle Pending Human Approval Gate
    if st.session_state.pending_emergency_query:
        st.warning("⚠️ EMERGENCY DIRECTIVE REQUIRES SAFETY OFFICER SIGN-OFF")
        st.write(f"**Action Directive**: {st.session_state.pending_emergency_query}")
        col_app1, col_app2 = st.columns(2)
        if col_app1.button("✅ Confirm & Issue Emergency Stop Order", use_container_width=True):
            res = engine.run_safety_query(st.session_state.pending_emergency_query, human_approved=True)
            st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
            st.session_state.pending_emergency_query = None
            st.rerun()
        if col_app2.button("❌ Cancel Order", use_container_width=True):
            st.session_state.pending_emergency_query = None
            st.info("Emergency action cancelled.")
            st.rerun()

    # Render Existing Messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                st.caption("📌 Official References & Proofs:")
                for c in msg["citations"]:
                    page_loc = c.get('page_number') or c.get('section') or 'Standard Reference'
                    st.markdown(f"<span class='citation-badge'>📖 {c['doc_title']}</span> &nbsp; ✍️ **Author**: *{c['author']}* &nbsp; 📍 **Location**: `{page_loc}`", unsafe_allow_html=True)

    # Chat Input Box
    chat_input_val = st.chat_input("Ask a question about mine safety or hazard rules...")
    active_user_q = chat_input_val or sample_q

    if active_user_q:
        st.session_state.messages.append({"role": "user", "content": active_user_q})
        with st.chat_message("user"):
            st.markdown(active_user_q)

        with st.chat_message("assistant"):
            with st.spinner("Searching official MSHA & OSHA safety guidelines..."):
                res = engine.run_safety_query(active_user_q)
                
                if res.get("requires_human_approval"):
                    st.session_state.pending_emergency_query = active_user_q
                    st.warning(res["answer"])
                    st.rerun()
                else:
                    st.markdown(res["answer"])
                    if res["citations"]:
                        st.caption("📌 Official References & Proofs:")
                        for c in res["citations"]:
                            page_loc = c.get('page_number') or c.get('section') or 'Standard Reference'
                            st.markdown(f"<span class='citation-badge'>📖 {c['doc_title']}</span> &nbsp; ✍️ **Author**: *{c['author']}* &nbsp; 📍 **Location**: `{page_loc}`", unsafe_allow_html=True)
                        
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": res["answer"],
                        "citations": res.get("citations", [])
                    })
                    st.rerun()

# TAB 2: Interactive Mine Risk Plan Generator Form
with tabs[1]:
    st.subheader("Generate a Mine Safety Checklist & Prevention Plan")
    st.caption("Select your mine operational details to get a customized accident prevention checklist based on real MSHA fatality investigations.")

    col1, col2 = st.columns(2)
    with col1:
        mine_type = st.selectbox(
            "Mining Operation Type",
            ["Underground Coal (Bord & Pillar / Pillar Extraction)", "Underground Coal (Longwall Working)", "Opencast / Surface Coal Mine", "Opencast Heavy Equipment Site"]
        )
        depth_m = st.slider("Operating Depth / Bench Height (metres)", 20, 600, 350)
    with col2:
        equipment = st.selectbox(
            "Primary Equipment Deployed",
            ["Shuttle Cars & Continuous Miners", "Longwall Shearer & Powered Roof Supports", "100-Ton Rear Dumpers & Excavators", "6.6kV Electric Shovel Trailing Cables", "Bench Drilling & Blasting Explosives"]
        )
        workers = st.number_input("Shift Workforce Size (Miners)", 10, 200, 45)

    if st.button("🛡️ Generate Safety & Hazard Prevention Plan", use_container_width=True):
        scenario_query = f"I am operating a {mine_type} at {depth_m}m depth deploying {equipment} with {workers} workers per shift. What are the historical fatality causes, mandatory safety precautions under MSHA/DGMS, and required emergency training plans for this setup?"
        
        if not engine:
            st.error("Engine loading. Please try again in 5 seconds.")
        else:
            with st.spinner("Analyzing safety rules and historical accident reports..."):
                res = engine.run_safety_query(scenario_query)
                
                st.markdown("### 📋 Mine Safety & Accident Prevention Checklist")
                st.markdown(res["answer"])
                
                st.subheader("📌 Official Regulations Cited")
                for c in res["citations"]:
                    page_loc = c.get('page_number') or c.get('section') or 'Standard Reference'
                    st.markdown(f"<span class='citation-badge'>📖 {c['doc_title']}</span> &nbsp; ✍️ **Author**: *{c['author']}* &nbsp; 📍 **Location**: `{page_loc}`", unsafe_allow_html=True)

# TAB 3: Official Safety Rules & Manuals
with tabs[2]:
    st.subheader("📖 Official Mining Regulations & Fatality Report Library")
    st.caption("Key regulatory standards covered in this safety assistant:")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        ### ⛏️ MSHA & DGMS Safety Codes
        - **MSHA-FAT-2023-01**: Shuttle Car Crush Accidents & Proximity Detection Systems (PDS).
        - **MSHA-FAT-2023-02**: Underground Roof Falls, RMR Support Plans & Resin Bolting.
        - **MSHA-FAT-2024-01**: Opencast Dumper Overturns & Parapet Wall / Berm Heights.
        - **MSHA-FAT-2023-05**: 6.6kV Electric Shovel Trailing Cable Ground Continuity.
        """)
    with col_b:
        st.markdown("""
        ### 🛡️ OSHA Health & Safety Standards
        - **OSHA 29 CFR 1910.147**: Lockout/Tagout (LOTO) for Conveyor & Machine Maintenance.
        - **OSHA 29 CFR 1910.120**: HAZWOPER & Multi-Gas Respirator Protection (H2S / Toxic Gas).
        - **CMR 2017 Regulations**: Ventilation, Gas Limits & Strata Control Rules.
        """)
