# MineSafety-AI 🛡️: MSHA, OSHA & DGMS Mine Accident Prevention & Risk Copilot

**MineSafety-AI** is an enterprise-grade, domain-specific **Mine Safety & Accident Prevention Copilot** built using **LangChain**, **LangGraph StateGraph**, **Groq / Ollama LLMs**, **BM25 + Qdrant Hybrid Search**, and **DVC Data Version Control**.

Designed for Mining Engineers, Safety Officers, and Field Miners, it indexes **48,071 semantic chunks** across **1,324 official MSHA (Mine Safety and Health Administration) Fatality Investigation Reports (1995–2025)**, **DGMS Legislation (Coal Mines Regulations 2017, Metalliferous Mines Regulations 2017, Mines Act 1952)**, **OSHA 29 CFR Health & Safety Standards**, and **Mining Engineering Reference Textbooks**.

---

## 🌟 Key Engineering Features

1. **48,071 Chunk Knowledge Base**:
   - **1,324 Scraped MSHA Fatality Reports (1995–2025)**: Spontaneous heating, mine fires, methane explosions, shuttle car crush injuries, dumper edge overturns, 6.6kV trailing cable shocks, roof falls, and inundation events.
   - **Official Rules & Legislation (`data/Rules/`)**: *Coal Mines Regulations 2017 (CMR)*, *Metalliferous Mines Regulations 2017 (OMR)*, *Mines Act 1952*, *Mines Rescue Rules*, *Ammonium Nitrate Rules 2012*, *India OSHA Code*.
   - **Official Incident Reports & Circulars (`data/Reports/`)**: DGMS Technical Safety Circulars, Coal & Non-Coal Annual Safety Reports, Dust Suppression Standards, Railway Siding Guidelines.
   - **Mining Reference Textbooks (`data/weebly_books/` & `data/pdf_books/`)**: Subsurface Ventilation, Rock Mechanics, Open Pit Design, Surface Blast Engineering, Mine Power Systems.

2. **Dynamic Statistical Risk Analytics Engine (`src/analytics/risk_analyzer.py`)**:
   Computes exact mathematical probabilities in real time over the 1,324-report corpus:
   - **Accident Occurrence Probability**: 
     $$P(\text{Accident}) = \frac{\text{Hazard Incident Cases } (N_{\text{hazard}})}{\text{Total Incident Reports } (N_{\text{total}})} \times 100\%$$
   - **Fatality Probability Given Accident (Mortality Rate)**: 
     $$P(\text{Fatality} \mid \text{Accident}) = \frac{\text{Fatal Outcome Cases } (F_{\text{hazard}})}{\text{Hazard Incident Cases } (N_{\text{hazard}})} \times 100\%$$

3. **2-Step Analytical Safety Framework**:
   - **Paragraph 1 (Historical Fatality & Root Cause Analysis)**: Analyzes historical case studies and presents exact mathematical accident occurrence & fatality probability percentages.
   - **Paragraph 2 (Regulatory Compliance & Required Prevention Controls)**: Cross-references official regulations (CMR 2017, OMR 2017, Mines Act 1952, MSHA 30 CFR, OSHA) to detail mandatory engineering controls (parapet berm height = tyre radius, Proximity Detection Systems, FRAS belts, LOTO).
   - **Citations**: Attaches grounded source metadata tags at the bottom.

4. **Strict Domain Scope Guardrail (`src/evals/guardrails.py`)**:
   - Enforces strict domain focus on mine safety, accident investigation reports, and mining regulations. Refuses off-topic general knowledge queries politely.

5. **DVC + DAGsHub Data Version Control**:
   - 100% of large PDF files and 1,324 report datasets version-controlled via DVC (`data/*.dvc`) and stored on DAGsHub cloud storage (`https://dagshub.com/PriyankaHichkad/Mine-Safety-Assistant.dvc`).

6. **Dual 100% Green CI/CD Quality Gates (`.github/workflows/`)**:
   - Automated GitHub Actions workflows ([`ci_eval.yml`](file:///.github/workflows/ci_eval.yml) and [`rag_eval_ci.yml`](file:///.github/workflows/rag_eval_ci.yml)) testing accuracy (100%), citation grounding (100%), and containment SLAs.

7. **Clean Streamlit Web Interface (`ui/streamlit_app.py`)**:
   - High-contrast, clean layout without left AI sidebars or sample question buttons. Includes plain written sample questions, interactive Mine Risk Plan Generator, and chat interface.

---

## 🛠️ System Architecture

```mermaid
flowchart TD
    UserQuery["User Input Query"] --> Guardrail{"Domain & Safety Guardrail"}
    Guardrail -- Out-of-Domain --> Refusal["Domain Refusal Response"]
    Guardrail -- In-Domain --> RiskEngine["Statistical Risk Analytics Engine"]
    RiskEngine --> Formulas["Calculate P(Accident) & P(Fatality|Accident)"]
    Formulas --> HybridRetriever["BM25 + Qdrant Hybrid Search (48,071 Chunks)"]
    HybridRetriever --> Reranker["Cross-Encoder Reranker (BAAI/bge-reranker-base)"]
    Reranker --> PromptBuilder["2-Step Prompt Construction"]
    PromptBuilder --> LLM["Groq Cloud / Ollama LLM"]
    LLM --> StructuredResponse["2-Paragraph Narrative Answer + Bottom Citations"]
```

---

## 🚀 Quickstart Guide

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/PriyankaHichkad/Mine-Safety-Assistant.git
cd Mine-Safety-Assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Pull Datasets with DVC (DAGsHub Remote)
```bash
dvc pull
```

### 3. Build Knowledge Base & Indexes
```bash
python scripts/scrape_msha_accidents.py
python scripts/ingest_docs.py
```

### 4. Run Automated CI Quality Evaluation Suite
```bash
python scripts/run_eval_ci.py
```

### 5. Launch Streamlit Web Application
```bash
python -m streamlit run ui/streamlit_app.py
```
Open **`http://localhost:8501`** in your browser!

---

## 📝 Senior-Matched Placement Resume Bullets

1. **Enterprise RAG Architecture & Vector Search**:
   > *"Designed and deployed an end-to-end Mine Safety RAG pipeline indexing 48,071 semantic chunks across 1,324 scraped MSHA fatality reports, DGMS circulars, and OSHA 29 CFR standards using BM25, Qdrant Vector DB, and BAAI/bge-reranker-base."*
2. **Mathematical Risk Analytics & LangGraph Engineering**:
   > *"Built a dynamic statistical risk engine calculating real-time accident occurrence and fatality probabilities ($P(\text{Fatality} \mid \text{Accident}) = \frac{F_{\text{hazard}}}{N_{\text{hazard}}}$) integrated into a 2-step LangGraph StateGraph workflow with 100% citation grounding."*
3. **MLOps, DVC Cloud Versioning & CI/CD**:
   > *"Version-controlled 1,400+ compliance PDFs and text corpora via DVC on DAGsHub and established dual GitHub Actions CI/CD quality gates enforcing 100% accuracy and citation SLA benchmarks."*
