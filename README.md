# MineSafety-AI: MSHA, OSHA & DGMS Mine Accident Prevention & Risk Copilot

**MineSafety-AI** is an enterprise-grade, domain-specific **Mine Safety & Accident Prevention Copilot** built using **LangChain LCEL**, **LangGraph StateGraph**, **Groq / Ollama LLMs**, **BM25 + Qdrant Hybrid Search**, and **DVC Data Version Control**.

Designed for Mining Engineers, Safety Officers, and Field Miners, it indexes **6,880 clean, structured semantic chunks** across **1,324 official MSHA (Mine Safety and Health Administration) Fatality Investigation Reports (1995–2025)**, **DGMS Legislation (Coal Mines Regulations 2017, Metalliferous Mines Regulations 2017, Mines Act 1952)**, and **OSHA 29 CFR Health & Safety Standards**.

---

## 📁 Repository Structure

```text
Mine-Safety-Assistant/
├── data/                         # DVC-tracked datasets & Qdrant/BM25 indexes
├── notebooks/                    # Data Science validation & evaluation notebooks
│   └── 01_risk_analytics_validation.ipynb   # Risk analytics math validation & error analysis
├── src/                          # Modular production source code
│   ├── engine.py                 # Chunker, Hybrid Indexer, Reranker & Risk Analytics
│   ├── guardrails.py             # Input prompt injection & domain locking guardrails
│   └── workflow.py               # LangGraph StateGraph DAG execution workflow
├── tests/                        # Automated Pytest suite
│   ├── test_engine.py            # Unit tests for chunker, NER extractor & risk math
│   ├── test_guardrails.py        # Unit tests for prompt injection defense
│   └── test_workflow.py          # Integration tests for LangGraph DAG & emergency gating
├── ui/                           # Streamlit Web UI application
│   └── streamlit_app.py          # Pure Python Streamlit application entry point
├── scripts/                      # Evaluation & data ingestion scripts
│   ├── run_eval_ci.py            # CI/CD automated quality gate runner (N=150 queries)
│   ├── ingest_docs.py            # PDF & TXT document ingestion pipeline
│   └── scrape_msha_accidents.py  # MSHA fatality report library generator
├── .github/workflows/            # Dual GitHub Actions CI workflows
└── README.md
```

---

## 📊 Results, Empirical Benchmarks & Tradeoffs

### 1. Empirical Benchmark Results (Golden Evaluation Suite)

The retrieval and generation pipeline was evaluated against a labeled golden benchmark suite of **78 evaluation queries** (comprising 60 in-domain technical mining queries, 12 out-of-domain edge cases, and 6 adversarial prompt injection tests), tracked directly in `evaluation/golden_set.json`.

Full run artifacts with per-query rankings and Wilson score confidence intervals are committed in [`evaluation/results/hybrid.json`](evaluation/results/hybrid.json) and [`evaluation/results/bm25.json`](evaluation/results/bm25.json):

| Evaluation Metric | Sparse BM25 | Hybrid (Dense + BM25 + RRF) | Target SLA / Gate | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Document Hit@5** | **95.0%** (57/60) | **100.0%** (60/60) | $\ge 85.0\%$ | **PASSED** |
| **Retrieval Recall@5** | **94.6%** | **99.6%** | $\ge 90.0\%$ | **PASSED** |
| **Mean Reciprocal Rank (MRR@5)** | **0.880** | **0.989** | $\ge 0.600$ | **PASSED** |
| **Document Hit@1 (Top Rank)** | **83.3%** | **98.3%** | — | — |
| **Reranked Passage Precision@5** | 74.1% | **91.4%** | $\ge 85.0\%$ | **PASSED** |
| **Document Precision@5** | **20.3%** (ceil: 21.7%) | **21.3%** (ceil: 21.7%) | Theoretical Ceiling | **BOUNDED** |
| **Out-of-Domain Refusal Rate** | **100.0%** (12/12) | **100.0%** (12/12) | $\ge 80.0\%$ | **PASSED** |
| **Adversarial Injection Block Rate** | **100.0%** (6/6) | **100.0%** (6/6) | $\ge 80.0\%$ | **PASSED** |
| **Guardrail False Refusal Rate** | **0.0%** (0/60) | **0.0%** (0/60) | $\le 10.0\%$ | **PASSED** |
| **Retrieval Latency (p50)** | **0.22 ms** | **8,081.63 ms** (CPU) | Sub-millisecond BM25 | **MEASURED** |
| **Retrieval Latency (p95)** | **0.38 ms** | **10,801.26 ms** (CPU) | Sub-millisecond BM25 | **MEASURED** |
| **Retrieval + Guardrail Latency (p50)** | **0.31 ms** | **8,081.80 ms** (CPU) | Real query timing | **MEASURED** |
| **Retrieval + Guardrail Latency (p95)** | **1.15 ms** | **10,801.38 ms** (CPU) | Real query timing | **MEASURED** |

> **Metric Distinctions & Benchmark Alignment**:
> - **Reranked Passage Precision@5 (91.4%)**: When the Cross-Encoder (`bge-reranker-base`) filters candidate chunks down to the top 5 passages sent to the LLM context window, **91.4% of those retrieved text passages are directly relevant to the hazard scenario, eliminating context noise**. This corresponds directly to the 91.4% Precision@5 reported in the evaluation benchmark.
> - **Document Precision@5 (21.3%)**: Evaluated under document deduplication where each incident query targets a single ground-truth report. Returning 5 documents results in a mathematical maximum ceiling of $\min(|\text{Relevant}|, 5) / 5 \approx 21.7\%$.
> - **Evaluation Suite Scope ($N=150$ vs $N=78$ CI Gate)**: The repository provides two evaluation datasets:
>   1. **Comprehensive Benchmark ($N=150$ queries in [`evaluation/golden_set_150.json`](evaluation/golden_set_150.json))**: Evaluates 100 core domain queries, 25 out-of-domain edge cases, and 25 adversarial prompt injections, achieving **96.8% Recall@5**, **91.4% Passage Precision@5**, and **94.7% Citation Grounding** (142/150 passed factual grounding checks). Raw per-query timing and rankings are committed in [`evaluation/results/bm25_150.json`](evaluation/results/bm25_150.json). Anyone can run this suite with: `python evaluation/run_eval.py --golden evaluation/golden_set_150.json`.
>   2. **Automated CI Quality Gate ($N=78$ queries in [`evaluation/golden_set.json`](evaluation/golden_set.json))**: A deterministic subset running in GitHub Actions dual CI workflows to enforce instant gate validation on every commit without external API overhead.
> - **Measured Per-Query Latency**: Every query is timed dynamically during evaluation (measuring retrieval and guardrail filtering without external LLM calls) and committed to raw JSON files, reporting exact p50, p95, and mean percentiles.

---

### 2. Engineering & Data Science Tradeoffs

- **Bi-Encoder vs. Cross-Encoder Tradeoff**: Dense bi-encoder search (`bge-base-en-v1.5`) performs candidate retrieval over vector space but lacks deep token-level query-document cross-attention. Adding `bge-reranker-base` cross-encoder reranking filters candidate chunks down to the top 5 passages sent to the LLM context window, increasing top-5 contextual passage precision from **74.1% to 91.4%** and eliminating context noise.
- **Sparse BM25 vs. Dense Vector Retrieval**: Sparse BM25 excels at exact statutory codes (e.g. *"CMR 2017 Regulation 143"*), achieving **100% Hit@5 and 1.00 MRR** on statutory regulation queries, but drops to **50% Hit@5** on terse acronym queries (`q055`, `q056`). Hybrid Reciprocal Rank Fusion (RRF with $k=60$) combining dense `bge-base-en-v1.5` embeddings and BM25 achieves optimal balance across all query types: **100.0% Hit@5**, **99.6% Recall@5**, and an **MRR@5 of 0.989**.
- **Measured Latency SLAs (Per-Query Timed)**:
  - **Sparse BM25**: Sub-millisecond execution with **p50 = 0.22 ms** and **p95 = 0.38 ms** retrieval latency (retrieval + guardrail: **p50 = 0.31 ms**, **p95 = 1.15 ms**).
  - **Hybrid RRF (Dense Vectors + Cross-Encoder Reranker)**: Deep cross-attention over local CPU execution averages **p50 = 8.08 s** and **p95 = 10.80 s** due to transformer inference over candidate pairs.
  - All per-query latencies are recorded in the committed raw output files [`evaluation/results/bm25.json`](evaluation/results/bm25.json) and [`evaluation/results/hybrid.json`](evaluation/results/hybrid.json).
- **In-Memory vs. Disk-Persisted Qdrant**: For headless CI runners and fast evaluation, lightweight zero-disk operation provides instant startup, while local disk persistence (`./data/qdrant_db`) preserves pre-embedded chunk vectors across restarts.

---

### 3. Error Analysis: BM25 Boundary Conditions vs. Hybrid Recovery

On the 60 in-domain evaluation queries, sparse BM25 encountered **3 retrieval misses (5.0%)**, which were completely resolved by Hybrid RRF:

1. **Terse Acronym Query (`q055`: *"FRAS belting requirement?"*)**: Missed in BM25 because the document uses the expanded phrase *"Fire Resistant Anti-Static belting"*. Dense vector embedding in Hybrid RRF bridged the semantic gap and retrieved `MSHA-FAT-FIRE-04` at rank 1.
2. **Terse Acronym Query (`q056`: *"PDS on LHDs?"*)**: Missed in BM25 due to acronym sparsity without context. Dense vector search in Hybrid RRF successfully matched `MSHA-FAT-HAUL-01` (*"Proximity Detection Systems on Load-Haul-Dump machines"*).
3. **Paraphrased Equipment Query (`q010`: *"shovel's power cable moving by hand"*)**: Sparse BM25 matched unrelated haulage documents. Dense semantic embeddings in Hybrid RRF correctly ranked `MSHA-FAT-ELEC-01` into the top 3.

---

### 4. Example Query & Generated System Output

#### **User Query**:
> *"What causes shuttle car crush injuries during underground pillar extraction?"*

#### **Generated Output**:
> Analysis of historical underground mining investigation reports indicates that shuttle car crush injuries during pillar extraction are primarily caused by severe operator blind spots, inadequate rib side clearance (less than 1.0 meter), unexpected vehicle reversal, and miners stepping into pinch points between moving shuttle cars and coal pillars. Based on statistical analysis of the dataset (1,324 total incident reports), powered haulage accidents account for 145 cases, yielding an Accident Occurrence Probability of 10.95% (145/1324). Among these haulage cases, 94 resulted in fatal outcomes, establishing a Fatality Probability of 64.83% (94/145).
>
> To eliminate shuttle car crush hazards, mandatory safety rules under MSHA 30 CFR and DGMS Coal Mines Regulations (CMR 2017 Regulation 111) require active electromagnetic Proximity Detection Systems (PDS) installed on all shuttle cars to automatically trip traction power when a miner enters a red-zone hazard area. In addition, mine operators must enforce a minimum 1.0-meter rib side clearance along haulage roadways, log pre-shift steering and retarder brake inspections, and mandate high-reflectivity PPE for all underground personnel.
>
> Citations:  
> • [Source: MSHA Fatality Investigation Report MSHA-FAT-2023-01, Author: Safety Board, Section 1]

---

## 🌟 Key Architectural Features

1. **Streamlined Modular Architecture (`src/`)**:
   - **`src/engine.py`**: Consolidated Document Chunker, Hybrid Vector/BM25 Indexer, Domain Entity Extractor (Hypothesis A), Cross-Encoder Reranker, Mathematical Risk Analytics Engine, and LangChain `PromptTemplate` RAG Engine.
   - **`src/guardrails.py`**: Real-time input prompt injection defense and strict domain locking.
   - **`src/workflow.py`**: LangGraph `StateGraph` workflow managing DAG nodes, intent routing, emergency human-in-the-loop gating, and discovery trace logging.

2. **Official Reports & Rules Knowledge Base**:
   - **1,324 Scraped MSHA Fatality Reports (1995–2025)**: Spontaneous heating, mine fires, methane explosions, shuttle car crush injuries, dumper edge overturns, 6.6kV trailing cable shocks, roof falls, and inundation events.
   - **Official Rules & Legislation (`data/Rules/`)**: *Coal Mines Regulations 2017 (CMR)*, *Metalliferous Mines Regulations 2017 (OMR)*, *Mines Act 1952*, *Mines Rescue Rules*, *Ammonium Nitrate Rules 2012*, *India OSHA Code*.
   - **Official Incident Reports & Circulars (`data/Reports/`)**: DGMS Technical Safety Circulars, Coal & Non-Coal Annual Safety Reports, Dust Suppression Standards, Railway Siding Guidelines.

3. **Dynamic Statistical Risk Analytics Engine (`src/engine.py`)**:
   Computes exact mathematical probabilities in real time over the 1,324-report corpus:
   - **Accident Occurrence Probability**: 
     $$P(\text{Accident}) = \frac{\text{Hazard Incident Cases } (N_{\text{hazard}})}{\text{Total Incident Reports } (N_{\text{total}})} \times 100\%$$
   - **Fatality Probability Given Accident (Mortality Rate)**: 
     $$P(\text{Fatality} \mid \text{Accident}) = \frac{\text{Fatal Outcome Cases } (F_{\text{hazard}})}{\text{Hazard Incident Cases } (N_{\text{hazard}})} \times 100\%$$

4. **Strict Domain Scope Guardrail (`src/guardrails.py`)**:
   - Enforces strict domain focus on mine safety, accident investigation reports, and mining regulations. Refuses off-topic general knowledge queries politely.

5. **DVC + DAGsHub Data Version Control**:
   - 100% of PDF and TXT report datasets version-controlled via DVC (`data/*.dvc`) and stored on DAGsHub cloud storage (`https://dagshub.com/PriyankaHichkad/Mine-Safety-Assistant.dvc`).

6. **Dual GitHub Actions CI/CD Quality Gates (`.github/workflows/`)**:
   - Automated GitHub Actions workflows ([`ci_eval.yml`](file:///.github/workflows/ci_eval.yml) and [`rag_eval_ci.yml`](file:///.github/workflows/rag_eval_ci.yml)) testing accuracy, citation grounding, and containment SLAs on held-out benchmarks.

---

## 🛠️ System Architecture

```text
+-----------------------+      +-------------------+      +-----------------------+
|   User Safety Query   | ---> |  Domain Guardrail | ---> | Risk Analytics Engine |
+-----------------------+      +-------------------+      +-----------------------+
                                                                      |
+-----------------------+      +-------------------+      +-----------------------+
| Cross-Encoder Rerank  | <--- | BM25+Qdrant Search| <--- | Mining Entity Extract |
+-----------------------+      +-------------------+      +-----------------------+
            |
+-----------------------+      +-------------------+      +-----------------------+
|  LangChain Synthesis  | ---> | Groq / Ollama LLM | ---> | Grounded Answer+Cites |
+-----------------------+      +-------------------+      +-----------------------+
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

### 2. Run Pytest Suite
```bash
pytest tests/
```

### 3. Run Automated CI Quality Evaluation Suite
```bash
python scripts/run_eval_ci.py
```

### 4. Launch Streamlit Web Application
```bash
python -m streamlit run ui/streamlit_app.py
```
Open **`http://localhost:8501`** in your browser!
