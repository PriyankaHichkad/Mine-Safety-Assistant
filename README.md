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

#### A. Document Retrieval Benchmarks (BM25 vs. Hybrid RRF)

| Retrieval Metric | Sparse BM25 | Hybrid (Dense + BM25 + RRF) | Target SLA / Gate | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Document Hit@5** | **95.0%** (57/60) | **100.0%** (60/60) | $\ge 85.0\%$ | **PASSED** |
| **Retrieval Recall@5** | **94.6%** | **99.6%** | $\ge 90.0\%$ | **PASSED** |
| **Mean Reciprocal Rank (MRR@5)** | **0.880** | **0.989** | $\ge 0.600$ | **PASSED** |
| **Document Hit@1 (Top Rank)** | **83.3%** | **98.3%** | — | — |
| **Retrieval Latency (p50)** | **0.22 ms** | **8,081.63 ms** (CPU) | Sub-millisecond BM25 | **MEASURED** |
| **Retrieval Latency (p95)** | **0.38 ms** | **10,801.26 ms** (CPU) | Sub-millisecond BM25 | **MEASURED** |
| **Retrieval + Guardrail Latency (p50)** | **0.31 ms** | **8,081.80 ms** (CPU) | Real query timing | **MEASURED** |
| **Retrieval + Guardrail Latency (p95)** | **1.15 ms** | **10,801.38 ms** (CPU) | Real query timing | **MEASURED** |

#### B. Generative Faithfulness & Citation Grounding: 100-Query LLM-as-a-Judge Evaluation vs. Offline Extractive CI Sanity Check

| Evaluation Metric | Production Generative RAG (100 Queries, Live LLM Judge) | Extractive Fallback CI Check (100 Queries, Offline Sandbox) | Target SLA / Gate | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Passage-Level Precision@5** | **88.4%** | **88.4%** | $\ge 85.0\%$ | **PASSED** |
| **Citation Grounding (Faithfulness)** | **82.5%** ($[0.783, 0.868]$ 95% CI) | **100.0%** (Format check on extractive chunks) | $\ge 80.0\%$ | **PASSED** |
| **Citation Integrity Rate** | **100.0%** (100/100 valid source tags) | **100.0%** | $100.0\%$ | **PASSED** |
| **SLA Compliance Rate ($\text{Score} \ge 0.85$)** | **57.0%** ($[0.472, 0.663]$ Wilson CI) | **100.0%** (True by extractive construction) | — | **MEASURED** |
| **Mean Generation Latency** | **8.77 s / query** (Cloud Groq API + Network) | **0.10 ms / query** (Deterministic In-Memory) | — | **MEASURED** |
| **Mean Evaluation / Judge Latency** | **10.87 s / query** (Structured Entailment Judge) | **0.00 ms** (Lexical Overlap Matrix) | — | **MEASURED** |
| **Mean Total Pipeline Latency** | **19.64 s / query** (Retrieval + Gen + Judge) | **0.30 ms / query** | — | **MEASURED** |
| **Evaluation Dataset & Sample Scope** | **100 technical queries** (`cause`, `control`, `regulation`, `topic`, `terse`) | **100 technical queries** (Identical sample) | Full in-domain benchmark | **COMMITTED** |
| **Model Attribution & Verification** | **`qwen/qwen3.8-27b` / `openai/gpt-oss-120b`** (Logged per row) | **`extractive_fallback`** (Direct chunk extraction) | Explicitly Verified | **VERIFIED** |
| **Evaluation Engine** | **LLM-as-a-Judge** (Claim entailment & risk verification) | **Format Check on Extractive Output** | — | **REPRODUCIBLE** |
| **Committed Results Artifact** | [`evaluation/results/groq_live_eval_100.json`](evaluation/results/groq_live_eval_100.json) | [`evaluation/results/grounding_eval.json`](evaluation/results/grounding_eval.json) | Labeled Benchmark | **COMMITTED** |

#### C. Safety & Security Guardrail SLAs

| Guardrail Metric | Measured Result | 95% Confidence Interval | Target SLA / Gate | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Out-of-Domain Refusal Rate** | **100.0%** (12/12) | $[0.758, 1.000]$ | $\ge 80.0\%$ | **PASSED** |
| **Adversarial Injection Block Rate** | **100.0%** (6/6) | $[0.610, 1.000]$ | $\ge 80.0\%$ | **PASSED** |
| **Guardrail False Refusal Rate** | **0.0%** (0/60) | $[0.000, 0.060]$ | $\le 10.0\%$ | **PASSED** |

> **Evaluation Methodology & Benchmark Datasets**:
> - **Production Generative RAG (100 Queries, Live LLM-as-a-Judge)**:
>   - Unlike naive keyword overlap, faithfulness is evaluated claim-by-claim using a separate LLM judge (`openai/gpt-oss-120b` / `qwen/qwen3.8-27b`) with structured JSON schema output across all 100 in-domain queries in [`evaluation/golden_set_150.json`](evaluation/golden_set_150.json).
>   - The LLM judge evaluates **both** factual statements (causes, equipment dynamics, regulations) against retrieved contexts **and** numerical statistical claims (accident probability %, fatality rate %, incident counts) against the risk analytics engine's corpus statistics.
>   - Every individual query record in [`evaluation/results/groq_live_eval_100.json`](evaluation/results/groq_live_eval_100.json) records the responding `model_used`, `judge_model`, `generation_latency_ms`, `retrieval_latency_ms`, `judge_latency_ms`, and itemized `claim_evaluations` with explicit textual rationales.
>   - Yields a genuine production faithfulness score of **82.5% Citation Grounding** ($[0.783, 0.868]$ 95% CI) and **88.4% Passage Precision@5**, rigorously penalizing subtle hallucinations and unsubstantiated general assertions.
> - **Offline Extractive CI Mode (Format & Ingestion Sanity Check)**:
>   - In zero-network GitHub Actions sandboxes without API access, the pipeline defaults to deterministic extractive synthesis (`_grounded_fallback` in `src/engine.py`), copying statutory sentences directly from retrieved chunks.
>   - Because sentences are directly extracted from chunks, token overlap against the same chunks is 100% by construction. This score is explicitly classified as an **ingestion & format integrity sanity check** (verifying that retrieved chunks are parsed, formatted, and tagged correctly in CI in 11 seconds at $0 cost) rather than a measure of generative language modeling faithfulness.
> - **Document Retrieval Evaluation**: Evaluated across 60 technical mining queries in [`evaluation/golden_set.json`](evaluation/golden_set.json) comparing fast lexical BM25 against the full Hybrid RRF pipeline (Dense `bge-base-en-v1.5` + BM25 + Cross-Encoder `bge-reranker-base`). Dense cross-encoder reranking eliminates BM25 lexical misses (100% Hit@5, 0.989 MRR@5) with recorded per-query latency in [`evaluation/results/hybrid.json`](evaluation/results/hybrid.json) and [`evaluation/results/bm25.json`](evaluation/results/bm25.json).
> - **150-Query Benchmark Suite ([`evaluation/golden_set_150.json`](evaluation/golden_set_150.json))**: Contains 100 in-domain technical queries, 25 out-of-domain edge cases, and 25 adversarial prompt injections. On this suite, BM25 alone achieves **Recall@5 of 96.8%**, **Hit@5 of 97.0%**, and an **MRR@5 of 0.900** (raw results committed in [`evaluation/results/bm25_150.json`](evaluation/results/bm25_150.json)).
> - **Measured Per-Query Latency**: Every query is timed dynamically during evaluation (measuring retrieval, guardrail filtering, generation, and LLM judge latency) and committed to raw JSON files, reporting exact p50, p95, and mean percentiles.

---

### 2. Engineering & Data Science Tradeoffs

- **Bi-Encoder vs. Cross-Encoder Tradeoff**: Dense bi-encoder search (`bge-base-en-v1.5`) performs candidate retrieval over vector space but lacks deep token-level query-document cross-attention. Adding `bge-reranker-base` cross-encoder reranking over RRF candidate pairs boosts **MRR@5 from 0.880 to 0.989** and **Hit@5 from 95.0% to 100.0%**, successfully resolving all BM25 lexical misses at the cost of cross-attention latency.
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
   - Automated GitHub Actions workflows ([`ci_eval.yml`](file:///.github/workflows/ci_eval.yml) and [`rag_eval_ci.yml`](file:///.github/workflows/rag_eval_ci.yml)) testing IR retrieval accuracy (Hit@5, MRR@5) and guardrail containment SLAs on labeled benchmarks.

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
# 1. Retrieval & guardrail gate (Hit@5, MRR@5, latency, refusal SLAs)
python scripts/run_eval_ci.py

# 2. Citation grounding & passage-level precision gate (Faithfulness & Relevance SLAs)
python scripts/run_grounding_eval_ci.py
```

### 4. Launch Streamlit Web Application
```bash
python -m streamlit run ui/streamlit_app.py
```
Open **`http://localhost:8501`** in your browser!
