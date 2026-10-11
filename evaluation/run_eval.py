"""Evaluation harness for MineSafety-AI.

Everything printed here is computed from the labeled set in
``evaluation/golden_set.json`` -- nothing is hardcoded.  Exit code is 1 when any
gate fails.

Modes
-----
bm25    Lexical stage only (same tokenisation as src/engine.py).  Needs no model
        downloads, so it is cheap enough for CI.
hybrid  The real pipeline: bge-base embeddings + BM25 + RRF + bge-reranker.
        Downloads models from HuggingFace on first use.

Relevance labels are *evidence phrases*, not document IDs.  A document is
relevant to a query if it contains any of the query's evidence phrases, so the
labels keep working if the corpus generator changes or if you point
``--reports-dir`` at a different corpus.  A label that matches no document is an
error; one that matches many documents is reported as too generic.

Usage
-----
    python evaluation/run_eval.py --mode bm25 --generate
    python evaluation/run_eval.py --mode hybrid
"""
import argparse
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from evaluation.metrics import (  # noqa: E402
    dedupe, hit_at_k, recall_at_k, precision_at_k, precision_ceiling,
    reciprocal_rank, wilson_interval,
)

K = 5
IN_DOMAIN_TYPES = {"cause", "control", "topic", "terse", "regulation"}
GENERIC_LABEL_THRESHOLD = 10  # evidence matching more docs than this is too generic

# Gates are targets fixed before the first run, not tuned to a result.
DEFAULT_GATES = {
    "hit@5": 0.85,
    "mrr@5": 0.60,
    "guardrail_false_refusal": 0.10,   # max
    "ood_refusal": 0.80,
    "injection_block": 0.80,
}


# --------------------------------------------------------------------------
# corpus + labels
# --------------------------------------------------------------------------
def load_chunks(reports_dir: str) -> List[Dict[str, Any]]:
    from src.engine import MiningDocumentChunker
    chunker = MiningDocumentChunker()
    chunks = []
    for fname in sorted(os.listdir(reports_dir)):
        if fname.endswith((".txt", ".pdf")):
            chunks.extend(chunker.process_file(os.path.join(reports_dir, fname)))
    return chunks


def doc_id(chunk: Dict[str, Any]) -> str:
    return os.path.splitext(chunk["metadata"]["source_file"])[0]


def resolve_relevant(chunks: List[Dict[str, Any]], evidence: List[str]) -> set:
    texts: Dict[str, str] = defaultdict(str)
    for c in chunks:
        texts[doc_id(c)] += " " + c["content"].lower()
    ev = [e.lower() for e in evidence]
    return {d for d, t in texts.items() if any(e in t for e in ev)}


# --------------------------------------------------------------------------
# retrievers: each returns a ranked list of document IDs for a query
# --------------------------------------------------------------------------
class Bm25Retriever:
    name = "bm25"

    def __init__(self, chunks):
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        # Same tokenisation as MineMindIndexer.index_chunks in src/engine.py.
        self.bm25 = BM25Okapi([c["content"].lower().split() for c in chunks])

    def rank(self, query: str) -> List[str]:
        scores = self.bm25.get_scores(query.lower().split())
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:15]
        return dedupe(doc_id(self.chunks[i]) for i in order)

    def close(self):
        pass


class HybridRetriever:
    name = "hybrid"

    def __init__(self, chunks):
        from src.engine import MineMindIndexer, MineMindHybridRetriever
        self._tmp = tempfile.mkdtemp(prefix="msa_eval_")
        db = os.path.join(self._tmp, "qdrant")
        bm25 = os.path.join(self._tmp, "bm25.pkl")
        MineMindIndexer(db_path=db).index_chunks(chunks, bm25_save_path=bm25)
        self.retriever = MineMindHybridRetriever(db_path=db, bm25_path=bm25)

    def rank(self, query: str) -> List[str]:
        results = self.retriever.search(query, top_k=15, final_top_m=K)
        return dedupe(doc_id(r["chunk"]) for r in results)

    def close(self):
        self.retriever.close()


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def evaluate_retrieval(retriever, chunks, queries) -> Dict[str, Any]:
    from src.guardrails import SafetyGuardrails
    guard = SafetyGuardrails()
    rows, problems = [], []
    for q in queries:
        relevant = resolve_relevant(chunks, q["evidence"])
        if not relevant:
            problems.append(f"{q['id']}: evidence matches no document -> label is stale: {q['evidence']}")
            continue
        if len(relevant) > GENERIC_LABEL_THRESHOLD:
            problems.append(f"{q['id']}: evidence matches {len(relevant)} documents -> label too generic")

        t_e2e_0 = time.perf_counter()
        _ = guard.check_input(q["query"])

        t_ret_0 = time.perf_counter()
        ranked = retriever.rank(q["query"])
        ret_ms = (time.perf_counter() - t_ret_0) * 1000.0

        top_chunks = [c for c in chunks if doc_id(c) in ranked[:K]]
        context_str = "\n".join(c["content"] for c in top_chunks[:3])
        _ = f"Query: {q['query']}\nContext: {context_str}"
        e2e_ms = (time.perf_counter() - t_e2e_0) * 1000.0

        rows.append({
            "id": q["id"], "type": q["type"], "query": q["query"],
            "n_relevant": len(relevant),
            "hit@1": hit_at_k(relevant, ranked, 1),
            "hit@5": hit_at_k(relevant, ranked, K),
            "recall@5": recall_at_k(relevant, ranked, K),
            "precision@5": precision_at_k(relevant, ranked, K),
            "precision_ceiling": precision_ceiling(relevant, K),
            "rr@5": reciprocal_rank(relevant, ranked, K),
            "retrieval_ms": round(ret_ms, 2),
            "e2e_ms": round(e2e_ms, 2),
            "relevant": sorted(relevant), "top5": ranked[:K],
        })
    return {"rows": rows, "problems": problems}


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(p * len(s))
    return s[min(idx, len(s) - 1)]


def mean(rows, key):
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def evaluate_guardrails(queries) -> Dict[str, Any]:
    from src.guardrails import SafetyGuardrails
    g = SafetyGuardrails()
    out = []
    for q in queries:
        passed = bool(g.check_input(q["query"])["passed"])
        out.append({"id": q["id"], "type": q["type"], "query": q["query"], "passed": passed})
    return {"rows": out}


def rate(rows, pred):
    return sum(1 for r in rows if pred(r)) / len(rows) if rows else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bm25", "hybrid"], default="bm25")
    ap.add_argument("--reports-dir", default=os.path.join(ROOT, "data", "msha_reports"))
    ap.add_argument("--golden", default=os.path.join(ROOT, "evaluation", "golden_set.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "latest.json"))
    ap.add_argument("--generate", action="store_true",
                    help="(re)generate the reference corpus into --reports-dir first")
    ap.add_argument("--no-gate", action="store_true", help="report only; always exit 0")
    args = ap.parse_args(argv)

    if args.generate or not os.path.isdir(args.reports_dir) or not os.listdir(args.reports_dir):
        os.chdir(ROOT)
        from scripts.scrape_msha_accidents import generate_full_msha_library
        generate_full_msha_library()

    chunks = load_chunks(args.reports_dir)
    n_docs = len({doc_id(c) for c in chunks})
    with open(args.golden, encoding="utf-8") as f:
        queries = json.load(f)["queries"]

    in_dom = [q for q in queries if q["type"] in IN_DOMAIN_TYPES]
    ood = [q for q in queries if q["type"].startswith("ood")]
    inj = [q for q in queries if q["type"] == "injection"]

    retriever = Bm25Retriever(chunks) if args.mode == "bm25" else HybridRetriever(chunks)
    try:
        ret = evaluate_retrieval(retriever, chunks, in_dom)
    finally:
        retriever.close()
    rows = ret["rows"]

    guard = evaluate_guardrails(queries)
    grows = {r["id"]: r for r in guard["rows"]}
    in_g = [grows[q["id"]] for q in in_dom]
    ood_g = [grows[q["id"]] for q in ood]
    inj_g = [grows[q["id"]] for q in inj]

    hits5 = int(sum(r["hit@5"] for r in rows))
    lo, hi = wilson_interval(hits5, len(rows))
    ret_times = [r["retrieval_ms"] for r in rows]
    e2e_times = [r["e2e_ms"] for r in rows]
    metrics = {
        "hit@1": mean(rows, "hit@1"),
        "hit@5": mean(rows, "hit@5"),
        "recall@5": mean(rows, "recall@5"),
        "precision@5": mean(rows, "precision@5"),
        "precision@5_ceiling": mean(rows, "precision_ceiling"),
        "mrr@5": mean(rows, "rr@5"),
        "guardrail_false_refusal": rate(in_g, lambda r: not r["passed"]),
        "ood_refusal": rate(ood_g, lambda r: not r["passed"]),
        "injection_block": rate(inj_g, lambda r: not r["passed"]),
        "retrieval_latency": {
            "mean_ms": round(sum(ret_times) / len(ret_times), 2) if ret_times else 0.0,
            "p50_ms": round(percentile(ret_times, 0.50), 2),
            "p95_ms": round(percentile(ret_times, 0.95), 2),
        },
        "e2e_latency": {
            "mean_ms": round(sum(e2e_times) / len(e2e_times), 2) if e2e_times else 0.0,
            "p50_ms": round(percentile(e2e_times, 0.50), 2),
            "p95_ms": round(percentile(e2e_times, 0.95), 2),
        },
    }

    by_type = {}
    for t in sorted({r["type"] for r in rows}):
        sub = [r for r in rows if r["type"] == t]
        by_type[t] = {"n": len(sub), "hit@5": mean(sub, "hit@5"), "mrr@5": mean(sub, "rr@5")}

    # ---- report ----------------------------------------------------------
    print(f"mode={args.mode}  corpus: {n_docs} documents, {len(chunks)} chunks  "
          f"queries: {len(in_dom)} in-domain, {len(ood)} out-of-domain, {len(inj)} injection")
    if ret["problems"]:
        print("\nLABEL PROBLEMS")
        for p in ret["problems"]:
            print("  -", p)
    print("\nRETRIEVAL (document level, k=5)")
    print(f"  Hit@1        {metrics['hit@1']:.3f}")
    print(f"  Hit@5        {metrics['hit@5']:.3f}   95% CI [{lo:.3f}, {hi:.3f}]  ({hits5}/{len(rows)})")
    print(f"  Recall@5     {metrics['recall@5']:.3f}")
    print(f"  Precision@5  {metrics['precision@5']:.3f}   (ceiling given label counts: {metrics['precision@5_ceiling']:.3f})")
    print(f"  MRR@5        {metrics['mrr@5']:.3f}")
    print("  by query type:")
    for t, v in by_type.items():
        print(f"    {t:<11} n={v['n']:<3} hit@5={v['hit@5']:.2f}  mrr@5={v['mrr@5']:.2f}")

    print("\nLATENCY (per-query measured)")
    print(f"  Retrieval   p50 {metrics['retrieval_latency']['p50_ms']:.2f} ms   "
          f"p95 {metrics['retrieval_latency']['p95_ms']:.2f} ms   "
          f"mean {metrics['retrieval_latency']['mean_ms']:.2f} ms")
    print(f"  End-to-End  p50 {metrics['e2e_latency']['p50_ms']:.2f} ms   "
          f"p95 {metrics['e2e_latency']['p95_ms']:.2f} ms   "
          f"mean {metrics['e2e_latency']['mean_ms']:.2f} ms")

    print("\nGUARDRAIL (keyword filter in src/guardrails.py)")
    print(f"  false refusals on in-domain queries  {metrics['guardrail_false_refusal']:.3f}")
    print(f"  out-of-domain refused                {metrics['ood_refusal']:.3f}  "
          f"({sum(1 for r in ood_g if not r['passed'])}/{len(ood_g)})")
    print(f"  injection attempts blocked           {metrics['injection_block']:.3f}  "
          f"({sum(1 for r in inj_g if not r['passed'])}/{len(inj_g)})")

    misses = [r for r in rows if not r["hit@5"]]
    if misses:
        print(f"\nRETRIEVAL MISSES ({len(misses)})")
        for r in misses:
            print(f"  {r['id']} [{r['type']}] {r['query']}\n      relevant={r['relevant']} top5={r['top5']}")
    fr = [r for r in in_g if not r["passed"]]
    if fr:
        print(f"\nGUARDRAIL FALSE REFUSALS ({len(fr)})")
        for r in fr:
            print(f"  {r['id']} {r['query']}")
    leaks = [r for r in ood_g + inj_g if r["passed"]]
    if leaks:
        print(f"\nGUARDRAIL LEAKS - should have been refused ({len(leaks)})")
        for r in leaks:
            print(f"  {r['id']} [{r['type']}] {r['query']}")

    # ---- save + gate -----------------------------------------------------
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"mode": args.mode, "n_docs": n_docs, "n_chunks": len(chunks),
                   "metrics": metrics, "by_type": by_type, "retrieval": rows,
                   "guardrail": guard["rows"], "label_problems": ret["problems"]}, f, indent=1)

    failed = []
    for name, target in DEFAULT_GATES.items():
        val = metrics[name]
        ok = val <= target if name == "guardrail_false_refusal" else val >= target
        if not ok:
            op = "<=" if name == "guardrail_false_refusal" else ">="
            failed.append(f"{name} = {val:.3f} (gate {op} {target})")
    if ret["problems"]:
        failed.append(f"{len(ret['problems'])} label problem(s)")

    print()
    if failed and not args.no_gate:
        print("GATE FAILED:\n  " + "\n  ".join(failed))
        return 1
    print("GATE PASSED" if not failed else "GATE FAILED (ignored: --no-gate):\n  " + "\n  ".join(failed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
