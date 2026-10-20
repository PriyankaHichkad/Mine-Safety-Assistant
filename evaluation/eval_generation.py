"""Judge-based evaluation of generated answers: claim-level grounding and passage relevance.

What this measures
------------------
For every in-domain query the RAG engine retrieves five passages and writes an answer.
Two things are then judged by a *separate* LLM acting as a strict grader:

1. Claim grounding -- the answer is split into sentences; the judge labels each one
   ``supported`` / ``partial`` / ``unsupported`` using ONLY the retrieved passages.
   True-in-the-real-world but absent-from-the-passages counts as unsupported.
2. Passage relevance -- the judge labels each of the five passages as relevant or not
   to the question (judged relevance, not keyword overlap).

Statistic sentences ("Accident Occurrence Probability of x% (a/b)") come from the risk
analyzer, not from the passages, so they are not graded for grounding. They are checked
deterministically instead: every number must equal what the analyzer returned.

Why it is built this way
------------------------
* The generator model is recorded per query. If the LLM call fails and the extractive
  fallback answers instead, that query is flagged and EXCLUDED from the headline score,
  never silently counted as an LLM result.
* An LLM judge is only as good as its agreement with a human. ``--audit-sample N`` exports
  random claims for hand labelling and ``--score-audit`` reports judge-vs-human agreement
  (raw agreement and Cohen's kappa). Report that number next to the grounding score.
* Confidence intervals are percentile bootstrap over QUERIES (claims inside one answer are
  not independent).

Usage
-----
    export GROQ_API_KEY=...            # generator and judge both use Groq
    export JUDGE_MODEL=openai/gpt-oss-120b   # pick a judge different from the generator
    python evaluation/eval_generation.py --sample 100 --retriever bm25
    python evaluation/eval_generation.py --audit-sample 40       # then label the CSV by hand
    python evaluation/eval_generation.py --score-audit evaluation/results/audit_claims.csv
"""
import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

IN_DOMAIN_TYPES = {"cause", "control", "topic", "terse", "regulation"}
VERDICT_SCORE = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}
CONTEXT_CHARS = 1800  # per passage, keeps judge prompts bounded


# ----------------------------------------------------------------------------
# judge clients
# ----------------------------------------------------------------------------
class GroqJudge:
    """Thin Groq chat client: temperature 0, retries on rate limits, records the model."""

    def __init__(self, model: str, api_key: Optional[str] = None, max_retries: int = 5):
        import groq
        self.model = model
        self.client = groq.Groq(api_key=api_key or os.environ["GROQ_API_KEY"])
        self.max_retries = max_retries

    def complete(self, system: str, user: str) -> str:
        delay = 4.0
        for attempt in range(self.max_retries):
            try:
                out = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=0.0,
                    max_tokens=1200,
                )
                return out.choices[0].message.content or ""
            except Exception as exc:  # rate limit / transient
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
        return ""


# ----------------------------------------------------------------------------
# claim handling
# ----------------------------------------------------------------------------
_STAT_RE = re.compile(r"probabilit|\(\s*\d+\s*/\s*\d+\s*\)|\d+(?:\.\d+)?\s*%", re.I)


def split_answer(text: str) -> Tuple[List[str], List[str]]:
    """Return (factual_claims, statistic_claims). Citation lines and short fragments are dropped."""
    body = re.split(r"\n\s*citations?\s*:", text, flags=re.I)[0]
    body = re.sub(r"<think>.*?</think>", "", body, flags=re.S)
    claims, stats = [], []
    for sent in re.split(r"(?<=[.!?])\s+|\n+", body):
        s = sent.strip(" -•*\t")
        if len(s.split()) < 4:
            continue
        (stats if _STAT_RE.search(s) else claims).append(s)
    return claims, stats


def stat_consistency(stat_claims: Sequence[str], risk: Dict[str, Any]) -> Optional[float]:
    """Share of numbers in statistic sentences that equal a value the risk analyzer returned.

    This verifies the answer reproduces the analyzer's numbers; it says nothing about
    whether those numbers are statistically meaningful.
    """
    if not stat_claims:
        return None
    allowed = set()
    for key in ("prob_accident_occurrence", "prob_fatality_given_accident",
                "total_reports", "hazard_matches", "fatal_matches"):
        v = risk.get(key)
        if isinstance(v, (int, float)):
            allowed.add(round(float(v), 2))
    nums = [round(float(n), 2) for c in stat_claims for n in re.findall(r"\d+(?:\.\d+)?", c)]
    if not nums:
        return None
    return sum(1 for n in nums if n in allowed) / len(nums)


# ----------------------------------------------------------------------------
# judge prompts and parsing
# ----------------------------------------------------------------------------
GROUNDING_SYSTEM = (
    "You are a strict fact-checker for a mine-safety assistant. You are given numbered PASSAGES "
    "and numbered CLAIMS from an answer. Judge each claim using ONLY the passages. Do not use "
    "outside knowledge: a claim that is true in the real world but not stated in the passages is "
    "'unsupported'. Labels: 'supported' = stated or directly entailed by a passage; 'partial' = "
    "some of the claim is stated but a specific detail (a number, regulation, equipment, "
    "requirement) is not; 'unsupported' = not in the passages or contradicted. "
    'Reply with JSON only: {"claims":[{"id":1,"verdict":"supported","passage":2}]} '
    "where passage is the number of the best supporting passage or null."
)
RELEVANCE_SYSTEM = (
    "You grade retrieval for a mine-safety assistant. Given a QUESTION and numbered PASSAGES, mark "
    "a passage relevant only if it contains information that would help answer this specific "
    "question (the cause, the control measure, or the regulation asked about). Passages that only "
    "share generic mining vocabulary are not relevant. "
    'Reply with JSON only: {"passages":[{"id":1,"relevant":true}]}'
)


def _format_passages(passages: Sequence[str]) -> str:
    return "\n\n".join(f"[PASSAGE {i}]\n{p[:CONTEXT_CHARS]}" for i, p in enumerate(passages, 1))


def parse_json(text: str) -> Optional[Dict[str, Any]]:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def judge_claims(judge, passages: Sequence[str], claims: Sequence[str]) -> List[Dict[str, Any]]:
    """One verdict per claim. Missing or malformed verdicts become 'unjudged' (excluded, counted)."""
    if not claims:
        return []
    user = (_format_passages(passages) + "\n\nCLAIMS:\n"
            + "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1)))
    data = parse_json(judge.complete(GROUNDING_SYSTEM, user)) or {}
    by_id = {}
    for item in data.get("claims", []) if isinstance(data, dict) else []:
        try:
            by_id[int(item.get("id"))] = item
        except (TypeError, ValueError):
            continue
    out = []
    for i, claim in enumerate(claims, 1):
        item = by_id.get(i, {})
        verdict = str(item.get("verdict", "")).lower().strip()
        out.append({
            "claim": claim,
            "verdict": verdict if verdict in VERDICT_SCORE else "unjudged",
            "passage": item.get("passage"),
        })
    return out


def judge_relevance(judge, query: str, passages: Sequence[str]) -> List[Optional[bool]]:
    user = f"QUESTION: {query}\n\n" + _format_passages(passages)
    data = parse_json(judge.complete(RELEVANCE_SYSTEM, user)) or {}
    by_id = {}
    for item in data.get("passages", []) if isinstance(data, dict) else []:
        try:
            by_id[int(item.get("id"))] = bool(item.get("relevant"))
        except (TypeError, ValueError):
            continue
    return [by_id.get(i) for i in range(1, len(passages) + 1)]


# ----------------------------------------------------------------------------
# aggregation
# ----------------------------------------------------------------------------
def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 0) -> Tuple[float, float]:
    vals = list(values)
    if not vals:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(n_boot))
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1])


def pct(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(math.ceil(q * len(s))) - 1)]


def summarise(rows: List[Dict[str, Any]], judge_model: str) -> Dict[str, Any]:
    scored = [r for r in rows if not r["used_fallback"] and r["claims"]]
    strict, lenient, prec = [], [], []
    verdicts = Counter()
    for r in scored:
        judged = [c for c in r["claims"] if c["verdict"] != "unjudged"]
        verdicts.update(c["verdict"] for c in r["claims"])
        if judged:
            strict.append(sum(1 for c in judged if c["verdict"] == "supported") / len(judged))
            lenient.append(sum(VERDICT_SCORE[c["verdict"]] for c in judged) / len(judged))
    for r in rows:
        if not r["used_fallback"]:
            flags = [f for f in r["passage_relevant"] if f is not None]
            if flags:
                prec.append(sum(flags) / len(flags))
    stat = [r["stat_consistency"] for r in scored if r["stat_consistency"] is not None]
    gen_ms = [r["generation_ms"] for r in rows if not r["used_fallback"]]
    models = Counter(r["generator_model"] for r in rows)
    summary = {
        "n_queries": len(rows),
        "n_scored": len(scored),
        "n_fallback_excluded": sum(1 for r in rows if r["used_fallback"]),
        "generator_models": dict(models),
        "judge_model": judge_model,
        "judge_same_as_generator": judge_model in models,
        "claims_judged": sum(verdicts[v] for v in VERDICT_SCORE),
        "claims_unjudged": verdicts["unjudged"],
        "verdict_counts": dict(verdicts),
        "grounding_strict": round(sum(strict) / len(strict), 4) if strict else None,
        "grounding_strict_95ci": [round(x, 4) for x in bootstrap_ci(strict)],
        "grounding_lenient": round(sum(lenient) / len(lenient), 4) if lenient else None,
        "passage_precision@5_judged": round(sum(prec) / len(prec), 4) if prec else None,
        "passage_precision@5_95ci": [round(x, 4) for x in bootstrap_ci(prec)],
        "stat_sentence_consistency": round(sum(stat) / len(stat), 4) if stat else None,
        "queries_with_grounding_ge_0.85": round(sum(1 for s in strict if s >= 0.85) / len(strict), 4) if strict else None,
        "generation_ms": {"p50": round(pct(gen_ms, 0.5), 1), "p95": round(pct(gen_ms, 0.95), 1),
                          "mean": round(sum(gen_ms) / len(gen_ms), 1) if gen_ms else 0.0},
    }
    return summary


# ----------------------------------------------------------------------------
# human audit of the judge
# ----------------------------------------------------------------------------
def export_audit(rows: List[Dict[str, Any]], n: int, path: str, seed: int = 0) -> int:
    pool = [(r, c) for r in rows if not r["used_fallback"] for c in r["claims"] if c["verdict"] != "unjudged"]
    random.Random(seed).shuffle(pool)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audit_id", "query_id", "claim", "passages", "judge_verdict", "human_verdict (supported|partial|unsupported)"])
        for i, (r, c) in enumerate(pool[:n], 1):
            w.writerow([i, r["id"], c["claim"], " || ".join(p[:700] for p in r["passages"]), c["verdict"], ""])
    return min(n, len(pool))


def cohens_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    labels = set(a) | set(b)
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def score_audit(path: str) -> Dict[str, Any]:
    judge, human = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            h = (row.get("human_verdict (supported|partial|unsupported)") or "").strip().lower()
            if h in VERDICT_SCORE:
                judge.append(row["judge_verdict"].strip().lower())
                human.append(h)
    binj = ["supported" if j == "supported" else "not" for j in judge]
    binh = ["supported" if h == "supported" else "not" for h in human]
    return {
        "n_labelled": len(human),
        "agreement_3way": round(sum(1 for j, h in zip(judge, human) if j == h) / len(human), 4) if human else None,
        "agreement_supported_vs_not": round(sum(1 for j, h in zip(binj, binh) if j == h) / len(human), 4) if human else None,
        "cohens_kappa_supported_vs_not": round(cohens_kappa(binj, binh), 4) if human else None,
        "judge_too_lenient": sum(1 for j, h in zip(binj, binh) if j == "supported" and h == "not"),
        "judge_too_strict": sum(1 for j, h in zip(binj, binh) if j == "not" and h == "supported"),
    }


# ----------------------------------------------------------------------------
# pipeline
# ----------------------------------------------------------------------------
def build_engine(retriever_type: str, reports_dir: str):
    from evaluation.eval_grounding import FastBM25Retriever
    from evaluation.run_eval import load_chunks, HybridRetriever
    from src.engine import MineMindRAGEngine
    from rank_bm25 import BM25Okapi

    chunks = load_chunks(reports_dir)
    if retriever_type == "hybrid":
        hybrid = HybridRetriever(chunks)
        return MineMindRAGEngine(hybrid.retriever), hybrid.close
    bm25 = BM25Okapi([c["content"].lower().split() for c in chunks])
    return MineMindRAGEngine(FastBM25Retriever(chunks, bm25)), (lambda: None)


def evaluate_query(engine, judge, item: Dict[str, Any]) -> Dict[str, Any]:
    res = engine.answer_query(item["query"])
    tel = res.get("telemetry", {})
    passages = [r["content"] for r in res.get("retrieved", [])]
    claims, stats = split_answer(res.get("answer", ""))
    used_fallback = bool(tel.get("used_fallback"))
    row = {
        "id": item["id"], "type": item.get("type"), "query": item["query"],
        "generator_model": tel.get("llm_used", "unknown"),
        "used_fallback": used_fallback,
        "generation_ms": tel.get("generation_latency_ms", 0.0),
        "llm_attempts": tel.get("llm_attempts", []),
        "answer": res.get("answer", ""),
        "passages": passages,
        "claims": [], "passage_relevant": [], "stat_consistency": None,
    }
    if used_fallback:
        return row  # not an LLM answer: do not grade it as one
    row["claims"] = judge_claims(judge, passages, claims)
    row["passage_relevant"] = judge_relevance(judge, item["query"], passages)
    row["stat_consistency"] = stat_consistency(stats, res.get("risk", {}))
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", default=os.path.join(ROOT, "evaluation", "golden_set_150.json"))
    ap.add_argument("--reports-dir", default=os.path.join(ROOT, "data", "msha_reports"))
    ap.add_argument("--retriever", choices=["bm25", "hybrid"], default="bm25")
    ap.add_argument("--sample", type=int, default=100, help="random in-domain queries to run (seeded)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "openai/gpt-oss-120b"))
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between queries (rate limits)")
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "generation_eval.json"))
    ap.add_argument("--audit-sample", type=int, default=0, help="export N judged claims for hand labelling")
    ap.add_argument("--score-audit", default=None, help="CSV from --audit-sample, filled in by hand")
    ap.add_argument("--min-grounding", type=float, default=None, help="exit 1 if strict grounding is below this")
    args = ap.parse_args(argv)

    audit_csv = os.path.join(os.path.dirname(args.out), "audit_claims.csv")

    if args.score_audit:
        print(json.dumps(score_audit(args.score_audit), indent=2))
        return 0

    if args.audit_sample:
        with open(args.out, encoding="utf-8") as f:
            rows = json.load(f)["rows"]
        n = export_audit(rows, args.audit_sample, audit_csv, args.seed)
        print(f"Wrote {n} claims to {audit_csv}. Fill the last column by hand, then run --score-audit.")
        return 0

    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set: the generator would fall back to extractive text, "
              "which this evaluation refuses to grade as an LLM result.")
        return 2

    with open(args.golden, encoding="utf-8") as f:
        queries = [q for q in json.load(f)["queries"] if q.get("type") in IN_DOMAIN_TYPES]
    random.Random(args.seed).shuffle(queries)
    queries = queries[:args.sample]

    engine, close = build_engine(args.retriever, args.reports_dir)
    judge = GroqJudge(args.judge_model)

    # checkpoint so a rate-limit crash does not lose a run
    ckpt = args.out + ".partial.jsonl"
    done = {}
    if os.path.exists(ckpt):
        with open(ckpt, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done[r["id"]] = r
    rows = []
    try:
        for i, item in enumerate(queries, 1):
            if item["id"] in done:
                rows.append(done[item["id"]])
                continue
            row = evaluate_query(engine, judge, item)
            rows.append(row)
            with open(ckpt, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{i}/{len(queries)}] {item['id']} model={row['generator_model']} "
                  f"claims={len(row['claims'])}", flush=True)
            time.sleep(args.sleep)
    finally:
        close()

    summary = summarise(rows, judge.model)
    summary["retriever"] = args.retriever
    summary["seed"] = args.seed
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)
    print(json.dumps(summary, indent=2))
    if summary["n_fallback_excluded"]:
        print(f"WARNING: {summary['n_fallback_excluded']} queries fell back to extractive text and were excluded.")
    if summary["judge_same_as_generator"]:
        print("WARNING: judge model is also a generator model; use a different one to limit self-preference bias.")
    if args.min_grounding is not None and (summary["grounding_strict"] or 0) < args.min_grounding:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
