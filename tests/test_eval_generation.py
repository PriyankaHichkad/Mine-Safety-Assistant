import csv
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation import eval_generation as eg


class StubJudge:
    """Returns canned JSON; records prompts so tests can check what the judge was shown."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, system, user):
        self.prompts.append((system, user))
        return self.replies.pop(0)


def test_split_answer_separates_statistics_and_drops_citations():
    text = ("Incidents were caused by missing berms at the tipping edge. "
            "Matching hazard cases account for 12 incidents, an Accident Occurrence Probability of 0.9% (12/1324).\n\n"
            "Citations:\n• [Source: MSHA-FAT-001]")
    claims, stats = eg.split_answer(text)
    assert claims == ["Incidents were caused by missing berms at the tipping edge."]
    assert len(stats) == 1 and "0.9%" in stats[0]


def test_parse_json_handles_fences_and_think_blocks():
    raw = '<think>reasoning {not json}</think>\n```json\n{"claims":[{"id":1,"verdict":"supported"}]}\n```'
    assert eg.parse_json(raw)["claims"][0]["verdict"] == "supported"
    assert eg.parse_json("no json here") is None


def test_judge_claims_marks_missing_and_invalid_verdicts_unjudged():
    judge = StubJudge(['{"claims":[{"id":1,"verdict":"supported","passage":1},{"id":3,"verdict":"maybe"}]}'])
    out = eg.judge_claims(judge, ["p1"], ["claim one here ok", "claim two here ok", "claim three here ok"])
    assert [c["verdict"] for c in out] == ["supported", "unjudged", "unjudged"]


def test_judge_sees_passages_not_the_answer_generator_context_leak():
    judge = StubJudge(['{"claims":[]}'])
    eg.judge_claims(judge, ["alpha passage"], ["some claim about alpha ok"])
    system, user = judge.prompts[0]
    assert "ONLY the passages" in system and "[PASSAGE 1]" in user and "alpha passage" in user


def test_stat_consistency_flags_invented_numbers():
    risk = {"prob_accident_occurrence": 0.9, "total_reports": 1324, "hazard_matches": 12}
    ok = ["Probability of 0.9% (12/1324) across the corpus."]
    bad = ["Probability of 7.5% (99/1324) across the corpus."]
    assert eg.stat_consistency(ok, risk) == 1.0
    assert eg.stat_consistency(bad, risk) < 1.0
    assert eg.stat_consistency([], risk) is None


def _row(i, fallback, verdicts, rel):
    return {"id": f"q{i}", "used_fallback": fallback, "generator_model": "extractive_fallback" if fallback else "m1",
            "generation_ms": 100.0, "stat_consistency": 1.0,
            "claims": [{"claim": "c", "verdict": v} for v in verdicts], "passage_relevant": rel}


def test_summary_excludes_fallback_rows_from_headline():
    rows = [_row(1, False, ["supported", "supported", "unsupported", "supported"], [True] * 5),
            _row(2, True, ["supported"] * 4, [True] * 5)]           # fallback must not inflate the score
    s = eg.summarise(rows, "judge-x")
    assert s["n_scored"] == 1 and s["n_fallback_excluded"] == 1
    assert s["grounding_strict"] == 0.75
    assert s["judge_same_as_generator"] is False


def test_unjudged_claims_do_not_count_as_supported_or_unsupported():
    s = eg.summarise([_row(1, False, ["supported", "unjudged"], [True, False])], "j")
    assert s["grounding_strict"] == 1.0 and s["claims_unjudged"] == 1
    assert s["passage_precision@5_judged"] == 0.5


def test_evaluate_query_does_not_judge_fallback_answers():
    class FakeEngine:
        def answer_query(self, q):
            return {"answer": "x", "risk": {}, "retrieved": [{"content": "p"}],
                    "telemetry": {"used_fallback": True, "llm_used": "extractive_fallback"}}
    judge = StubJudge([])
    row = eg.evaluate_query(FakeEngine(), judge, {"id": "q1", "query": "q", "type": "cause"})
    assert row["used_fallback"] and row["claims"] == [] and judge.prompts == []


def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    vals = [0.5, 0.75, 1.0, 1.0, 0.25, 0.9]
    lo, hi = eg.bootstrap_ci(vals)
    assert lo <= sum(vals) / len(vals) <= hi
    assert eg.bootstrap_ci(vals) == (lo, hi)


def test_kappa_and_audit_roundtrip(tmp_path):
    assert eg.cohens_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert abs(eg.cohens_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"])) < 1e-9
    rows = [dict(_row(i, False, ["supported", "unsupported"], [True]), passages=["ctx"]) for i in range(3)]
    path = str(tmp_path / "audit.csv")
    assert eg.export_audit(rows, 4, path) == 4
    with open(path, newline="", encoding="utf-8") as f:
        data = list(csv.reader(f))
    header, body = data[0], data[1:]
    for r in body:                      # a human who agrees with the judge on every claim
        r[-1] = r[4]
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header] + body)
    res = eg.score_audit(path)
    assert res["n_labelled"] == 4 and res["agreement_3way"] == 1.0 and res["cohens_kappa_supported_vs_not"] == 1.0
