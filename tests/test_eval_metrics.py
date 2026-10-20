import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.metrics import (
    dedupe, hit_at_k, recall_at_k, precision_at_k, precision_ceiling,
    reciprocal_rank, wilson_interval, mean_confidence_interval,
)


def test_dedupe_keeps_first_occurrence_order():
    assert dedupe(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_hit_and_reciprocal_rank():
    ranked = ["x", "y", "z", "w", "v"]
    assert hit_at_k({"z"}, ranked, 5) == 1.0
    assert hit_at_k({"z"}, ranked, 2) == 0.0
    assert reciprocal_rank({"z"}, ranked, 5) == pytest.approx(1 / 3)
    assert reciprocal_rank({"q"}, ranked, 5) == 0.0
    # first relevant result decides the rank
    assert reciprocal_rank({"y", "z"}, ranked, 5) == pytest.approx(1 / 2)


def test_recall_and_precision():
    ranked = ["a", "b", "c", "d", "e"]
    rel = {"a", "c", "z"}
    assert recall_at_k(rel, ranked, 5) == pytest.approx(2 / 3)
    assert precision_at_k(rel, ranked, 5) == pytest.approx(2 / 5)
    assert precision_ceiling(rel, 5) == pytest.approx(3 / 5)
    assert precision_ceiling({"a"}, 5) == pytest.approx(1 / 5)


def test_recall_requires_relevant_docs():
    with pytest.raises(ValueError):
        recall_at_k(set(), ["a"], 5)


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(57, 60)
    assert 0.0 <= lo < 57 / 60 < hi <= 1.0
    assert wilson_interval(0, 0) == (0.0, 0.0)
    lo, hi = wilson_interval(10, 10)
    assert hi == pytest.approx(1.0) and lo < 1.0


def test_mean_confidence_interval():
    vals = [0.95, 0.96, 0.94, 0.98, 0.97]
    mean = sum(vals) / len(vals)
    lo, hi = mean_confidence_interval(vals)
    assert lo <= mean <= hi
    assert 0.0 <= lo and hi <= 1.0
    assert mean_confidence_interval([]) == (0.0, 0.0)
    assert mean_confidence_interval([0.8]) == (0.8, 0.8)
