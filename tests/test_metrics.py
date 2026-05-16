"""Tests for eval/metrics.py.

ROUGE-L is skipped if rouge-score is unavailable (it has no prebuilt
wheel here); the lazy imports keep the module usable regardless.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import corpus_bleu4, evaluate  # noqa: E402

_HAS_ROUGE = importlib.util.find_spec("rouge_score") is not None


def test_bleu_identical_is_one():
    assert corpus_bleu4(["a b c d e"], ["a b c d e"]) == pytest.approx(1.0)


def test_bleu_disjoint_is_low():
    assert corpus_bleu4(["a b c d e"], ["x y z w q"]) < 0.05


def test_evaluate_reads_jsonl(tmp_path):
    p = tmp_path / "preds.jsonl"
    rows = [
        {"reference": "tumor in left frontal lobe", "prediction": "tumor in left frontal lobe"},
        {"reference": "enhancing mass right", "prediction": "enhancing mass right"},
    ]
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    res = evaluate(str(p))
    assert res["n"] == 2
    # Identical refs/hyps -> high BLEU; not exactly 1.0 because the
    # 3-word sample has no 4-grams and smoothing applies.
    assert res["bleu4"] > 0.85
    if _HAS_ROUGE:
        assert res["rougeL"] == pytest.approx(1.0, abs=1e-6)
    else:
        assert res["rougeL"] is None


@pytest.mark.skipif(not _HAS_ROUGE, reason="rouge-score not installed")
def test_rouge_identical_is_one():
    from eval.metrics import mean_rouge_l

    assert mean_rouge_l(["a b c"], ["a b c"]) == pytest.approx(1.0)
