"""Caption metrics: corpus BLEU-4 and ROUGE-L over a predictions JSONL.

    python eval/metrics.py --preds preds.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Dict, List

logger = logging.getLogger("metrics")


def _tokenize(s: str) -> List[str]:
    return s.lower().split()


def corpus_bleu4(refs: List[str], hyps: List[str]) -> float:
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    references = [[_tokenize(r)] for r in refs]
    hypotheses = [_tokenize(h) for h in hyps]
    return float(
        corpus_bleu(
            references,
            hypotheses,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=SmoothingFunction().method1,
        )
    )


def mean_rouge_l(refs: List[str], hyps: List[str]) -> float:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)
    ]
    return float(sum(scores) / max(len(scores), 1))


def evaluate(preds_path: str) -> Dict[str, float]:
    refs, hyps = [], []
    with open(preds_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            refs.append(o["reference"])
            hyps.append(o["prediction"])
    try:
        rouge = mean_rouge_l(refs, hyps)
    except ImportError:
        logger.warning("rouge-score not installed; reporting rougeL=null")
        rouge = None
    return {
        "n": len(refs),
        "bleu4": corpus_bleu4(refs, hyps),
        "rougeL": rouge,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    args = ap.parse_args()
    results = evaluate(args.preds)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
