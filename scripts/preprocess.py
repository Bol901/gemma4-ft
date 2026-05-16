"""Offline slice extraction + caching (Module 8).

Reads a training JSONL, extracts N slices per sample with a process pool
and writes one ``{case_id}_{sequence}.npz`` per sample into the cache
dir. Re-running skips samples already cached.

Usage:
    python scripts/preprocess.py \
        --input_jsonl data/train.jsonl \
        --cache_dir /scratch/slice_cache \
        --n_slices 16 --n_workers 16
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.slice_extractor import extract_slices  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("preprocess")


def _process_one(
    row: dict, cache_dir: str, n_slices: int, compressed: bool
) -> tuple:
    cache_dir = Path(cache_dir)
    out = cache_dir / f"{row['case_id']}_{row['sequence']}.npz"
    if out.exists():
        return row["case_id"], "skipped"
    try:
        imgs, z = extract_slices(
            row["nifti_path"], row.get("mask_path"), n_slices=n_slices
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        save = np.savez_compressed if compressed else np.savez
        save(
            out,
            slices=np.stack([np.asarray(im) for im in imgs]),
            z_indices=np.asarray(z, dtype=np.int32),
        )
        return row["case_id"], "ok"
    except Exception as e:  # noqa: BLE001
        return row["case_id"], f"error: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--n_slices", type=int, default=16)
    ap.add_argument("--n_workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="0 = all (use for a 10-case smoke test)")
    ap.add_argument(
        "--compressed",
        action="store_true",
        help="zlib-compress the cache (smaller on disk, slower dataloader). "
        "Default off for training-I/O speed; must match data.cache_compressed.",
    )
    args = ap.parse_args()

    rows = []
    with open(args.input_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]
    logger.info("Processing %d samples with %d workers", len(rows), args.n_workers)

    counts = {"ok": 0, "skipped": 0, "error": 0}
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futs = [
            ex.submit(
                _process_one, r, args.cache_dir, args.n_slices,
                args.compressed,
            )
            for r in rows
        ]
        for i, fut in enumerate(as_completed(futs), 1):
            cid, status = fut.result()
            key = "error" if status.startswith("error") else status
            counts[key] += 1
            if status.startswith("error"):
                logger.error("%s -> %s", cid, status)
            if i % 100 == 0 or i == len(futs):
                logger.info("[%d/%d] %s", i, len(futs), counts)

    logger.info("Done: %s", counts)
    if counts["error"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
