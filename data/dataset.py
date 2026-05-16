"""PyTorch Dataset turning JSONL rows into Gemma processor inputs.

Module 2. Slices are read from the offline cache produced by
scripts/preprocess.py (one ``{case_id}_{sequence}.npz`` per sample) so
training I/O does not re-run NIfTI decoding. Labels mask the prompt with
-100 so loss is computed only on the assistant report.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.prompt_builder import build_chat_messages
from data.slice_extractor import extract_slices

logger = logging.getLogger(__name__)

# Tensors that come back from the processor with a leading batch dim of 1.
_SEQ_KEYS = ("input_ids", "attention_mask", "token_type_ids", "labels")


def _cache_path(cache_dir: Path, case_id: str, sequence: str) -> Path:
    return cache_dir / f"{case_id}_{sequence}.npz"


def load_or_extract_slices(
    row: dict,
    cache_dir: Path,
    n_slices: int,
    cache_slices: bool = True,
    cache_compressed: bool = False,
):
    """Return ``(list[PIL.Image], list[int])`` from cache, extracting on miss."""
    cpath = _cache_path(cache_dir, row["case_id"], row["sequence"])
    if cache_slices and cpath.exists():
        with np.load(cpath) as data:
            arr = data["slices"]  # (n, H, W, 3) uint8
            z_indices = data["z_indices"].tolist()
        imgs = [Image.fromarray(arr[i], mode="RGB") for i in range(arr.shape[0])]
        return imgs, z_indices

    imgs, z_indices = extract_slices(
        row["nifti_path"], row.get("mask_path"), n_slices=n_slices
    )
    if cache_slices:
        cache_dir.mkdir(parents=True, exist_ok=True)
        save = np.savez_compressed if cache_compressed else np.savez
        save(
            cpath,
            slices=np.stack([np.asarray(im) for im in imgs]),
            z_indices=np.asarray(z_indices, dtype=np.int32),
        )
    return imgs, z_indices


class GBMSliceDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        processor,
        n_slices: int = 16,
        total_slices: int = 144,
        max_soft_tokens: int = 280,
        cache_slices: bool = True,
        cache_dir: str = "./slice_cache",
        cache_compressed: bool = False,
        is_training: bool = True,
    ):
        self.processor = processor
        self.n_slices = n_slices
        self.total_slices = total_slices
        self.max_soft_tokens = max_soft_tokens
        self.cache_slices = cache_slices
        self.cache_compressed = cache_compressed
        self.cache_dir = Path(cache_dir)
        self.is_training = is_training

        self.rows: List[dict] = []
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        logger.info("Loaded %d samples from %s", len(self.rows), jsonl_path)

    def __len__(self) -> int:
        return len(self.rows)

    def _apply_template(self, messages: List[dict], add_generation_prompt: bool):
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            max_soft_tokens=self.max_soft_tokens,
        )

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        imgs, z_indices = load_or_extract_slices(
            row, self.cache_dir, self.n_slices, self.cache_slices,
            self.cache_compressed,
        )

        messages = build_chat_messages(
            imgs,
            z_indices,
            self.total_slices,
            row["sequence"],
            row.get("report", ""),
            is_training=self.is_training,
        )
        enc = self._apply_template(messages, add_generation_prompt=False)
        # Only sequence tensors carry a leading batch-of-1 to squeeze.
        # Multimodal tensors (e.g. pixel_values) are already
        # (num_images, ...) and must be kept whole for the collator.
        item = {
            k: (v[0] if k in _SEQ_KEYS else v) for k, v in enc.items()
        }

        if self.is_training:
            input_ids = item["input_ids"]
            labels = input_ids.clone()
            # Reliable boundary for multi-image chat templates: re-render the
            # same conversation without the assistant turn (generation prompt
            # on). Image-token expansion is identical, so its length is the
            # exact count of prompt tokens to mask.
            prompt_msgs = [m for m in messages if m["role"] != "assistant"]
            prompt_enc = self._apply_template(
                prompt_msgs, add_generation_prompt=True
            )
            prompt_len = prompt_enc["input_ids"].shape[1]
            labels[:prompt_len] = -100
            # Never train on padding (none here; collator pads later).
            item["labels"] = labels

        return item
