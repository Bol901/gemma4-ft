"""Tests for data/dataset.py label masking and data/collator.py padding."""

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests._fakes import EOT, FakeProcessor, MODEL_TOK  # noqa: E402
from data.collator import DataCollatorForGemma4  # noqa: E402
from data.dataset import GBMSliceDataset  # noqa: E402


def _make_case(tmp_path: Path, cid: str, report: str) -> dict:
    vol = np.zeros((32, 32, 40), dtype=np.float32)
    vol[8:24, 8:24, 8:32] = 500.0
    p = tmp_path / f"{cid}.nii.gz"
    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(p))
    return {
        "case_id": cid,
        "sequence": "t1c",
        "nifti_path": str(p),
        "report": report,
    }


def _jsonl(tmp_path: Path, rows) -> str:
    p = tmp_path / "data.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(p)


def test_label_masking_only_on_assistant(tmp_path):
    rows = [_make_case(tmp_path, "c1", "tumor in left frontal lobe")]
    ds = GBMSliceDataset(
        _jsonl(tmp_path, rows),
        FakeProcessor(),
        n_slices=4,
        cache_slices=True,
        cache_dir=str(tmp_path / "cache"),
        is_training=True,
    )
    item = ds[0]
    ids, labels = item["input_ids"], item["labels"]
    assert ids.shape == labels.shape
    masked = labels == -100
    # Exactly the report tokens (+ trailing EOT) are supervised.
    supervised = labels[~masked]
    n_report_words = len("tumor in left frontal lobe".split())
    assert supervised.numel() == n_report_words + 1  # words + EOT
    # The token right before the first supervised position is MODEL_TOK,
    # i.e. masking ends exactly at the assistant content boundary.
    first_sup = (~masked).nonzero()[0].item()
    assert ids[first_sup - 1].item() == MODEL_TOK
    assert ids[-1].item() == EOT
    # Caching round-trips.
    assert (tmp_path / "cache" / "c1_t1c.npz").exists()
    item2 = ds[0]
    assert torch.equal(item["input_ids"], item2["input_ids"])


def test_inference_dataset_has_no_labels(tmp_path):
    rows = [_make_case(tmp_path, "c2", "")]
    ds = GBMSliceDataset(
        _jsonl(tmp_path, rows),
        FakeProcessor(),
        n_slices=4,
        cache_dir=str(tmp_path / "cache"),
        is_training=False,
    )
    assert "labels" not in ds[0]


def test_collator_pads_and_stacks(tmp_path):
    rows = [
        _make_case(tmp_path, "a", "short report"),
        _make_case(tmp_path, "b", "a considerably longer radiology report here"),
    ]
    ds = GBMSliceDataset(
        _jsonl(tmp_path, rows),
        FakeProcessor(),
        n_slices=4,
        cache_dir=str(tmp_path / "cache"),
        is_training=True,
    )
    coll = DataCollatorForGemma4(pad_token_id=0)
    batch = coll([ds[0], ds[1]])

    L = batch["input_ids"].shape[1]
    assert batch["input_ids"].shape == (2, L)
    assert batch["attention_mask"].shape == (2, L)
    assert batch["labels"].shape == (2, L)
    # Padded positions: attention 0, labels -100.
    pad_mask = batch["input_ids"] == 0
    assert torch.all(batch["attention_mask"][pad_mask] == 0)
    assert torch.all(batch["labels"][pad_mask] == -100)
    # pixel_values concatenated across the batch: 4 + 4 images.
    assert batch["pixel_values"].shape[0] == 8
    assert batch["token_type_ids"].shape == (2, L)
