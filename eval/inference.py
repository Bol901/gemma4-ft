"""Load a LoRA checkpoint and generate captions (Module 7).

    python eval/inference.py --config configs/train_config.yaml \
        --adapter ./checkpoints --jsonl data/val.jsonl \
        --out preds.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.prompt_builder import build_chat_messages  # noqa: E402
from data.slice_extractor import extract_slices  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("inference")

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_for_inference(model_id: str, adapter_dir: str, torch_dtype):
    processor = AutoProcessor.from_pretrained(adapter_dir, trust_remote_code=True)
    base = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch_dtype, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, processor


@torch.no_grad()
def generate_caption(
    model,
    processor,
    nifti_path,
    sequence_type,
    mask_path=None,
    n_slices=16,
    total_slices=144,
    max_soft_tokens=280,
    max_new_tokens=512,
) -> str:
    slices, z_indices = extract_slices(
        nifti_path, mask_path, n_slices=n_slices
    )
    messages = build_chat_messages(
        slices, z_indices, total_slices, sequence_type, "", is_training=False
    )
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        max_soft_tokens=max_soft_tokens,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    gen = outputs[0][inputs["input_ids"].shape[1]:]
    return processor.decode(gen, skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_config.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", default="preds.jsonl")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    m, d, e = cfg["model"], cfg["data"], cfg["eval"]
    model, processor = load_for_inference(
        m["model_id"], args.adapter, _DTYPES[m["torch_dtype"]]
    )

    with open(args.jsonl) as f, open(args.out, "w") as fo:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pred = generate_caption(
                model, processor,
                row["nifti_path"], row["sequence"], row.get("mask_path"),
                n_slices=d["n_slices"], total_slices=d["total_slices"],
                max_soft_tokens=d["max_soft_tokens"],
                max_new_tokens=e["max_new_tokens"],
            )
            fo.write(json.dumps({
                "case_id": row["case_id"],
                "sequence": row["sequence"],
                "reference": row.get("report", ""),
                "prediction": pred,
            }) + "\n")
            logger.info("%s done", row["case_id"])


if __name__ == "__main__":
    main()
