"""Main training entrypoint (Module 5).

Loads hyperparameters from a YAML config (nothing hard-coded), builds the
dataset / collator / LoRA model, runs a single forward+backward dry run
to fail fast, then hands off to the HF Trainer (DeepSpeed ZeRO-2).

    accelerate launch --config_file ... training/train.py \
        --config configs/train_config.yaml [--overfit]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from transformers import Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.collator import DataCollatorForGemma4  # noqa: E402
from data.dataset import GBMSliceDataset  # noqa: E402
from model.load_model import load_model_with_lora  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("train")

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _enable_fast_math() -> None:
    """Allow TF32 for any *fp32* matmul/conv that isn't autocast to bf16.

    Note: in this bf16 + DeepSpeed run almost nothing hits an fp32 matmul,
    so this is near-zero gain in practice -- kept only as harmless hygiene
    (it has no effect on the bf16 numeric path). The real speedups are
    no-grad-checkpointing / FA2 / Liger / batch size / torch.compile.
    """
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _liger_supported(enabled: bool) -> bool:
    """Use Liger only if requested AND the package is importable."""
    if not enabled:
        return False
    try:
        import liger_kernel  # noqa: F401
    except ImportError:
        logger.warning(
            "use_liger_kernel=true but liger-kernel not installed; "
            "continuing without it."
        )
        return False
    return True


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _get_pad_token_id(processor) -> int:
    tok = getattr(processor, "tokenizer", processor)
    pad = getattr(tok, "pad_token_id", None)
    if pad is None:
        pad = getattr(tok, "eos_token_id", 0)
    return pad


def dry_run(model, dataset, collator) -> None:
    """One forward + backward on a 1-sample batch before the real loop."""
    logger.info("Dry run: single forward + backward...")
    batch = collator([dataset[0]])
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}
    model.train()
    out = model(**batch)
    loss = out.loss
    logger.info("Dry-run loss: %.4f", loss.item())
    loss.backward()
    model.zero_grad(set_to_none=True)
    logger.info("Dry run OK.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_config.yaml")
    ap.add_argument(
        "--overfit",
        action="store_true",
        help="Train on the first 10 train samples (pipeline sanity check).",
    )
    ap.add_argument("--no_dry_run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    m, lo, d, tr = cfg["model"], cfg["lora"], cfg["data"], cfg["training"]

    _enable_fast_math()

    model, processor = load_model_with_lora(
        model_id=m["model_id"],
        lora_r=lo["r"],
        lora_alpha=lo["alpha"],
        lora_dropout=lo["dropout"],
        target_modules=lo["target_modules"],
        train_vision_encoder=m["train_vision_encoder"],
        train_projector=m["train_projector"],
        use_qlora=m["use_qlora"],
        torch_dtype=_DTYPES[m["torch_dtype"]],
        gradient_checkpointing=tr["gradient_checkpointing"],
        attn_implementation=m.get("attn_implementation", "sdpa"),
    )

    _ds_kw = dict(
        n_slices=d["n_slices"],
        total_slices=d["total_slices"],
        max_soft_tokens=d["max_soft_tokens"],
        cache_slices=d["cache_slices"],
        cache_dir=d["cache_dir"],
        cache_compressed=d.get("cache_compressed", False),
        is_training=True,
    )
    train_ds = GBMSliceDataset(d["train_jsonl"], processor, **_ds_kw)
    eval_ds = GBMSliceDataset(d["eval_jsonl"], processor, **_ds_kw)

    if args.overfit:
        train_ds.rows = train_ds.rows[:10]
        eval_ds.rows = train_ds.rows
        tr = {**tr, "num_train_epochs": 50, "eval_steps": 50, "save_steps": 1000}
        logger.info("OVERFIT mode: 10 samples, %d epochs", tr["num_train_epochs"])

    collator = DataCollatorForGemma4(pad_token_id=_get_pad_token_id(processor))

    if not args.no_dry_run:
        dry_run(model, train_ds, collator)

    targs = TrainingArguments(
        output_dir=tr["output_dir"],
        num_train_epochs=tr["num_train_epochs"],
        per_device_train_batch_size=tr["per_device_train_batch_size"],
        per_device_eval_batch_size=tr["per_device_eval_batch_size"],
        gradient_accumulation_steps=tr["gradient_accumulation_steps"],
        gradient_checkpointing=tr["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=float(tr["learning_rate"]),
        warmup_ratio=tr["warmup_ratio"],
        lr_scheduler_type=tr["lr_scheduler_type"],
        bf16=(m["torch_dtype"] == "bfloat16"),
        optim="adamw_torch_fused",
        weight_decay=tr["weight_decay"],
        logging_steps=tr["logging_steps"],
        save_strategy="steps",
        save_steps=tr["save_steps"],
        save_total_limit=tr["save_total_limit"],
        eval_strategy="steps",
        eval_steps=tr["eval_steps"],
        report_to=tr["report_to"],
        deepspeed=tr["deepspeed"],
        remove_unused_columns=False,  # keep pixel_values
        dataloader_num_workers=tr["dataloader_num_workers"],
        dataloader_pin_memory=tr.get("dataloader_pin_memory", True),
        dataloader_persistent_workers=tr.get(
            "dataloader_persistent_workers", False
        ),
        dataloader_prefetch_factor=tr.get("dataloader_prefetch_factor", None),
        torch_compile=tr.get("torch_compile", False),
        use_liger_kernel=_liger_supported(tr.get("use_liger_kernel", False)),
        seed=tr["seed"],
        logging_first_step=True,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(tr["output_dir"])
    processor.save_pretrained(tr["output_dir"])
    logger.info("Training complete; adapter saved to %s", tr["output_dir"])


if __name__ == "__main__":
    main()
