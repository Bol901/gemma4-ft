"""Load the Gemma multimodal model and attach LoRA.

Module 4. The spec names ``Gemma4ForConditionalGeneration`` /
``Gemma4Processor`` directly, but the exact class names depend on the
installed ``transformers`` build. We load through ``AutoProcessor`` /
``AutoModelForImageTextToText`` with a configurable ``model_id`` so the
code works whether the host ships Gemma 3 or Gemma 4. The LoRA target
modules are verified against the live module names at load time (see the
"common pitfalls" note in IMPLEMENTATION.md).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

logger = logging.getLogger(__name__)

DEFAULT_LLM_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
# SigLIP-style ViT attention names; verified against live modules below.
DEFAULT_VISION_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj"]


def _log_trainable_params(model: torch.nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable params: %s / %s (%.4f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100.0 * trainable / max(total, 1),
    )
    return trainable, total


def _resolve_targets(
    model: torch.nn.Module, requested: List[str]
) -> List[str]:
    """Keep only target suffixes that actually match a Linear module name.

    Guards against the Gemma-3 vs Gemma-4 naming differences called out in
    the spec: a requested suffix that matches nothing is dropped with a
    warning instead of silently training zero adapters.
    """
    present = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            leaf = name.split(".")[-1]
            for t in requested:
                if leaf == t or name.endswith(t):
                    present.add(t)
    missing = sorted(set(requested) - present)
    if missing:
        logger.warning("LoRA targets not found and dropped: %s", missing)
    resolved = sorted(present)
    if not resolved:
        raise ValueError(
            "No LoRA target modules matched the model. Inspect "
            "model.named_modules() and update target_modules."
        )
    logger.info("Resolved LoRA target_modules: %s", resolved)
    return resolved


def load_model_with_lora(
    model_id: str = "google/gemma-4-31b-it",
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    train_vision_encoder: bool = False,
    train_projector: bool = True,
    use_qlora: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
    gradient_checkpointing: bool = True,
    attn_implementation: str = "eager",
):
    """Load the model in bf16 (or 4-bit), freeze it, then attach LoRA.

    Returns ``(peft_model, processor)``.
    """
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    model_kwargs = dict(
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    if use_qlora:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )

    model = AutoModelForImageTextToText.from_pretrained(
        model_id, **model_kwargs
    )

    if use_qlora:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )

    # Step 1: freeze everything.
    for p in model.parameters():
        p.requires_grad = False

    # Step 2: optionally unfreeze the multimodal projector.
    if train_projector:
        projector = getattr(model, "multi_modal_projector", None)
        if projector is None:
            logger.warning(
                "model.multi_modal_projector not found; skipping projector "
                "unfreeze (verify the attribute name on this model)."
            )
        else:
            for p in projector.parameters():
                p.requires_grad = True
            logger.info("Projector set trainable.")

    # Step 3: LoRA targets.
    requested = list(target_modules) if target_modules else list(
        DEFAULT_LLM_TARGETS
    )
    if train_vision_encoder:
        requested = sorted(set(requested) | set(DEFAULT_VISION_TARGETS))
    resolved = _resolve_targets(model, requested)

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=resolved,
    )
    model = get_peft_model(model, peft_config)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    _log_trainable_params(model)
    return model, processor
