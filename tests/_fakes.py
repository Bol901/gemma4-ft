"""Shared test fixtures: a deterministic fake Gemma processor.

A real Gemma checkpoint cannot be downloaded in this environment, so the
fake reproduces the token-layout contract the dataset relies on:
re-rendering the conversation without the assistant turn (generation
prompt on) yields exactly the prefix length to mask with -100.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IMG_TOKENS = 4
BOS, SOT, MODEL_TOK, EOT = 1, 2, 3, 4


class _Tok:
    pad_token_id = 0
    eos_token_id = 9


class FakeProcessor:
    """Mimics processor.apply_chat_template return_dict=True."""

    tokenizer = _Tok()

    def apply_chat_template(
        self,
        messages,
        add_generation_prompt=False,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        max_soft_tokens=280,
    ):
        ids = [BOS]
        n_images = 0
        for m in messages:
            if m["role"] == "assistant":
                ids += [SOT, MODEL_TOK]
                for c in m["content"]:
                    ids += [100 + i for i in range(len(c["text"].split()))]
                ids += [EOT]
                continue
            for c in m["content"]:
                if c["type"] == "text":
                    ids += [50 + i for i in range(len(c["text"].split()))]
                else:
                    ids += [7] * IMG_TOKENS
                    n_images += 1
        if add_generation_prompt:
            ids += [SOT, MODEL_TOK]

        input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        out = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "token_type_ids": torch.zeros_like(input_ids),
        }
        if n_images:
            out["pixel_values"] = torch.zeros(
                n_images, 3, 16, 16, dtype=torch.float32
            )
        return out
