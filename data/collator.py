"""Custom collator for the Gemma multi-image batch.

Pads ``input_ids`` / ``attention_mask`` / ``labels`` to the batch max
length (left padding kept off; loss positions are -100 anyway) and
stacks the per-sample multimodal tensors. ``remove_unused_columns=False``
must be set on TrainingArguments or ``pixel_values`` is dropped before
this runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch


@dataclass
class DataCollatorForGemma4:
    pad_token_id: int
    label_pad_token_id: int = -100
    # Keys stacked as-is (one tensor per sample, equal trailing shape).
    multimodal_keys: tuple = ("pixel_values", "token_type_ids", "image_grid_thw")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(f["input_ids"].shape[0] for f in features)
        batch: Dict[str, Any] = {}

        input_ids, attn, labels = [], [], []
        has_labels = "labels" in features[0]
        for f in features:
            ids = f["input_ids"]
            n = max_len - ids.shape[0]
            pad = lambda t, v: torch.cat(  # noqa: E731
                [t, torch.full((n,), v, dtype=t.dtype)]
            ) if n > 0 else t
            input_ids.append(pad(ids, self.pad_token_id))
            am = f.get("attention_mask", torch.ones_like(ids))
            attn.append(pad(am, 0))
            if has_labels:
                labels.append(pad(f["labels"], self.label_pad_token_id))

        batch["input_ids"] = torch.stack(input_ids)
        batch["attention_mask"] = torch.stack(attn)
        if has_labels:
            batch["labels"] = torch.stack(labels)

        for key in self.multimodal_keys:
            if key in features[0]:
                vals = [f[key] for f in features]
                if key == "token_type_ids":
                    vals = [
                        torch.cat(
                            [
                                v,
                                torch.zeros(
                                    max_len - v.shape[0], dtype=v.dtype
                                ),
                            ]
                        )
                        if v.shape[0] < max_len
                        else v
                        for v in vals
                    ]
                    batch[key] = torch.stack(vals)
                else:
                    batch[key] = torch.cat(vals, dim=0)

        return batch
