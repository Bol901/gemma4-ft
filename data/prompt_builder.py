"""Build Gemma chat messages for the MRI captioning task.

Module 3. Produces the message list consumed by
``processor.apply_chat_template``. Kept dependency-free (only PIL typing)
so the message structure can be unit tested without the model.
"""

from __future__ import annotations

from typing import List, Sequence

from PIL import Image

SYSTEM_PROMPT = (
    "You are an expert neuroradiologist. Given axial slices from a brain MRI, "
    "describe the findings in clinical radiological language."
)


def build_chat_messages(
    slices: Sequence[Image.Image],
    z_indices: Sequence[int],
    total_slices: int,
    sequence_type: str,
    report: str,
    is_training: bool = True,
) -> List[dict]:
    """Build messages in Gemma chat format.

    System prompt explains the task; the user turn interleaves an intro,
    per-slice text labels with their images, and a final instruction. The
    assistant turn (the target report) is appended only for training.
    """
    if len(slices) != len(z_indices):
        raise ValueError(
            f"slices ({len(slices)}) and z_indices ({len(z_indices)}) "
            "must have equal length"
        )

    user_content: List[dict] = [
        {
            "type": "text",
            "text": (
                f"Below are {len(slices)} axial slices from a "
                f"{sequence_type.upper()} MRI sequence, ordered from superior "
                f"to inferior. The full volume has {total_slices} slices; "
                f"selected z-positions: {list(z_indices)}."
            ),
        }
    ]
    for i, (img, z) in enumerate(zip(slices, z_indices)):
        user_content.append({"type": "text", "text": f"Slice {i + 1} (z={z}):"})
        user_content.append({"type": "image", "image": img})
    user_content.append({"type": "text", "text": "Describe the findings."})

    messages: List[dict] = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]
    if is_training:
        if not report:
            raise ValueError("report must be non-empty when is_training=True")
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": report}],
            }
        )
    return messages
