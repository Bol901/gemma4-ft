"""Structure tests for data/prompt_builder.py.

Transformers / the Gemma processor are not installable in this
environment, so we validate the message structure (roles, content
ordering, image count, train vs. inference) rather than tokenization.
"""

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.prompt_builder import build_chat_messages  # noqa: E402


def _imgs(n):
    return [Image.new("RGB", (8, 8)) for _ in range(n)]


def test_training_message_has_assistant_turn():
    n = 16
    msgs = build_chat_messages(
        _imgs(n), list(range(n)), 144, "t1c", "A mass.", is_training=True
    )
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[2]["content"][0]["text"] == "A mass."


def test_inference_message_omits_assistant_turn():
    n = 4
    msgs = build_chat_messages(
        _imgs(n), list(range(n)), 144, "flair", "", is_training=False
    )
    assert [m["role"] for m in msgs] == ["system", "user"]


def test_image_count_and_interleave_order():
    n = 16
    z = list(range(100, 100 - n, -1))
    msgs = build_chat_messages(_imgs(n), z, 144, "t2", "r", is_training=True)
    user = msgs[1]["content"]
    images = [c for c in user if c["type"] == "image"]
    assert len(images) == n
    # First item is the intro text, last is the instruction.
    assert user[0]["type"] == "text" and "axial slices" in user[0]["text"]
    assert user[-1] == {"type": "text", "text": "Describe the findings."}
    # Each image is immediately preceded by its "Slice k (z=..)" label.
    for i, c in enumerate(user):
        if c["type"] == "image":
            assert user[i - 1]["type"] == "text"
            assert user[i - 1]["text"].startswith("Slice ")
    assert "T2" in user[0]["text"]  # sequence upper-cased


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        build_chat_messages(_imgs(3), [1, 2], 144, "t1", "r")


def test_training_requires_report():
    with pytest.raises(ValueError):
        build_chat_messages(_imgs(2), [1, 2], 144, "t1", "", is_training=True)
