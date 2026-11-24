"""Prepare chosen-only examples and pad them without changing their targets.

Raw string prompts are already-formatted prefixes: they are concatenated with
the chosen text exactly, without inserting separators or special tokens. Chat
records use the tokenizer's own template. Only the literal content of the final
assistant message receives labels; prompt, earlier turns, and template suffixes
remain context. Pretokenized rows provide an escape hatch for custom templates.

There is deliberately no truncation policy here. An overlength example either
raises or is skipped when the caller explicitly asks for ``on_overflow='skip'``.
"""

from __future__ import annotations

import copy
from numbers import Integral
from typing import Any, Iterable, Mapping
import uuid
import warnings


IGNORE_INDEX = -100

_ROLES = {
    "assistant": "assistant", "gpt": "assistant", "bot": "assistant",
    "model": "assistant", "user": "user", "human": "user",
    "system": "system", "developer": "developer", "tool": "tool",
    "function": "function",
}


def _integer_list(value: Any, name: str, *, allow_bool: bool = False) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a flat sequence of integer token values.")
    if any((isinstance(item, bool) and not allow_bool) or not isinstance(item, Integral) for item in value):
        raise ValueError(f"{name} must be a flat sequence of integer token values.")
    return [int(item) for item in value]


def _normalize_messages(value: Any, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list of text messages.")
    normalized = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{name}[{index}] must be a message object.")
        role = raw.get("role", raw.get("from"))
        content = raw.get("content", raw.get("value"))
        if not isinstance(role, str) or role.lower() not in _ROLES:
            raise ValueError(f"{name}[{index}] has an unsupported role: {role!r}.")
        if not isinstance(content, str):
            raise ValueError(f"{name}[{index}].content must be text, not multimodal content.")
        message = {
            key: copy.deepcopy(item) for key, item in raw.items()
            if key not in {"role", "content", "from", "value"}
        }
        message.update(role=_ROLES[role.lower()], content=content)
        normalized.append(message)
    return normalized


def _assistant(chosen: Any, prompt: list[dict[str, Any]]) -> dict[str, Any]:
    if isinstance(chosen, str):
        return {"role": "assistant", "content": chosen}
    if isinstance(chosen, Mapping):
        chosen = [chosen]
    messages = _normalize_messages(chosen, "chosen")
    if prompt and messages[:len(prompt)] == prompt:
        messages = messages[len(prompt):]
    if len(messages) != 1 or messages[0]["role"] != "assistant":
        raise ValueError(
            "chosen must be one assistant response, or a full conversation whose "
            "prefix exactly matches prompt and whose only remaining turn is assistant."
        )
    return messages[0]


def _split_example(example: Mapping[str, Any]) -> tuple[str | list[dict[str, Any]], dict[str, Any]]:
    if "prompt" in example and "chosen" in example:
        prompt = example["prompt"]
        if isinstance(prompt, str):
            return prompt, _assistant(example["chosen"], [])
        prompt = _normalize_messages(prompt, "prompt")
        return prompt, _assistant(example["chosen"], prompt)

    field = next((key for key in ("messages", "conversations", "conversation") if key in example), None)
    # Full chosen conversations already contain their own prompt. Existing
    # rejected data is intentionally neither inspected nor used for training.
    if field is None and isinstance(example.get("chosen"), (list, tuple)):
        field = "chosen"
    if field is None:
        raise ValueError(
            "Expected prompt/chosen, messages, ShareGPT conversations, a full chosen "
            "conversation, or pretokenized input_ids and labels."
        )
    messages = _normalize_messages(example[field], field)
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError(
            "The final message must be an assistant turn; trailing user/tool turns "
            "are not silently discarded."
        )
    if "system" in example:
        system = example["system"]
        if not isinstance(system, str):
            raise ValueError("The top-level system field must be text.")
        if messages[0]["role"] == "system":
            if messages[0]["content"] != system:
                raise ValueError("Top-level system text conflicts with the first system message.")
        else:
            messages.insert(0, {"role": "system", "content": system})
    return messages[:-1], messages[-1]


def _render(prompt: str | list[dict[str, Any]], chosen: dict[str, Any], tokenizer: Any) -> tuple[str, int, int]:
    content = chosen["content"]
    if not content:
        raise ValueError("The chosen assistant content is empty.")
    if isinstance(prompt, str):
        return prompt + content, len(prompt), len(prompt) + len(content)
    if not callable(getattr(tokenizer, "apply_chat_template", None)):
        raise ValueError("Conversational examples require a tokenizer with a chat template.")
    marker = "TOKENPREFERENCECONTENT" + uuid.uuid4().hex + "ENDCONTENT"
    marked_message = {**chosen, "content": marker}
    try:
        marked = tokenizer.apply_chat_template(
            [*prompt, marked_message], tokenize=False, add_generation_prompt=False,
        )
        actual = tokenizer.apply_chat_template(
            [*prompt, chosen], tokenize=False, add_generation_prompt=False,
        )
    except Exception as exc:
        raise ValueError(f"The tokenizer could not render this conversation: {exc}") from exc
    if not isinstance(marked, str) or not isinstance(actual, str) or marked.count(marker) != 1:
        raise ValueError("The chat template must expose one unmodified final assistant-content span.")
    prefix, suffix = marked.split(marker)
    if actual != prefix + content + suffix:
        raise ValueError(
            "The chat template transforms, strips, or rearranges the chosen content. "
            "Provide explicitly tokenized input_ids/labels for this custom format."
        )
    return actual, len(prefix), len(prefix) + len(content)


def _encode(tokenizer: Any, text: str, *, offsets: bool = False) -> Mapping[str, Any]:
    try:
        encoded = tokenizer(
            text, add_special_tokens=False, truncation=False,
            return_attention_mask=True, return_offsets_mapping=offsets,
        )
    except Exception as exc:
        raise ValueError(f"Tokenizer could not encode the chosen-only example: {exc}") from exc
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise ValueError("Tokenizer must return a mapping containing input_ids.")
    return encoded


def _validate_prepared(example: Mapping[str, Any]) -> dict[str, list[int]]:
    if "input_ids" not in example or "labels" not in example:
        raise ValueError("Pretokenized examples require both input_ids and labels.")
    ids = _integer_list(example["input_ids"], "input_ids")
    labels = _integer_list(example["labels"], "labels")
    attention = _integer_list(example.get("attention_mask", [1] * len(ids)), "attention_mask", allow_bool=True)
    if not ids or len(labels) != len(ids) or len(attention) != len(ids):
        raise ValueError("input_ids, labels, and attention_mask must have equal, nonzero lengths.")
    if any(token < 0 for token in ids):
        raise ValueError("input_ids must contain nonnegative token IDs.")
    if any(value not in (0, 1) for value in attention):
        raise ValueError("attention_mask must contain only 0 and 1.")
    if attention != sorted(attention, reverse=True):
        raise ValueError("Only unpadded or right-padded examples are supported.")
    # There is no preceding logit for position zero. Mask it consistently with
    # Transformers' causal shift and token-count normalization.
    labels[0] = IGNORE_INDEX
    for index, (token, label, attended) in enumerate(zip(ids, labels, attention)):
        if not attended:
            labels[index] = IGNORE_INDEX
        elif label != IGNORE_INDEX and label != token:
            raise ValueError(
                f"labels[{index}] differs from input_ids[{index}]. All supervised "
                "targets must be the original chosen tokens."
            )
    if not any(label != IGNORE_INDEX for label in labels[1:]):
        raise ValueError("The example has no predictable chosen target tokens.")
    return {"input_ids": ids, "attention_mask": attention, "labels": labels}


def prepare_example(
    example: Mapping[str, Any],
    tokenizer: Any = None,
    *,
    max_length: int | None = None,
    on_overflow: str = "error",
) -> dict[str, list[int]] | None:
    """Return a chosen-only tokenized row, or ``None`` for explicit overflow skips.

    Raw examples require fast-tokenizer character offsets. A token that crosses
    the prompt/content or content/template-suffix boundary causes a clear error;
    no partial token is labeled and no boundary is guessed. A supplied labels
    column must use token IDs equal to input_ids, or ``-100`` for masked positions.
    Existing pretokenized labels at position zero and padding are set to ``-100``.
    """
    if not isinstance(example, Mapping):
        raise ValueError("Each dataset row must be a mapping.")
    if max_length is not None and (
        isinstance(max_length, bool) or not isinstance(max_length, Integral) or max_length < 1
    ):
        raise ValueError("max_length must be a positive integer or None.")
    if on_overflow not in {"error", "skip"}:
        raise ValueError("on_overflow must be 'error' or 'skip'; silent truncation is unsupported.")
    if "input_ids" in example or "labels" in example:
        prepared = _validate_prepared(example)
    else:
        if tokenizer is None:
            raise ValueError("Raw text examples require a tokenizer.")
        if getattr(tokenizer, "is_fast", True) is False:
            raise ValueError("Raw examples require a fast tokenizer with character offsets; use pretokenized rows otherwise.")
        prompt, chosen = _split_example(example)
        text, start, end = _render(prompt, chosen, tokenizer)
        encoded = _encode(tokenizer, text, offsets=True)
        ids = _integer_list(encoded["input_ids"], "tokenizer input_ids")
        offsets = encoded.get("offset_mapping")
        if hasattr(offsets, "tolist"):
            offsets = offsets.tolist()
        if not isinstance(offsets, (list, tuple)) or len(offsets) != len(ids):
            raise ValueError("Tokenizer must provide one character offset for each token.")
        labels = [IGNORE_INDEX] * len(ids)
        selected = []
        previous = (0, 0)
        for index, offset in enumerate(offsets):
            pair = _integer_list(offset, f"offset_mapping[{index}]")
            if len(pair) != 2:
                raise ValueError("Each token offset must contain a character start and end.")
            left, right = pair
            if not 0 <= left < right <= len(text):
                raise ValueError(
                    "Tokenizer returned an empty or out-of-range character offset. "
                    "Use explicit input_ids/labels when offsets cannot identify content."
                )
            if left < previous[0] or right < previous[1]:
                raise ValueError("Tokenizer character offsets must be monotonic.")
            previous = (left, right)
            if left < start < right or left < end < right:
                raise ValueError(
                    "A token crosses the prompt/chosen or chosen/template boundary. "
                    "Use a stable tokenizer boundary or explicitly prepared input_ids/labels."
                )
            if start <= left and right <= end:
                labels[index] = ids[index]
                selected.append(index)
        if not selected or selected[0] == 0:
            raise ValueError(
                "The chosen content must have target tokens preceded by at least one "
                "context token. Supply a prompt/BOS prefix or explicit labels."
            )
        prefix_ids = _integer_list(_encode(tokenizer, text[:start])["input_ids"], "prefix input_ids")
        content_end_ids = _integer_list(_encode(tokenizer, text[:end])["input_ids"], "content-end input_ids")
        if ids[:selected[0]] != prefix_ids or ids[:selected[-1] + 1] != content_end_ids:
            raise ValueError(
                "Joint tokenization changes a prompt/chosen or chosen/template boundary. "
                "No boundary token was guessed, dropped, or partially supervised."
            )
        prepared = _validate_prepared({
            "input_ids": ids,
            "attention_mask": encoded.get("attention_mask", [1] * len(ids)),
            "labels": labels,
        })
    if max_length is not None and len(prepared["input_ids"]) > max_length:
        if on_overflow == "skip":
            return None
        raise ValueError(
            f"Example has {len(prepared['input_ids'])} tokens, exceeding max_length={max_length}; "
            "it was not truncated. Use on_overflow='skip' to explicitly discard it."
        )
    return prepared


def prepare_dataset(
    examples: Iterable[Mapping[str, Any]],
    tokenizer: Any = None,
    *,
    max_length: int | None = None,
    on_overflow: str = "error",
) -> list[dict[str, list[int]]]:
    """Prepare an iterable of rows, adding row indices to validation errors.

    Returns ordinary dictionaries suitable for a PyTorch dataset or
    ``datasets.Dataset.from_list``. Explicit overflow skips are counted in a
    warning; all other invalid rows always raise.
    """
    result = []
    skipped = 0
    for index, example in enumerate(examples):
        try:
            prepared = prepare_example(
                example, tokenizer, max_length=max_length, on_overflow=on_overflow,
            )
        except ValueError as exc:
            raise ValueError(f"Dataset row {index}: {exc}") from exc
        if prepared is None:
            skipped += 1
        else:
            result.append(prepared)
    if skipped:
        warnings.warn(f"Explicitly skipped {skipped} overlength example(s).", UserWarning, stacklevel=2)
    return result


class ChosenDataCollator:
    """Right-pad chosen-only rows and leave the tokenizer configuration untouched.

    ``pad_token_id`` overrides the tokenizer's pad ID. If the tokenizer has no
    pad token, its EOS ID can safely serve as right padding because labels and
    attention are masked by position, never by token identity. With a tokenizer,
    raw rows can also be prepared on demand; preprocessing once is preferable for
    repeated traversals. Position zero is always excluded from causal targets.
    """

    def __init__(
        self,
        tokenizer: Any = None,
        *,
        pad_token_id: int | None = None,
        pad_to_multiple_of: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        if pad_token_id is None and tokenizer is not None:
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, Integral) or pad_token_id < 0:
            raise ValueError("Provide a nonnegative pad_token_id, or a tokenizer with a pad/EOS token ID.")
        if pad_to_multiple_of is not None and (
            isinstance(pad_to_multiple_of, bool)
            or not isinstance(pad_to_multiple_of, Integral)
            or pad_to_multiple_of < 1
        ):
            raise ValueError("pad_to_multiple_of must be a positive integer or None.")
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[Mapping[str, Any]]) -> dict[str, Any]:
        import torch

        if not features:
            raise ValueError("Cannot collate an empty batch.")
        rows = [prepare_example(feature, self.tokenizer) for feature in features]
        length = max(len(row["input_ids"]) for row in rows)
        if self.pad_to_multiple_of is not None:
            multiple = self.pad_to_multiple_of
            length = ((length + multiple - 1) // multiple) * multiple
        ids, attention, labels = [], [], []
        for row in rows:
            missing = length - len(row["input_ids"])
            ids.append(row["input_ids"] + [self.pad_token_id] * missing)
            attention.append(row["attention_mask"] + [0] * missing)
            labels.append(row["labels"] + [IGNORE_INDEX] * missing)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
