"""Data contracts: chosen context is immutable and only intended content is labeled."""

import copy

import pytest

from token_preference.data import ChosenDataCollator, prepare_dataset, prepare_example


class CharacterTokenizer:
    """A deterministic, Unicode-aware tokenizer requiring no network fixtures."""

    is_fast = True
    pad_token_id = 0
    eos_token_id = 7

    def __call__(self, text, *, add_special_tokens, truncation, return_attention_mask, return_offsets_mapping):
        assert add_special_tokens is False
        assert truncation is False
        result = {"input_ids": [ord(char) + 10 for char in text], "attention_mask": [1] * len(text)}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is False
        rendered = "<B>"
        for message in messages:
            name = ":" + message["name"] if "name" in message else ""
            reasoning = "<think>" + message["reasoning_content"] + "</think>" if "reasoning_content" in message else ""
            rendered += "[" + message["role"] + name + "]" + reasoning + message["content"] + "<E>"
        return rendered


def chosen_text(row):
    return "".join(chr(label - 10) for label in row["labels"] if label != -100)


def input_text(row):
    return "".join(chr(token - 10) for token in row["input_ids"])


@pytest.fixture
def tokenizer():
    return CharacterTokenizer()


def test_raw_prompt_is_exact_prefix_and_rejected_is_ignored(tokenizer):
    example = {"prompt": "Question:\n", "chosen": " answer", "rejected": object()}
    row = prepare_example(example, tokenizer)
    assert input_text(row) == "Question:\n answer"
    assert chosen_text(row) == " answer"
    assert row["labels"][:10] == [-100] * 10


def test_final_assistant_only_and_structural_suffix_masked(tokenizer):
    messages = [
        {"role": "user", "content": "First?"},
        {"role": "assistant", "content": "Earlier answer."},
        {"role": "user", "content": "Second?"},
        {"role": "assistant", "content": "Final answer."},
    ]
    original = copy.deepcopy(messages)
    row = prepare_example({"messages": messages}, tokenizer)
    assert chosen_text(row) == "Final answer."
    assert "Earlier answer." in input_text(row)
    assert row["labels"][-3:] == [-100] * 3
    assert messages == original


def test_sharegpt_roles_and_top_level_system(tokenizer):
    row = prepare_example({"system": "Be precise.", "conversations": [
        {"from": "human", "value": "Hello"},
        {"from": "gpt", "value": "Hello to you."},
    ]}, tokenizer)
    assert input_text(row).startswith("<B>[system]Be precise.<E>[user]")
    assert chosen_text(row) == "Hello to you."


def test_prompt_and_full_chosen_conversation(tokenizer):
    prompt = [{"role": "user", "content": "A?"}]
    row = prepare_example({"prompt": prompt, "chosen": [*prompt, {"role": "assistant", "content": "B."}]}, tokenizer)
    assert chosen_text(row) == "B."
    assert input_text(row).count("[user]") == 1


def test_full_chosen_conversation_does_not_inspect_rejected(tokenizer):
    row = prepare_example({"chosen": [
        {"role": "user", "content": "A?"},
        {"role": "assistant", "content": "B."},
    ], "rejected": "malformed old rejected data is irrelevant"}, tokenizer)
    assert chosen_text(row) == "B."


def test_chosen_message_metadata_retained_in_native_render(tokenizer):
    row = prepare_example({"prompt": [{"role": "user", "content": "A?"}], "chosen": {
        "role": "assistant", "name": "Narrator", "reasoning_content": "Private context.", "content": "Visible text.",
    }}, tokenizer)
    assert "[assistant:Narrator]<think>Private context.</think>Visible text." in input_text(row)
    assert chosen_text(row) == "Visible text."


def test_unicode_and_literal_special_spelling_remain_targets(tokenizer):
    text = "é 🙂 漢字 <E>"
    row = prepare_example({"prompt": "Prefix:", "chosen": text}, tokenizer)
    assert chosen_text(row) == text


@pytest.mark.parametrize("example,pattern", [
    ({"messages": [{"role": "assistant", "content": "A"}, {"role": "user", "content": "B"}]}, "final message"),
    ({"messages": []}, "final message"),
    ({"messages": [{"role": "assistant", "content": []}]}, "must be text"),
    ({"prompt": "Q", "chosen": ""}, "empty"),
    ({"prompt": "", "chosen": "answer"}, "context token"),
    ({"prompt": [{"role": "user", "content": "A"}], "chosen": [{"role": "user", "content": "B"}, {"role": "assistant", "content": "C"}]}, "prefix exactly matches"),
    ({"system": "A", "messages": [{"role": "system", "content": "B"}, {"role": "assistant", "content": "C"}]}, "conflicts"),
])
def test_invalid_raw_examples_fail_explicitly(tokenizer, example, pattern):
    with pytest.raises(ValueError, match=pattern):
        prepare_example(example, tokenizer)


def test_template_that_strips_chosen_is_not_silently_accepted():
    class StrippingTokenizer(CharacterTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return super().apply_chat_template([{**message, "content": message["content"].strip()} for message in messages], **kwargs)

    with pytest.raises(ValueError, match="transforms, strips"):
        prepare_example({"messages": [{"role": "assistant", "content": " padded "}]}, StrippingTokenizer())


def test_boundary_crossing_token_is_rejected():
    class MergingTokenizer(CharacterTokenizer):
        def __call__(self, text, **kwargs):
            if text == "ab":
                return {"input_ids": [500], "attention_mask": [1], "offset_mapping": [(0, 2)]}
            return super().__call__(text, **kwargs)

    with pytest.raises(ValueError, match="token crosses"):
        prepare_example({"prompt": "a", "chosen": "b"}, MergingTokenizer())


def test_joint_tokenization_prefix_change_is_rejected():
    class UnstableTokenizer(CharacterTokenizer):
        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            if text == "a":
                result["input_ids"] = [900]
            return result

    with pytest.raises(ValueError, match="Joint tokenization changes"):
        prepare_example({"prompt": "a", "chosen": "b"}, UnstableTokenizer())


def test_content_suffix_crossing_token_is_rejected():
    class SuffixMergingTokenizer(CharacterTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return "P" + messages[-1]["content"] + "!"

        def __call__(self, text, **kwargs):
            if text == "Pz!":
                return {"input_ids": [90, 500], "attention_mask": [1, 1], "offset_mapping": [(0, 1), (1, 3)]}
            return super().__call__(text, **kwargs)

    with pytest.raises(ValueError, match="token crosses"):
        prepare_example({"messages": [{"role": "assistant", "content": "z"}]}, SuffixMergingTokenizer())


def test_no_silent_truncation_and_explicit_skip(tokenizer):
    example = {"prompt": "Q:", "chosen": "A long answer"}
    with pytest.raises(ValueError, match="was not truncated"):
        prepare_example(example, tokenizer, max_length=5)
    assert prepare_example(example, tokenizer, max_length=5, on_overflow="skip") is None
    with pytest.warns(UserWarning, match="skipped 1"):
        rows = prepare_dataset([example, {"prompt": "Q", "chosen": "A"}], tokenizer, max_length=5, on_overflow="skip")
    assert len(rows) == 1


def test_prepare_dataset_reports_bad_row_index(tokenizer):
    with pytest.raises(ValueError, match="Dataset row 1"):
        prepare_dataset([{"prompt": "Q", "chosen": "A"}, {"prompt": "Q", "chosen": ""}], tokenizer)


def test_pretokenized_rows_preserve_chosen_ids_and_mask_position_zero():
    example = {"input_ids": [4, 5, 6], "labels": [4, 5, 6]}
    row = prepare_example(example)
    assert row == {"input_ids": [4, 5, 6], "labels": [-100, 5, 6], "attention_mask": [1, 1, 1]}
    assert example["labels"] == [4, 5, 6]


def test_boolean_attention_mask_is_supported():
    row = prepare_example({"input_ids": [4, 5, 0], "labels": [-100, 5, -100], "attention_mask": [True, True, False]})
    assert row["attention_mask"] == [1, 1, 0]


@pytest.mark.parametrize("example,pattern", [
    ({"input_ids": [4, 5], "labels": [-100, 7]}, "differs"),
    ({"input_ids": [4, 5]}, "both input_ids and labels"),
    ({"input_ids": [4, 5], "labels": [-100, -100]}, "no predictable"),
    ({"input_ids": [4], "labels": [4]}, "no predictable"),
    ({"input_ids": [0, 4, 5], "labels": [-100, -100, 5], "attention_mask": [0, 1, 1]}, "right-padded"),
    ({"input_ids": [4, 5], "labels": [-100, 5], "attention_mask": [1, 2]}, "only 0 and 1"),
    ({"input_ids": [4, -5], "labels": [-100, -5]}, "nonnegative"),
])
def test_bad_pretokenized_rows_fail(example, pattern):
    with pytest.raises(ValueError, match=pattern):
        prepare_example(example)


def test_pad_equals_eos_masks_only_padding_positions():
    collator = ChosenDataCollator(pad_token_id=7)
    batch = collator([
        {"input_ids": [4, 7, 8], "labels": [-100, 7, 8]},
        {"input_ids": [4, 6], "labels": [-100, 6]},
    ])
    assert batch["input_ids"].tolist() == [[4, 7, 8], [4, 6, 7]]
    assert batch["labels"].tolist() == [[-100, 7, 8], [-100, 6, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]


def test_collator_raw_input_and_padding_multiple(tokenizer):
    batch = ChosenDataCollator(tokenizer, pad_to_multiple_of=4)([
        {"prompt": "Q", "chosen": "AB"}, {"prompt": "Q", "chosen": "A"},
    ])
    assert tuple(batch["input_ids"].shape) == (2, 4)
    assert batch["labels"].tolist() == [[-100, 75, 76, -100], [-100, 75, -100, -100]]


def test_collator_eos_fallback_does_not_modify_tokenizer(tokenizer):
    tokenizer.pad_token_id = None
    collator = ChosenDataCollator(tokenizer)
    assert collator.pad_token_id == tokenizer.eos_token_id
    assert tokenizer.pad_token_id is None


def test_collator_masks_existing_right_padding_and_first_position():
    batch = ChosenDataCollator(pad_token_id=0)([{
        "input_ids": [3, 4, 0], "labels": [3, 4, 0], "attention_mask": [1, 1, 0],
    }])
    assert batch["labels"].tolist() == [[-100, 4, -100]]


@pytest.mark.parametrize("kwargs", [{"pad_token_id": -1}, {"pad_token_id": True}, {"pad_token_id": 0, "pad_to_multiple_of": 0}])
def test_collator_rejects_bad_configuration(kwargs):
    with pytest.raises(ValueError):
        ChosenDataCollator(**kwargs)


def test_empty_batch_fails():
    with pytest.raises(ValueError, match="empty batch"):
        ChosenDataCollator(pad_token_id=0)([])


def test_real_fast_tokenizer_chat_alignment_when_available():
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0, "[B]": 1, "[E]": 2, "user": 3, "assistant": 4, "Question": 5, "Answer": 6}, unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", bos_token="[B]", eos_token="[E]")
    tokenizer.chat_template = "{{ bos_token }} {% for message in messages %}{{ message['role'] }} {{ message['content'] }} {{ eos_token }} {% endfor %}"
    row = prepare_example({"messages": [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}]}, tokenizer)
    assert [label for label in row["labels"] if label != -100] == [6]
    assert row["input_ids"][-1] == tokenizer.eos_token_id
    assert row["labels"][-1] == -100


def test_real_bytelevel_tokenizer_unicode_and_literal_eos():
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    backend = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = tokenizers.decoders.ByteLevel()
    backend.train_from_iterator(
        ["Question Answer é 🙂 漢字"],
        trainer=tokenizers.trainers.BpeTrainer(
            vocab_size=280,
            initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<unk>", "<bos>", "<eot>", "<user>", "<assistant>"],
        ),
    )
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", bos_token="<bos>", eos_token="<eot>",
        additional_special_tokens=["<user>", "<assistant>"],
    )
    tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{{ '<' + message['role'] + '>' }}{{ message['content'] }}{{ eos_token }}{% endfor %}"
    content = "é 🙂 漢字 <eot>"
    row = prepare_example({"messages": [
        {"role": "user", "content": "Question"}, {"role": "assistant", "content": content},
    ]}, tokenizer)
    labels = [label for label in row["labels"] if label != -100]
    assert tokenizer.decode(labels, skip_special_tokens=False, clean_up_tokenization_spaces=False) == content
    # An EOS spelling in the chosen content stays a target; the template EOS
    # after it remains masked even though it has exactly the same token ID.
    assert labels[-1] == tokenizer.eos_token_id
    assert row["labels"][-1] == -100
