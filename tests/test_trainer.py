"""Integration contracts; all models are tiny, local, and initialized randomly."""

import copy
import json

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("accelerate")
from transformers import TrainingArguments

from token_preference import (
    ChosenDataCollator, GatedUnlikelihoodLoss, PreferenceConfig, PreferenceTrainer,
)


class ProbeModel(torch.nn.Module):
    # No labels or **kwargs: the adapter must not forward loss bookkeeping.
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 8)
        self.head = torch.nn.Linear(8, 32)
        self.seen = []

    def forward(self, input_ids, attention_mask=None):
        self.seen.append(input_ids.detach().cpu().clone())
        return {"logits": self.head(self.embedding(input_ids))}


def rows():
    return [
        {"input_ids": [1, 3, 8, 9], "labels": [-100, -100, 8, 9]},
        {"input_ids": [1, 4, 10, 11, 12, 13, 14], "labels": [-100, -100, 10, 11, 12, 13, 14]},
    ]


def args(path, **overrides):
    values = dict(
        output_dir=str(path), use_cpu=True, report_to="none", disable_tqdm=True,
        max_steps=1, per_device_train_batch_size=2, per_device_eval_batch_size=2,
        learning_rate=.01, optim="sgd", lr_scheduler_type="constant",
        max_grad_norm=0., weight_decay=0., save_strategy="no", logging_strategy="no",
        seed=13, data_seed=13,
    )
    values.update(overrides)
    return TrainingArguments(**values)


def make_trainer(path, model=None, config=None, **arg_overrides):
    return PreferenceTrainer(
        model=model if model is not None else ProbeModel(),
        preference_config=config or PreferenceConfig(),
        args=args(path, **arg_overrides),
        data_collator=ChosenDataCollator(pad_token_id=0),
        train_dataset=rows(), eval_dataset=rows(),
    )


def test_adapter_matches_explicit_causal_shift_and_preserves_inputs(tmp_path):
    trainer = make_trainer(tmp_path)
    batch = trainer.data_collator(rows())
    original = {key: value.clone() for key, value in batch.items()}
    loss, outputs = trainer.compute_loss(trainer.model, batch, return_outputs=True)
    expected = GatedUnlikelihoodLoss(trainer.preference_config)(
        outputs["logits"][:, :-1], batch["labels"][:, 1:]
    )
    torch.testing.assert_close(loss, expected)
    for key in batch:
        torch.testing.assert_close(batch[key], original[key])
    torch.testing.assert_close(trainer.model.seen[-1], batch["input_ids"])
    loss.backward()
    assert trainer.model.head.weight.grad is not None


def test_supplied_accumulation_normalizer_is_used_once(tmp_path):
    trainer = make_trainer(tmp_path)
    batch = trainer.data_collator(rows())
    eligible = batch["labels"].ne(-100).sum()
    regular = trainer.compute_loss(trainer.model, batch)
    scaled = trainer.compute_loss(trainer.model, batch, num_items_in_batch=2 * eligible)
    torch.testing.assert_close(scaled, regular / 2)


def test_unequal_microbatches_match_one_combined_optimizer_update(tmp_path):
    torch.manual_seed(33)
    first = ProbeModel()
    second = copy.deepcopy(first)
    combined = make_trainer(tmp_path / "combined", model=first)
    accumulated = make_trainer(
        tmp_path / "accumulated", model=second,
        per_device_train_batch_size=1, gradient_accumulation_steps=2,
    )
    combined.train()
    accumulated.train()
    for left, right in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(left, right, rtol=2e-6, atol=2e-7)


def test_plain_model_without_labels_parameter_can_train_and_evaluate(tmp_path):
    trainer = make_trainer(tmp_path)
    assert trainer.label_names == ["labels"]
    before = trainer.model.head.weight.detach().clone()
    trainer.train()
    assert not torch.equal(before, trainer.model.head.weight)
    result = trainer.evaluate()
    assert "eval_loss" in result
    assert torch.isfinite(torch.tensor(result["eval_loss"]))


def test_inspection_does_not_update_or_change_mode(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.model.train()
    before = copy.deepcopy(trainer.model.state_dict())
    summary = trainer.inspect_batch(trainer.data_collator(rows()))
    assert summary["eligible_tokens"] == 7
    assert summary["active_tokens"] <= 7
    assert trainer.model.training
    for key, value in before.items():
        torch.testing.assert_close(value, trainer.model.state_dict()[key])


def test_save_model_preserves_objective_config(tmp_path):
    config = PreferenceConfig(probability_floor=.08, alternative_ratio=2.5)
    trainer = make_trainer(tmp_path, config=config)
    trainer.save_model(str(tmp_path / "saved"))
    payload = json.loads((tmp_path / "saved" / "preference_config.json").read_text())
    assert PreferenceConfig.from_dict(payload["config"]) == config


def test_resume_rejects_changed_objective_before_loading(tmp_path):
    original = make_trainer(tmp_path / "original")
    original.save_model(str(tmp_path / "saved"))
    changed = make_trainer(tmp_path / "changed", config=PreferenceConfig(probability_floor=.07))
    with pytest.raises(ValueError, match="differs"):
        changed.train(resume_from_checkpoint=str(tmp_path / "saved"))


def test_checkpoint_resume_runs_with_preserved_objective(tmp_path):
    first = make_trainer(tmp_path, save_strategy="steps", save_steps=1)
    first.train()
    checkpoint = tmp_path / "checkpoint-1"
    assert (checkpoint / "preference_config.json").is_file()
    resumed = make_trainer(tmp_path, max_steps=2)
    result = resumed.train(resume_from_checkpoint=str(checkpoint))
    assert result.global_step == 2


def test_reject_unverifiable_resume(tmp_path):
    trainer = make_trainer(tmp_path / "out")
    with pytest.raises(ValueError, match="missing"):
        trainer.train(resume_from_checkpoint=str(tmp_path / "unrelated"))


def test_no_checkpoint_for_boolean_resume(tmp_path):
    trainer = make_trainer(tmp_path)
    with pytest.raises(ValueError, match="No checkpoint"):
        trainer.train(resume_from_checkpoint=True)


def test_nonstandard_ignore_index_rejected_by_adapter(tmp_path):
    with pytest.raises(ValueError, match="ignore_index=-100"):
        make_trainer(tmp_path, config=PreferenceConfig(ignore_index=-1))


def test_reject_label_smoothing(tmp_path):
    with pytest.raises(ValueError, match="Label smoothing"):
        make_trainer(tmp_path, label_smoothing_factor=.1)


def test_reject_unmasked_first_target(tmp_path):
    trainer = make_trainer(tmp_path)
    batch = trainer.data_collator(rows())
    batch["labels"][0, 0] = batch["input_ids"][0, 0]
    with pytest.raises(ValueError, match="Mask labels"):
        trainer.compute_loss(trainer.model, batch)


def test_reject_replaced_training_targets(tmp_path):
    trainer = make_trainer(tmp_path)
    batch = trainer.data_collator(rows())
    batch["labels"][0, 2] = 31
    with pytest.raises(ValueError, match="original chosen"):
        trainer.compute_loss(trainer.model, batch)


def test_reject_supervised_padding(tmp_path):
    trainer = make_trainer(tmp_path)
    batch = trainer.data_collator(rows())
    batch["labels"][0, -1] = batch["input_ids"][0, -1]
    with pytest.raises(ValueError, match="Padding"):
        trainer.compute_loss(trainer.model, batch)


def test_train_preserves_parent_positional_arguments(tmp_path, monkeypatch):
    from transformers import Trainer
    trainer = make_trainer(tmp_path)
    trial = {"trial": 3}
    def parent_train(self, resume_from_checkpoint=None, trial=None, **kwargs):
        return resume_from_checkpoint, trial
    monkeypatch.setattr(Trainer, "train", parent_train)
    assert trainer.train(None, trial) == (None, trial)
