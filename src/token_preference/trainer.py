"""A thin Transformers adapter; model construction stays with the caller."""

from __future__ import annotations

import inspect
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

try:
    from transformers import Trainer
    from transformers.trainer_utils import get_last_checkpoint
except ImportError as exc:
    raise ImportError(
        "PreferenceTrainer requires Transformers and Accelerate. Install the "
        "'trainer' extra into an environment with your working PyTorch build."
    ) from exc

from .config import PreferenceConfig
from .loss import GatedUnlikelihoodLoss, loss_statistics

CONFIG_FILE = "preference_config.json"


class PreferenceTrainer(Trainer):
    """Train on original chosen input IDs with conditional token corrections.

    Supply a loaded causal model, prepared dataset, and ChosenDataCollator. This
    class leaves attention implementations, quantization, adapters, and trainable
    parameter selection under the caller's control. Transformers owns optimizer
    updates, checkpoint scheduling, and gradient accumulation.

    ``compute_loss_func`` is reserved for this adapter. For another training
    framework, import GatedUnlikelihoodLoss directly instead.
    """

    loss_is_scaled_for_ga = True

    def __init__(
        self,
        *args,
        preference_config: PreferenceConfig | None = None,
        forward_kwargs: dict[str, Any] | None = None,
        allow_objective_change_on_resume: bool = False,
        **kwargs,
    ):
        if "compute_loss_func" in kwargs:
            raise ValueError("PreferenceTrainer supplies its own compute_loss_func.")
        supplied_args = kwargs.get("args", args[1] if len(args) > 1 else None)
        if getattr(supplied_args, "label_smoothing_factor", 0):
            raise ValueError("Label smoothing is not part of this objective; set it to zero.")
        self.preference_config = preference_config or PreferenceConfig()
        if self.preference_config.ignore_index != -100:
            raise ValueError(
                "PreferenceTrainer requires ignore_index=-100 for Transformers' "
                "token accounting. Other ignore indices work in the standalone loss."
            )
        self.preference_loss = GatedUnlikelihoodLoss(self.preference_config)
        self.allow_objective_change_on_resume = allow_objective_change_on_resume
        self.forward_kwargs = dict(forward_kwargs or {})
        if {"input_ids", "labels", "attention_mask", "num_items_in_batch"} & self.forward_kwargs.keys():
            raise ValueError("forward_kwargs must not override inputs, masks, or loss bookkeeping.")
        super().__init__(*args, compute_loss_func=self._compute_preference_loss, **kwargs)
        self.model_accepts_loss_kwargs = False
        self.label_names = ["labels"]
        self._loss_shifts_labels = True
        self.loss_is_scaled_for_ga = True
        if self.args.label_smoothing_factor:
            raise ValueError("Label smoothing is not part of this objective; set it to zero.")
        if self.args.n_gpu > 1 and self.args.world_size == 1:
            raise ValueError(
                "Use one visible GPU or launch one process per GPU with torchrun. "
                "Implicit DataParallel does not provide exact token normalization here."
            )
        if self.args.world_size > 1 and not self.args.average_tokens_across_devices:
            raise ValueError(
                "Set average_tokens_across_devices=True for global eligible-token normalization."
            )
        signature = inspect.signature(self.model.forward)
        has_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
        )
        model_config = getattr(self.model, "config", None)
        if "use_cache" in signature.parameters or (
            has_var_kwargs and hasattr(model_config, "use_cache")
        ):
            self.forward_kwargs.setdefault("use_cache", False)

    def _compute_preference_loss(self, outputs, labels, num_items_in_batch=None):
        if labels is None:
            raise ValueError("Preference training requires chosen labels.")
        if isinstance(outputs, dict):
            logits = outputs.get("logits")
        else:
            logits = getattr(outputs, "logits", None)
        if logits is None:
            raise TypeError("The model must return a mapping or ModelOutput containing 'logits'.")
        if logits.ndim != 3 or logits.shape[:2] != labels.shape:
            raise ValueError(
                "Expected logits [batch, sequence, vocabulary] for every input position; "
                "disable last-token-only logits."
            )
        return self.preference_loss(
            logits[:, :-1, :], labels[:, 1:], normalizer=num_items_in_batch
        )

    @staticmethod
    def _validate_chosen_inputs(inputs):
        labels, input_ids = inputs.get("labels"), inputs.get("input_ids")
        if labels is None or input_ids is None:
            raise ValueError("Batches must contain input_ids and chosen labels.")
        if labels.ndim != 2 or labels.shape != input_ids.shape or labels.shape[1] < 2:
            raise ValueError("input_ids and labels must have the same [batch, sequence>=2] shape.")
        if torch.any(labels[:, 0] != -100):
            raise ValueError("Mask labels[:, 0] with -100; use ChosenDataCollator.")
        selected = labels.ne(-100)
        if torch.any(labels[selected] != input_ids[selected]):
            raise ValueError("Supervised labels must equal the original chosen input token IDs.")
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("This adapter accepts a 2-D token attention mask.")
            if torch.any(selected & attention_mask.eq(0)):
                raise ValueError("Padding positions cannot be supervised.")
        if "shift_labels" in inputs:
            raise ValueError("Supply unshifted labels; PreferenceTrainer handles the causal shift.")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self._validate_chosen_inputs(inputs)
        # Native Trainer removes labels, calls the model once, invokes our loss,
        # and performs its DDP scaling. Do not duplicate those operations here.
        forwarded = {**self.forward_kwargs, **inputs}
        return super().compute_loss(
            model,
            forwarded,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    @torch.no_grad()
    def inspect_batch(self, batch):
        """Measure violations in one explicit diagnostic forward (no update)."""
        self._validate_chosen_inputs(batch)
        model = self.model
        was_training = model.training
        model.eval()
        try:
            inputs = self._prepare_inputs(dict(batch))
            labels = inputs.pop("labels")
            outputs = model(**{**self.forward_kwargs, **inputs})
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            return loss_statistics(logits[:, :-1, :], labels[:, 1:], self.preference_config)
        finally:
            model.train(was_training)

    def save_model(self, output_dir=None, _internal_call=False):
        result = super().save_model(output_dir, _internal_call=_internal_call)
        if self.args.should_save:
            destination = Path(output_dir or self.args.output_dir)
            destination.mkdir(parents=True, exist_ok=True)
            payload = {
                "format_version": 1,
                "objective": "gated_unlikelihood_with_probability_floor",
                "config": self.preference_config.to_dict(),
            }
            descriptor, temporary = tempfile.mkstemp(prefix=".preference-", dir=destination)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                os.replace(temporary, destination / CONFIG_FILE)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return result

    def train(self, resume_from_checkpoint=None, *args, **kwargs):
        checkpoint = resume_from_checkpoint
        if checkpoint is True:
            checkpoint = get_last_checkpoint(self.args.output_dir)
            if checkpoint is None:
                raise ValueError("No checkpoint found in the requested output directory.")
        if checkpoint:
            path = Path(checkpoint) / CONFIG_FILE
            if not path.is_file():
                raise ValueError(
                    f"Cannot verify the resumed objective: {path} is missing. "
                    "Load model weights as a new run instead of resuming an unrelated trainer."
                )
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("format_version") != 1 or payload.get("objective") != (
                "gated_unlikelihood_with_probability_floor"
            ):
                raise ValueError("Unrecognized preference checkpoint format or objective.")
            saved = PreferenceConfig.from_dict(payload["config"])
            if saved != self.preference_config and not self.allow_objective_change_on_resume:
                raise ValueError(
                    "The checkpoint's preference configuration differs from this run. "
                    "Restore it or explicitly set allow_objective_change_on_resume=True."
                )
        return super().train(checkpoint, *args, **kwargs)
