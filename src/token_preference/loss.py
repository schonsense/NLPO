"""One-forward token unlikelihood with conditional selection inside backward.

The caller supplies *already aligned* logits and next-token labels. This module
never shifts labels, modifies input tokens, generates a continuation, or updates
model parameters. Its custom backward recomputes the candidate and both gates
from the saved, unchanged logits before constructing the corrective gradient.

The scalar returned by forward is the actual gated objective, rather than a
dummy value used solely to attach a custom gradient. Forward evaluates the
selection for reporting the value; backward independently evaluates it again.
The candidate index and binary gates are treated as constants for that snapshot.
Only first-order differentiation is supported.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import torch
from torch import Tensor, nn
from torch.autograd.function import once_differentiable

from .config import PreferenceConfig


def _working_dtype(logits: Tensor) -> torch.dtype:
    return torch.float64 if logits.dtype == torch.float64 else torch.float32


def _logit_rows(logits: Tensor, start: int, end: int) -> Tensor:
    """Flatten leading dimensions a chunk at a time, without a full copy.

    A causal slice such as ``logits[:, :-1, :]`` is usually noncontiguous for
    batch sizes above one. Calling reshape on that whole slice would allocate
    another full logits tensor before chunking. Integer row gathering confines
    any necessary copy to this chunk instead.
    """
    if logits.ndim == 1:
        return logits.unsqueeze(0)[start:end]
    if logits.is_contiguous():
        return logits.view(-1, logits.shape[-1])[start:end]
    count = math.prod(logits.shape[:-1])
    linear = torch.arange(start, min(end, count), device=logits.device)
    reversed_indices = []
    for size in reversed(logits.shape[:-1]):
        reversed_indices.append(linear.remainder(size))
        linear = torch.div(linear, size, rounding_mode="floor")
    return logits[tuple(reversed(reversed_indices))]


def _validate_inputs(logits: Tensor, labels: Tensor, config: PreferenceConfig) -> None:
    if not isinstance(logits, Tensor) or not isinstance(labels, Tensor):
        raise TypeError("logits and labels must be torch tensors")
    if logits.ndim < 1 or logits.shape[-1] < 2:
        raise ValueError("logits must have shape [..., vocabulary_size] with at least two tokens")
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("logits must have a floating-point dtype")
    if labels.dtype not in (torch.int32, torch.int64):
        raise TypeError("labels must have dtype torch.int32 or torch.int64")
    if tuple(labels.shape) != tuple(logits.shape[:-1]):
        raise ValueError("labels must match logits.shape[:-1]; causally shift both before calling")
    if labels.device != logits.device:
        raise ValueError("logits and labels must be on the same device")
    valid = labels.ne(config.ignore_index)
    invalid = valid & ((labels < 0) | (labels >= logits.shape[-1]))
    if bool(invalid.any()):
        raise ValueError("non-ignored labels must be valid vocabulary token IDs")


def _normalizer(
    labels: Tensor,
    logits: Tensor,
    config: PreferenceConfig,
    normalizer: Tensor | Real | None,
) -> Tensor:
    dtype = _working_dtype(logits)
    if normalizer is None:
        # The zero-target case is a differentiable zero, not a division by zero.
        return labels.ne(config.ignore_index).sum().clamp_min(1).to(dtype=dtype)
    if isinstance(normalizer, Tensor):
        if normalizer.numel() != 1 or normalizer.is_complex():
            raise ValueError("normalizer must be a positive, finite scalar")
        if normalizer.requires_grad:
            raise ValueError("normalizer is a fixed count and must not require gradients")
        result = normalizer.detach().to(device=logits.device, dtype=dtype).reshape(())
    elif isinstance(normalizer, Real) and not isinstance(normalizer, bool):
        result = torch.tensor(normalizer, device=logits.device, dtype=dtype)
    else:
        raise TypeError("normalizer must be a positive numeric scalar or scalar tensor")
    if not bool(torch.isfinite(result) & result.gt(0)):
        raise ValueError("normalizer must be a positive, finite scalar")
    return result


def _select_alternative(z: Tensor, chosen: Tensor) -> Tensor:
    """Lowest-ID maximizer excluding chosen, including floor-only/top-1 cases."""
    excluded = z.scatter(1, chosen[:, None], -torch.inf)
    return excluded.argmax(dim=1)


def _snapshot(
    z: Tensor, chosen: Tensor, config: PreferenceConfig
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Evaluate one chunk in the working dtype without mutating input tensors."""
    # Work relative to the largest logit. Subtracting an absolute log-normalizer
    # from very large, nearly equal logits can otherwise lose log probabilities
    # entirely even though softmax itself is well defined.
    z = z - z.amax(dim=1, keepdim=True)
    alternative = _select_alternative(z, chosen)
    chosen_z = z.gather(1, chosen[:, None]).squeeze(1)
    alternative_z = z.gather(1, alternative[:, None]).squeeze(1)
    without_alternative = z.scatter(1, alternative[:, None], -torch.inf)
    other_lse = torch.logsumexp(without_alternative, dim=1)
    log_normalizer = torch.logaddexp(alternative_z, other_lse)
    chosen_logp = chosen_z - log_normalizer
    # Using logits for the ratio comparison cancels a common normalizer and
    # avoids underflow of very small chosen probabilities. Inequalities are
    # strict in the working floating-point dtype, with no hidden tolerance.
    below_floor = chosen_logp < math.log(config.probability_floor)
    above_ratio = alternative_z - chosen_z > math.log(config.alternative_ratio)
    return (
        alternative,
        chosen_logp,
        alternative_z - other_lse,
        without_alternative,
        other_lse,
        below_floor,
        above_ratio,
    )


def _sum_objective(
    logits: Tensor,
    labels: Tensor,
    config: PreferenceConfig,
    *,
    collect_statistics: bool = False,
) -> tuple[Tensor, dict[str, Tensor]]:
    flat_labels = labels.reshape(-1)
    dtype = _working_dtype(logits)
    total = torch.zeros((), dtype=dtype, device=logits.device)
    finite = torch.ones((), dtype=torch.bool, device=logits.device)
    stats = {
        key: torch.zeros((), dtype=dtype, device=logits.device)
        for key in (
            "eligible_tokens",
            "active_tokens",
            "floor_violations",
            "ratio_violations",
            "both_violations",
            "chosen_probability_sum",
            "alternative_probability_sum",
            "negative_loss_sum",
            "floor_loss_sum",
        )
    } if collect_statistics else {}
    for start in range(0, flat_labels.numel(), config.chunk_size):
        end = start + config.chunk_size
        raw = _logit_rows(logits, start, end)
        finite = finite & torch.isfinite(raw).all()
        chunk_labels = flat_labels[start:end]
        eligible = chunk_labels.ne(config.ignore_index)
        z = raw[eligible].to(dtype=dtype)
        chosen = chunk_labels[eligible].long()
        alternative, logp, alt_logodds, _, _, floor, ratio = _snapshot(z, chosen, config)
        active = floor | ratio
        # softplus(log(q/(1-q))) == -log(1-q), including when q itself
        # rounds to 1. There is no probability clamp or finite-loss ceiling.
        negative_loss = torch.logaddexp(torch.zeros_like(alt_logodds), alt_logodds) * active.to(dtype)
        floor_loss = (math.log(config.probability_floor) - logp).clamp_min(0)
        total = total + config.negative_weight * negative_loss.sum()
        total = total + config.floor_weight * floor_loss.sum()
        if collect_statistics:
            stats["eligible_tokens"] += chosen.numel()
            stats["active_tokens"] += active.sum()
            stats["floor_violations"] += floor.sum()
            stats["ratio_violations"] += ratio.sum()
            stats["both_violations"] += (floor & ratio).sum()
            stats["chosen_probability_sum"] += logp.exp().sum()
            stats["alternative_probability_sum"] += alt_logodds.sigmoid().sum()
            stats["negative_loss_sum"] += negative_loss.sum()
            stats["floor_loss_sum"] += floor_loss.sum()
    if not bool(finite):
        raise ValueError("logits must be finite, including ignored positions")
    return total, stats


class _SnapshotUnlikelihood(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, logits: Tensor, labels: Tensor, denominator: Tensor, config: PreferenceConfig) -> Tensor:
        ctx.save_for_backward(logits, labels, denominator)
        ctx.config = config
        total, _ = _sum_objective(logits, labels, config)
        return total / denominator

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None, None]:
        logits, labels, denominator = ctx.saved_tensors
        config: PreferenceConfig = ctx.config
        dtype = _working_dtype(logits)
        flat_labels = labels.reshape(-1)
        gradient = torch.zeros(logits.shape, device=logits.device, dtype=logits.dtype)
        flat_gradient = gradient.reshape(-1, logits.shape[-1])
        scale = grad_output.to(dtype=dtype) / denominator
        for start in range(0, flat_labels.numel(), config.chunk_size):
            end = start + config.chunk_size
            chunk_labels = flat_labels[start:end]
            eligible = chunk_labels.ne(config.ignore_index)
            z = _logit_rows(logits, start, end)[eligible].to(dtype=dtype)
            chosen = chunk_labels[eligible].long()
            # Candidate IDs and gates are deliberately recomputed HERE from
            # the original saved forward snapshot, before any gradient flows
            # into the model. No intermediate optimizer update occurs.
            alternative, _, alt_logodds, without_alt, other_lse, floor, ratio = _snapshot(z, chosen, config)
            active = (floor | ratio).to(dtype)
            q = alt_logodds.sigmoid()
            # q * P(v | v != alternative) avoids the unstable expression
            # q*p_v/(1-q) when q rounds to one. Both are algebraically equal.
            remaining = (without_alt - other_lse[:, None]).exp()
            correction = -config.negative_weight * (active * q)[:, None] * remaining
            correction.scatter_(
                1, alternative[:, None], (config.negative_weight * active * q)[:, None]
            )
            if config.floor_weight:
                support = z.softmax(dim=1)
                support.scatter_add_(1, chosen[:, None], -torch.ones_like(chosen[:, None], dtype=dtype))
                correction += config.floor_weight * floor.to(dtype)[:, None] * support
            flat_gradient[start:end][eligible] = (correction * scale).to(logits.dtype)
        return gradient, None, None, None


class GatedUnlikelihoodLoss(nn.Module):
    """Selected-negative unlikelihood plus a chosen-probability floor.

    Args:
        config: Frozen loss settings. Defaults to a 5% floor and 3x ratio.

    ``logits[..., v]`` must predict ``labels[...]``; shifting is the caller's
    responsibility. Labels equal to ``ignore_index`` have zero direct gradient.
    By default the sum is divided by ALL eligible target tokens, not the active
    subset. Pass ``normalizer`` for a fixed accumulation-window token count.

    Loss values and calculations use float32 for low-precision inputs, or
    float64 for double inputs. The returned logit gradient has the input dtype.
    FP32 vocabulary-sized intermediates are limited to ``chunk_size`` rows;
    the input logits and its full-size gradient still exist. This is not a
    fused linear-head loss and does not remove the model's output-logit tensor.
    """

    def __init__(self, config: PreferenceConfig | None = None) -> None:
        super().__init__()
        if config is not None and not isinstance(config, PreferenceConfig):
            raise TypeError("config must be a PreferenceConfig")
        self.config = config or PreferenceConfig()

    def forward(self, logits: Tensor, labels: Tensor, *, normalizer: Tensor | Real | None = None) -> Tensor:
        _validate_inputs(logits, labels, self.config)
        denominator = _normalizer(labels, logits, self.config, normalizer)
        return _SnapshotUnlikelihood.apply(logits, labels, denominator, self.config)


@torch.no_grad()
def loss_statistics(
    logits: Tensor, labels: Tensor, config: PreferenceConfig | None = None
) -> dict[str, float | int]:
    """Inspect decisions on an aligned snapshot; no gradient or input mutation.

    Counts include all non-ignored labels. Means use all eligible positions,
    with zeros when no labels are eligible. Calling this is optional and costs
    an additional chunked vocabulary scan; it is intended for logging/eval.
    """
    config = config or PreferenceConfig()
    _validate_inputs(logits, labels, config)
    total, stats = _sum_objective(logits, labels, config, collect_statistics=True)
    values = {key: value.item() for key, value in stats.items()}
    count = int(values["eligible_tokens"])
    denominator = max(count, 1)
    result: dict[str, float | int] = {
        key: int(values[key])
        for key in ("eligible_tokens", "active_tokens", "floor_violations", "ratio_violations", "both_violations")
    }
    result.update(
        active_fraction=values["active_tokens"] / denominator,
        floor_fraction=values["floor_violations"] / denominator,
        ratio_fraction=values["ratio_violations"] / denominator,
        mean_chosen_probability=values["chosen_probability_sum"] / denominator,
        mean_alternative_probability=values["alternative_probability_sum"] / denominator,
        mean_negative_loss=values["negative_loss_sum"] / denominator,
        mean_floor_loss=values["floor_loss_sum"] / denominator,
        loss=total.item() / denominator,
    )
    return result
