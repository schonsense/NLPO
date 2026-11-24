"""Numerical contracts for the chosen-snapshot loss and custom backward."""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest
import torch

import token_preference.loss as loss_module
from token_preference.config import PreferenceConfig
from token_preference.loss import GatedUnlikelihoodLoss, loss_statistics


def reference_loss(logits, labels, config, normalizer=None):
    """Independent ordinary-autograd reference, using a scalar loop and LSEs."""
    terms = []
    for row, label in zip(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)):
        c = int(label)
        if c == config.ignore_index:
            continue
        candidates = [index for index in range(row.numel()) if index != c]
        a = candidates[int(row.detach()[candidates].argmax())]
        normalizing_lse = torch.logsumexp(row, dim=0)
        lp = row[c] - normalizing_lse
        lq = row[a] - normalizing_lse
        active = bool((lp.detach() < math.log(config.probability_floor)) |
                      ((lq - lp).detach() > math.log(config.alternative_ratio)))
        remaining = [index for index in range(row.numel()) if index != a]
        ul = normalizing_lse - torch.logsumexp(row[remaining], dim=0)
        floor = torch.relu(math.log(config.probability_floor) - lp)
        terms.append(config.negative_weight * float(active) * ul + config.floor_weight * floor)
    if not terms:
        return logits.sum() * 0
    return torch.stack(terms).sum() / (normalizer if normalizer is not None else len(terms))


def row_from_probabilities(chosen_probability, alternative_probability, vocabulary=64):
    rest = (1.0 - chosen_probability - alternative_probability) / (vocabulary - 2)
    assert rest <= alternative_probability
    return torch.tensor([chosen_probability, alternative_probability] + [rest] * (vocabulary - 2),
                        dtype=torch.float64).log()


@pytest.mark.parametrize("chunk_size", [1, 3, 128])
def test_value_and_gradient_match_independent_reference(chunk_size):
    torch.manual_seed(11)
    config = PreferenceConfig(chunk_size=chunk_size, negative_weight=0.7, floor_weight=1.3)
    logits = (torch.randn(2, 5, 64, dtype=torch.float64) * 1.4).requires_grad_()
    labels = torch.tensor([[0, 2, -100, 13, 32], [63, 8, 3, -100, 17]])
    actual = GatedUnlikelihoodLoss(config)(logits, labels)
    actual_grad, = torch.autograd.grad(actual, logits)
    reference_logits = logits.detach().clone().requires_grad_()
    expected = reference_loss(reference_logits, labels, config)
    expected_grad, = torch.autograd.grad(expected, reference_logits)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-11, atol=1e-12)
    assert torch.count_nonzero(actual_grad[labels == -100]) == 0


def test_finite_difference_gradcheck_away_from_gate_boundaries():
    logits = torch.tensor([[-1.0, 3.0, 0.0, -2.0], [1.0, -1.0, 0.0, 2.0]],
                          dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 1])
    loss = GatedUnlikelihoodLoss(PreferenceConfig(probability_floor=0.3, alternative_ratio=1.8))
    assert torch.autograd.gradcheck(lambda z: loss(z, labels), (logits,), eps=1e-6, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize(
    "p,q,expected_floor,expected_ratio",
    [(0.04, 0.035, True, False), (0.08, 0.30, False, True),
     (0.02, 0.40, True, True), (0.05, 0.14, False, False)],
)
def test_selection_truth_table(p, q, expected_floor, expected_ratio):
    logits = row_from_probabilities(p, q)[None, :].requires_grad_()
    labels = torch.tensor([0])
    stats = loss_statistics(logits, labels)
    assert stats["floor_violations"] == int(expected_floor)
    assert stats["ratio_violations"] == int(expected_ratio)
    assert stats["active_tokens"] == int(expected_floor or expected_ratio)
    GatedUnlikelihoodLoss()(logits, labels).backward()
    if expected_floor or expected_ratio:
        assert logits.grad[0, 1] > 0  # selected alternative receives suppression
    else:
        assert torch.count_nonzero(logits.grad) == 0


def test_floor_only_still_penalizes_runner_up_when_chosen_is_top_one():
    logits = row_from_probabilities(0.04, 0.035)[None, :].requires_grad_()
    assert logits.argmax().item() == 0
    config = PreferenceConfig(floor_weight=0.0)
    GatedUnlikelihoodLoss(config)(logits, torch.tensor([0])).backward()
    assert logits.grad[0, 1] > 0
    assert logits.grad[0, 0] < 0
    assert torch.count_nonzero(logits.grad > 0).item() == 1


def test_strict_equality_at_both_thresholds_has_no_direct_correction():
    # Exact powers of two avoid testing transcendental rounding of decimal
    # probabilities instead of the strict comparison contract.
    config = PreferenceConfig(probability_floor=0.25, alternative_ratio=2.0)
    logits = torch.tensor([[0.25, 0.5, 0.25]], dtype=torch.float64).log().requires_grad_()
    loss = GatedUnlikelihoodLoss(config)(logits, torch.tensor([0]))
    loss.backward()
    assert loss.item() == 0.0
    assert torch.count_nonzero(logits.grad) == 0
    assert loss_statistics(logits, torch.tensor([0]), config)["active_tokens"] == 0


def test_just_outside_equality_is_selected_without_hidden_tolerance():
    config = PreferenceConfig(probability_floor=0.25, alternative_ratio=2.0)
    logits = torch.tensor([[0.25, 0.5, 0.25]], dtype=torch.float64).log()
    logits[0, 0] -= 1e-10
    stats = loss_statistics(logits, torch.tensor([0]), config)
    assert stats["floor_violations"] == 1
    assert stats["ratio_violations"] == 1


def test_all_masked_is_differentiable_zero():
    logits = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    labels = torch.full((3,), -100)
    loss = GatedUnlikelihoodLoss()(logits, labels)
    loss.backward()
    assert loss.item() == 0.0
    assert torch.equal(logits.grad, torch.zeros_like(logits))
    stats = loss_statistics(logits, labels)
    assert stats["eligible_tokens"] == 0
    assert stats["loss"] == 0.0
    assert stats["mean_chosen_probability"] == 0.0


def test_empty_position_dimension_is_differentiable_zero():
    logits = torch.empty(0, 5, dtype=torch.float64, requires_grad=True)
    labels = torch.empty(0, dtype=torch.long)
    loss = GatedUnlikelihoodLoss()(logits, labels)
    loss.backward()
    assert loss.item() == 0.0
    assert logits.grad.shape == logits.shape


def test_denominator_counts_all_eligible_positions_including_inactive():
    active = row_from_probabilities(0.02, 0.4)
    inactive = row_from_probabilities(0.08, 0.15)
    one = GatedUnlikelihoodLoss()(active[None, :], torch.tensor([0]))
    two = GatedUnlikelihoodLoss()(torch.stack([active, inactive]), torch.tensor([0, 0]))
    torch.testing.assert_close(two, one / 2)


def test_external_normalizer_and_upstream_gradient_scaling():
    torch.manual_seed(3)
    logits = torch.randn(4, 32, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 1, -100, 3])
    config = PreferenceConfig()
    actual = GatedUnlikelihoodLoss(config)(logits, labels, normalizer=torch.tensor(17))
    actual_grad, = torch.autograd.grad(7.5 * actual, logits)
    reference_logits = logits.detach().clone().requires_grad_()
    expected = reference_loss(reference_logits, labels, config, normalizer=17)
    expected_grad, = torch.autograd.grad(7.5 * expected, reference_logits)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-11, atol=1e-12)


def test_microbatch_sums_equal_whole_batch_with_fixed_normalizer():
    torch.manual_seed(6)
    logits = torch.randn(7, 64, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 2, 3, -100, 17, 1, 63])
    criterion = GatedUnlikelihoodLoss()
    whole = criterion(logits, labels)
    whole_grad, = torch.autograd.grad(whole, logits)
    fragments = sum(criterion(logits[a:b], labels[a:b], normalizer=6) for a, b in [(0, 2), (2, 5), (5, 7)])
    fragment_grad, = torch.autograd.grad(fragments, logits)
    torch.testing.assert_close(fragments, whole, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(fragment_grad, whole_grad, rtol=1e-11, atol=1e-12)


def test_near_certain_alternative_has_finite_unclipped_loss_and_gradient():
    logits = torch.tensor([[0.0, 1000.0, -1.0]], dtype=torch.float64, requires_grad=True)
    config = PreferenceConfig(floor_weight=0.0)
    loss = GatedUnlikelihoodLoss(config)(logits, torch.tensor([0]))
    loss.backward()
    expected = 1000.0 - math.log1p(math.exp(-1.0))
    assert loss.item() == pytest.approx(expected, abs=1e-12)
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1].item() == pytest.approx(1.0)
    assert logits.grad[0, 0].item() == pytest.approx(-1.0 / (1.0 + math.exp(-1.0)))
    assert logits.grad.sum().item() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_low_precision_input_keeps_gradient_dtype_and_fp32_loss(dtype):
    torch.manual_seed(19)
    logits = torch.randn(4, 64).to(dtype).requires_grad_()
    labels = torch.tensor([0, 10, 20, -100])
    loss = GatedUnlikelihoodLoss()(logits, labels)
    loss.backward()
    assert loss.dtype == torch.float32
    assert logits.grad.dtype == dtype
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    reference = logits.detach().float().requires_grad_()
    reference_loss_value = GatedUnlikelihoodLoss()(reference, labels)
    reference_loss_value.backward()
    torch.testing.assert_close(loss, reference_loss_value)
    torch.testing.assert_close(logits.grad.float(), reference.grad, rtol=0.01, atol=0.002)


def test_noncontiguous_causal_slices_match_contiguous_reference():
    torch.manual_seed(21)
    base = torch.randn(3, 6, 40, dtype=torch.float64, requires_grad=True)
    logits = base[:, :-1, :]
    labels = torch.randint(0, 40, (3, 6))[:, 1:]
    assert not logits.is_contiguous()
    config = PreferenceConfig(chunk_size=2)
    loss = GatedUnlikelihoodLoss(config)(logits, labels)
    grad, = torch.autograd.grad(loss, base)
    copy = logits.detach().clone().requires_grad_()
    expected = reference_loss(copy, labels, config)
    expected_grad, = torch.autograd.grad(expected, copy)
    torch.testing.assert_close(loss, expected, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(grad[:, :-1], expected_grad, rtol=1e-11, atol=1e-12)
    assert torch.count_nonzero(grad[:, -1]) == 0


def test_noncontiguous_multiple_leading_dimensions():
    torch.manual_seed(9)
    base = torch.randn(2, 3, 4, 16, dtype=torch.float64, requires_grad=True)
    logits = base.transpose(0, 2)
    labels = torch.randint(0, 16, logits.shape[:-1])
    config = PreferenceConfig(chunk_size=3, probability_floor=0.2)
    actual = GatedUnlikelihoodLoss(config)(logits, labels)
    expected = reference_loss(logits, labels, config)
    actual_grad, = torch.autograd.grad(actual, base, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected, base)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-11, atol=1e-12)


def test_selection_is_recomputed_inside_backward():
    logits = torch.tensor([[0.0, 4.0, -1.0]], requires_grad=True)
    labels = torch.tensor([0])
    with patch.object(loss_module, "_select_alternative", wraps=loss_module._select_alternative) as select:
        value = GatedUnlikelihoodLoss()(logits, labels)
        forward_calls = select.call_count
        assert forward_calls > 0
        value.backward()
        assert select.call_count > forward_calls


def test_inputs_are_unchanged_and_snapshot_mutation_is_rejected():
    logits = torch.tensor([[0.0, 4.0, -1.0]], requires_grad=True)
    labels = torch.tensor([0])
    original_logits = logits.detach().clone()
    original_labels = labels.clone()
    value = GatedUnlikelihoodLoss()(logits, labels)
    value.backward(retain_graph=True)
    assert torch.equal(logits, original_logits)
    assert torch.equal(labels, original_labels)
    with torch.no_grad():
        logits.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        value.backward()


def test_all_vocabulary_entries_can_be_selected_and_ties_use_lowest_id():
    logits = torch.zeros(1, 4, dtype=torch.float64, requires_grad=True)
    config = PreferenceConfig(probability_floor=0.5, floor_weight=0)
    GatedUnlikelihoodLoss(config)(logits, torch.tensor([0])).backward()
    assert logits.grad[0, 1] > 0
    assert (logits.grad[0, [0, 2, 3]] < 0).all()
    largest_id = torch.tensor([[0.0, 0.0, 0.0, 8.0]], dtype=torch.float64, requires_grad=True)
    GatedUnlikelihoodLoss(config)(largest_id, torch.tensor([0])).backward()
    assert largest_id.grad[0, 3] > 0  # no special-token or vocabulary exclusions


def test_common_logit_shift_does_not_change_value_or_gradient():
    torch.manual_seed(14)
    logits = torch.randn(3, 32, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 4, 19])
    shifted = (logits.detach() + 15).requires_grad_()
    criterion = GatedUnlikelihoodLoss()
    first = criterion(logits, labels)
    second = criterion(shifted, labels)
    first_grad, = torch.autograd.grad(first, logits)
    second_grad, = torch.autograd.grad(second, shifted)
    torch.testing.assert_close(first, second, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(first_grad, second_grad, rtol=1e-11, atol=1e-12)


def test_scalar_label_and_one_dimensional_logits_are_supported():
    logits = torch.tensor([0.0, 4.0, -1.0], dtype=torch.float64, requires_grad=True)
    label = torch.tensor(0)
    actual = GatedUnlikelihoodLoss()(logits, label)
    expected = reference_loss(logits, label, PreferenceConfig())
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad.shape == logits.shape


def test_noncontiguous_one_dimensional_logits_are_supported():
    logits = torch.tensor([0.0, -4.0, 4.0, -8.0, -1.0], dtype=torch.float64)[::2].requires_grad_()
    assert not logits.is_contiguous()
    label = torch.tensor(0)
    actual = GatedUnlikelihoodLoss()(logits, label)
    expected = reference_loss(logits, label, PreferenceConfig())
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(logits.grad).all()


def test_large_common_logit_offset_does_not_destroy_probability_normalization():
    original = torch.zeros(2, 64, dtype=torch.float32, requires_grad=True)
    shifted = torch.full_like(original, 1e8, requires_grad=True)
    labels = torch.tensor([0, 3])
    criterion = GatedUnlikelihoodLoss()
    actual = criterion(shifted, labels)
    expected = criterion(original, labels)
    actual_grad, = torch.autograd.grad(actual, shifted)
    expected_grad, = torch.autograd.grad(expected, original)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert loss_statistics(shifted, labels)["floor_violations"] == 2


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_logits_are_rejected_even_on_ignored_positions(bad):
    logits = torch.tensor([[0.0, bad, 1.0]])
    with pytest.raises(ValueError, match="finite"):
        GatedUnlikelihoodLoss()(logits, torch.tensor([-100]))


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), torch.tensor([1, 2])])
def test_invalid_normalizer_is_rejected(bad):
    with pytest.raises(ValueError, match="normalizer"):
        GatedUnlikelihoodLoss()(torch.zeros(1, 4), torch.tensor([0]), normalizer=bad)


def test_trainable_normalizer_is_rejected():
    with pytest.raises(ValueError, match="must not require gradients"):
        GatedUnlikelihoodLoss()(torch.zeros(1, 4), torch.tensor([0]),
                               normalizer=torch.tensor(2.0, requires_grad=True))


@pytest.mark.parametrize("label", [-1, 4])
def test_out_of_range_label_is_rejected(label):
    with pytest.raises(ValueError, match="vocabulary token IDs"):
        GatedUnlikelihoodLoss()(torch.zeros(1, 4), torch.tensor([label]))


def test_shape_dtype_and_vocabulary_validation():
    criterion = GatedUnlikelihoodLoss()
    with pytest.raises(ValueError, match="labels must match"):
        criterion(torch.zeros(2, 3), torch.tensor([0]))
    with pytest.raises(TypeError, match="labels must have dtype"):
        criterion(torch.zeros(2, 3), torch.tensor([0.0, 1.0]))
    with pytest.raises(TypeError, match="floating-point"):
        criterion(torch.zeros(2, 3, dtype=torch.long), torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="at least two"):
        criterion(torch.zeros(2, 1), torch.tensor([0, 0]))


@pytest.mark.parametrize("settings", [
    {"probability_floor": 0}, {"probability_floor": 1}, {"probability_floor": float("nan")},
    {"alternative_ratio": 0}, {"alternative_ratio": float("inf")},
    {"negative_weight": -1}, {"floor_weight": -1}, {"negative_weight": 0, "floor_weight": 0},
    {"chunk_size": 0}, {"chunk_size": 1.5}, {"ignore_index": True}, {"negative_weight": True},
])
def test_configuration_validation(settings):
    with pytest.raises((TypeError, ValueError)):
        PreferenceConfig(**settings)


def test_configuration_serialization_and_immutability():
    config = PreferenceConfig(chunk_size=7, floor_weight=0.2)
    assert PreferenceConfig.from_dict(config.to_dict()) == config
    with pytest.raises(TypeError):
        PreferenceConfig.from_dict({"typo": 123})
    with pytest.raises(AttributeError):
        config.chunk_size = 8
