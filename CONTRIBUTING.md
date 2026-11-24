# Contributing

The public surface is deliberately small:

| File | Responsibility |
| --- | --- |
| `config.py` | Validated immutable objective settings |
| `loss.py` | Snapshot selection, stable scalar objective, custom backward, diagnostics |
| `data.py` | Chosen-only formatting, alignment, masks, collation |
| `trainer.py` | Native Transformers integration and objective checkpoint metadata |

## Preserve the baseline contract

Changes to the following are objective changes and should be explicit:

1. Every prediction uses the original chosen prefix.
2. The alternative is the strongest vocabulary token excluding chosen, including
   when chosen is already ranked first.
3. Selection is strict floor **OR** strict ratio, not AND.
4. Floor-only violations still penalize the selected alternative.
5. Positive chosen support stops at the probability floor.
6. Normalize by all eligible targets, not only active targets.
7. No generated token is fed back into another position's decision.
8. The custom backward selects from the saved forward snapshot. It does not
   mutate input IDs, saved activations, or model parameters.

## Adding another loss

Start with a standalone objective that consumes aligned logits and labels.
Keep candidate selection detached. Write an independent differentiable reference
and compare its gradients with the custom backward away from discontinuities.
If adding a reference-based loss, document how reference probabilities are
obtained for candidates selected from the current model.

Do not disguise a different gradient behind the name of an existing loss. If a
custom derivative is a surrogate instead of the derivative of the reported
scalar, state that directly. The shipped objective returns its real scalar value
and agrees with ordinary differentiation within each fixed selection region.

## High-value validation

Run `python -m pytest` and `python examples/tiny_cpu.py --steps 2`.
Tests cover numerical gradients, extreme probabilities, exact eligibility,
noncontiguous causal slices, data alignment, unequal-length accumulation, and
checkpoint resumption. Use locally initialized models/tokenizers so routine tests
do not need large downloads.

For a GPU or architecture integration, report the actual model, precision,
attention implementation, trainable modules, input/target lengths, observed peak
memory, and comparison against the working baseline. Separate executed checks
from compatibility inferred from an API.

For long-context work, a model-specific fused linear-head objective is a useful
future contribution. The existing chunked loss does not remove the native full
logits tensor. Do not replace an architecture's verified training attention with
an inference-only patch.

## Sharing

Use a branch and a pull request that explains the behavioral change and its
validation. Keep real training corpora, model weights, checkpoints, credentials,
and local paths out of commits. The included MIT license covers this code;
third-party model and dataset permissions remain separate.
