# Objective and backward contract

For target token c at position i, the model input is always the original chosen
prefix. Let z be the vocabulary logits at the prediction position, p_v their
softmax, a the strongest token except c, and q=p_a. Candidate ties use the lowest
vocabulary ID.

The gate is m = 1[p_c < f OR q > r*p_c], with f=.05 and r=3 by default.

The scalar objective is:

    L_i = w_negative * m * [-log(1-q)]
        + w_floor * max(0, log(f)-log(p_c))

Candidate identity and the binary gates are treated as fixed for the current
snapshot. No gradient is taken through argmax or the binary comparisons.

## Derivative

For the negative term alone:

    dL/dz_a = w_negative * m * q
    dL/dz_v = -w_negative * m * q * p_v/(1-q), for v != a

For the floor term, when p_c < f:

    dL/dz_v = w_floor * (p_v - 1[v=c])

Otherwise the floor term's derivative is zero. The custom backward adds these
vectors, divides by the eligible-token denominator, multiplies by the incoming
scalar gradient, and returns the result in the input logit dtype.

This is ordinary first-order differentiation within a fixed selection region.
It does not alter input token IDs or an internal model activation to pretend a
different forward took place. The hard gate is discontinuous at its boundary;
finite-difference checks are performed away from those boundaries.

## Stable implementation

The strongest alternative is found over the complete vocabulary after excluding
only c. For each chunk, subtract a common row maximum from logits. Let s be the
log-sum-exp of every logit except a. Then:

    -log(1-q) = softplus(z_a - s)
    q = sigmoid(z_a - s)
    p_v/(1-q) = exp(z_v - s), for v != a

This avoids division by a rounded zero when q is very close to one. It does not
clip the model's probabilities or suppress any vocabulary tokens. The ratio
comparison uses z_a-z_c > log(r), avoiding probability underflow. The floor uses
the normalized chosen log probability.

The forward computes the honest scalar loss. Backward reevaluates the candidate
and gates from the saved logits, inside the custom autograd Function, before
returning the output gradient. In both cases the model state is the same snapshot.

## Causal alignment and eligibility

The Trainer pairs logits[:, :-1, :] with labels[:, 1:]. The loss itself does not
shift. Labels are the original chosen token IDs, with -100 elsewhere. Prompt and
padding masks remove direct target-loss terms, not attention context. Earlier
context and shared parameters can still receive gradients through selected
predictions.

The denominator counts every eligible chosen target, including inactive ones.
During accumulation, each microbatch contributes its numerator divided by the
whole update's eligible-token count. Native Trainer handles distributed scaling.
No update is applied between tokens or layers of the backward pass.

## Model independence

The public loss consumes logits; the Trainer consumes a loaded causal model.
Neither performs architecture surgery or chooses trainable modules. This makes
the loss reusable with an existing adapter, selected-parameter training, or a
custom loop while retaining the model's working attention implementation.

The cost is a clear memory boundary: a native forward still produces full
sequence-by-vocabulary logits. FP32 loss intermediates are chunked, but the native
logits and their gradient remain. A fused linear-head loss is a separate future
optimization that needs architecture-specific numerical verification.

## Sources

- Token unlikelihood: https://arxiv.org/abs/1908.04319
- Custom autograd: https://docs.pytorch.org/docs/stable/notes/extending.html
- Trainer extension API: https://huggingface.co/docs/transformers/en/main_classes/trainer

The candidate-selection rule and floor-only positive support in this repository
are experimental adaptations. No model-quality superiority over another
objective is claimed without an actual comparison.
