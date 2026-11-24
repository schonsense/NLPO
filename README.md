# NLPO — Natural Language Preference Optimization

The Python package and import name is `token_preference`.

An importable PyTorch objective and Hugging Face trainer for **chosen-only,
teacher-forced token preference training**.

The model processes the original chosen sequence. A custom backward operation
selects the strongest non-chosen token at each eligible position and constructs
the correction from that same forward snapshot. Nothing is sampled or inserted
into the input sequence.

**Status:** experimental objective, with numerical and trainer integration tests.
The small CPU example works without downloading weights. CUDA, Ripple-65B, and
distributed production runs require validation in the model's working training
environment. See [verification](docs/verification.md) for the checks actually run.

## Install

Clone the repository and use the Python environment that already has your
working CUDA PyTorch build. Install this package without replacing dependencies:

```bash
git clone https://github.com/schonsense/NLPO.git
cd NLPO
python -m pip install --no-deps -e .
```

The standalone objective needs PyTorch. The trainer also needs Transformers and
Accelerate. The declared extras are useful when deliberately setting up a new
environment:

```bash
python -m pip install -e ".[trainer]"
```

This repository does not contain a CUDA installer or model weights. The package
is not published on PyPI; install from this Git checkout.

For an installation you do not intend to edit locally:

```bash
python -m pip install --no-deps "git+https://github.com/schonsense/NLPO.git"
```

## First test: no GPU or model download

```bash
python examples/tiny_cpu.py --steps 4
```

This creates a tiny random causal Transformer, reports the violation counts,
trains it, evaluates it, and saves its weights and objective configuration to a
temporary directory. It tests the pipeline, not model quality.

## Import the trainer

Use your existing model/tokenizer loading code, including its working attention
implementation, precision, adapters, and trainable-parameter selection. Then:

```python
from transformers import TrainingArguments
from token_preference import (
    ChosenDataCollator,
    PreferenceConfig,
    PreferenceTrainer,
    prepare_dataset,
)

# model and tokenizer are the objects loaded by your existing training setup.
records = [
    {
        "messages": [
            {"role": "user", "content": "Continue the scene with one sentence."},
            {"role": "assistant", "content": "The latch clicked, but nobody entered."},
        ]
    }
]
dataset = prepare_dataset(records, tokenizer)
collator = ChosenDataCollator(tokenizer=tokenizer)

trainer = PreferenceTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    data_collator=collator,
    preference_config=PreferenceConfig(
        probability_floor=0.05,
        alternative_ratio=3.0,
        negative_weight=1.0,
        floor_weight=1.0,
        chunk_size=128,
    ),
    args=TrainingArguments(
        output_dir="outputs/experiment_01",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-6,
        num_train_epochs=1,
        bf16=True,
        average_tokens_across_devices=True,
        report_to="none",
    ),
)

print(trainer.inspect_batch(collator(dataset[:1])))
trainer.train()
trainer.save_model()
```

The training settings above are an API example, not a tuned recipe or a claim
that an arbitrary model will fit. `inspect_batch` is an explicit diagnostic
forward with no update; it is not used to precompute training negatives.

For a compatible native Hugging Face model, a complete loading example is also
provided:

```bash
python examples/train_hf.py --model /path/to/model --data /path/to/data.jsonl --output-dir outputs/run01 --probability-floor 0.05 --alternative-ratio 3 --gradient-checkpointing
```

That example uses BF16 CUDA and disables PyTorch's math SDPA fallback. If a fused
kernel is unavailable it should fail rather than select the math kernel. It
cannot certify architecture-specific mask construction in remote model code.
For Ripple or another custom architecture, the recommended entrance is the
Python API with the model from your verified training loader.

## Import only the loss

This is the extension point for an existing training loop or another framework:

```python
from token_preference import GatedUnlikelihoodLoss, PreferenceConfig

criterion = GatedUnlikelihoodLoss(PreferenceConfig())
outputs = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    use_cache=False,
)
loss = criterion(
    outputs.logits[:, :-1, :],  # position i predicts token i+1
    batch["labels"][:, 1:],
)
loss.backward()                # conditional correction is constructed here
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

The standalone loss expects **already aligned** logits and labels. It does not
shift them itself. For gradient accumulation, supply `normalizer=` with the
eligible-token count over the entire intended update and handle your framework's
distributed gradient scaling. The supplied Trainer does this through the native
Transformers loss callback.

## Exact objective

For chosen token `c` at an eligible position:

```text
a = argmax over every vocabulary token except c
p = P(c | original chosen prefix)
q = P(a | original chosen prefix)
active = (p < probability_floor) OR (q > alternative_ratio * p)

loss = negative_weight * active * -log(1 - q)
     + floor_weight * max(0, log(probability_floor) - log(p))
```

Probabilities are the model's native T=1 softmax distribution. There is no
top-k/top-p filtering, repetition penalty, special-token exclusion, or reroll.
Tied alternatives use the lowest token ID. Inequalities are strict in the
working floating-point dtype, with no hidden boundary tolerance.

| Chosen | Best other | Direct correction |
| ---: | ---: | --- |
| 4% | 3.5% | Selected-other penalty plus chosen floor support |
| 8% | 30% | Selected-other penalty |
| 2% | 40% | Both terms |
| 5% | 14% | Neither term |
| 5% | 15% | Neither term at exact equality |

The denominator is **all eligible chosen targets**, including currently inactive
positions. It is not the number of violations. When both bounds are satisfied,
this objective supplies zero direct target loss at that position.

The custom `autograd.Function.forward` computes the real scalar objective for
logging/evaluation. Its `backward` independently selects candidates and evaluates
the gates again from the retained, unchanged forward logits. That is not a
second model forward. The candidate IDs and gates are fixed for differentiation
within that snapshot. Optimizer updates happen only in the surrounding trainer.

## Dataset contract

`prepare_example` and `prepare_dataset` accept:

- `prompt` / `chosen` records; an existing `rejected` field is ignored.
- Conversational prompts and assistant-message chosen values.
- `messages` records, with the final assistant message as the chosen content.
- ShareGPT `conversations`, including `human` / `gpt` role names.
- Pretokenized `input_ids`, `attention_mask`, and original-token `labels`.

Earlier messages, prompt text, padding, and structural assistant suffixes receive
`-100` labels. The chosen literal content is supervised. An EOS spelling that is
literally part of the chosen content remains eligible; a template-added EOS is
masked. Column zero is masked because it has no preceding prediction position.

Raw text preparation requires a fast tokenizer with offsets. Native chat
templates are rendered with the actual chosen message, then checked against a
placeholder rendering to locate its exact content span. Token boundaries that
cross the content boundary, content-changing templates, ambiguous records, and
trailing non-assistant turns produce an error rather than silently changing the
example. Pretokenized chosen rows provide an escape hatch for a specialized
template, with responsibility for its labels remaining with the caller.

`max_length=None` adds no preprocessing limit. An explicit `max_length` errors on
overflow by default; `on_overflow="skip"` is an explicit alternative that reports
the skip count. No truncation is performed.

The collator can prepare raw rows on the fly when given a tokenizer. If using
that route in Trainer, set `remove_unused_columns=False` so Transformers does not
discard `prompt`, `chosen`, or `messages` before collation. Preparing rows once
with `prepare_dataset` is the recommended default.

## Configuration and diagnostics

| Setting | Default | Meaning |
| --- | ---: | --- |
| `probability_floor` | `0.05` | Activate below this chosen probability |
| `alternative_ratio` | `3.0` | Activate above this best-other/chosen ratio |
| `negative_weight` | `1.0` | Selected-alternative unlikelihood strength |
| `floor_weight` | `1.0` | Chosen support below the floor |
| `chunk_size` | `128` | Rows per FP32 loss-computation chunk |
| `ignore_index` | `-100` | Ignored labels; fixed to this value in the HF adapter |

Weights can be zero for explicit ablations, but not both. The floor and ratio are
independent of the learning rate and loss weights.

`loss_statistics(aligned_logits, aligned_labels, config)` and
`trainer.inspect_batch(batch)` report eligible and active counts, separate floor
and ratio violations, overlap, mean chosen/alternative probabilities, and loss
components. Diagnostics perform another vocabulary scan and are explicit calls.
Native Trainer logs its normal training loss; native `eval_loss` aggregates batch
losses and is not an exact global token-weighted mean for uneven target lengths.

## Checkpoints and resumption

`save_model()` and scheduled checkpoints include `preference_config.json` next
to the model. A resume checks that the objective settings match:

```python
trainer.train(resume_from_checkpoint="outputs/run01/checkpoint-100")
```

For a deliberate objective change, construct the trainer with
`allow_objective_change_on_resume=True`. That permits the change while retaining
the checkpoint's optimizer/trainer state. Loading weights as a new experiment is
also supported through your usual model-loading code.

## Compatibility and memory boundaries

- The independent loss works on floating logits from an ordinary PyTorch model.
  It supports first-order backward only, and handles upstream loss scaling.
- The HF adapter expects a causal model returning `logits` for every input
  position, original-token labels, and a 2-D token attention mask. It is not an
  encoder-decoder, multimodal, packed-sequence, or sequence-parallel adapter.
- A preconfigured trainable adapter or selected trainable parameters can be
  passed in. The package does not choose or silently alter trainable modules.
- Attention implementation and architecture patches belong to the model loader.
  The library neither selects a dense attention backend nor replaces a working
  Flash/Flex implementation. Passing an inefficient model will not make it
  efficient merely by changing the trainer.
- Loss intermediates use FP32 in chunks; FP64 stays FP64 for numerical checks.
  The **model still materializes its output logits and their full gradient**.
  This release is not a fused linear-head loss and does not establish long-context
  65B-model memory feasibility.
- The HF adapter integrates native accumulation and DDP token normalization.
  Implicit DataParallel is rejected. Actual multi-process GPU validation remains
  outstanding; a source-compatible path is not a measured production result.
- No claim of compatibility with `torch.compile`, higher-order gradients, or
  alternate distributed model-partitioning systems is made in this release.

## Contributing and GitHub

The [NLPO repository](https://github.com/schonsense/NLPO) includes an MIT license,
tests, a CPU smoke example, and a GitHub Actions workflow. Use a fork and pull
request to propose changes. No package-registry release is implied.

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m build
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the behavioral contract and where to
change the loss. Model checkpoints and datasets should not be committed.

## Research basis

The negative-token term comes from [Neural Text Generation with Unlikelihood
Training](https://arxiv.org/abs/1908.04319). The strongest-other OR gate,
floor-limited positive support, and backward-time selection are this project's
experimental combination. They are not a published, validated recipe for Ripple.

For comparison, [DPO](https://arxiv.org/abs/2305.18290) and
[IPO](https://arxiv.org/abs/2310.12036) define different preference objectives.
This package currently implements the agreed unlikelihood-plus-floor objective,
not a switch that silently substitutes those objectives.
