# Verification record

Validation date: 2026-11-24.

Tests ran in isolated CPU environments in the authoring workspace, without
downloading a pretrained model or changing the user's training environment.

## Executed checks

| Check | Result |
| --- | --- |
| Core loss and gradient tests | 54 passed |
| Dataset alignment and collation tests | 39 passed |
| Trainer integration tests | 16 passed |
| Entire suite with Transformers 5.24.1 | 109 passed |
| Entire suite with Transformers 4.50.0, the declared lower bound | 109 passed |
| Tiny native causal Transformer | Three training updates, evaluation, and model/config save completed |
| Package build | Wheel and source distribution built successfully |
| Installed wheel | Tiny training/evaluation/save example completed using the installed artifact |
| Optional dependencies | Standalone loss imported and backpropagated with Transformers and Accelerate blocked |
| Generic loading example | CLI help entry point completed |

Both environments used Python 3.12 and CPU PyTorch 2.15.0; Accelerate 1.16.0 was
shared. The older-Transformers environment used its compatible Hub and tokenizer
packages. Runtime version numbers are recorded for reproduction, not a request
to replace a working CUDA installation.

The loss tests include independent ordinary-autograd comparisons, double-precision
finite-difference checks away from hard boundaries, FP16/BF16 input cases,
extreme probabilities, noncontiguous causal slices, explicit normalization, and
verification that candidate selection executes inside backward.

The data tests include real locally constructed fast tokenizers, Unicode/BPE
offsets, native chat templates, structural versus literal EOS handling,
prompt/content boundary rejection, original-token label checks, and pad=EOS.

The Trainer tests include a plain PyTorch model whose forward does not accept
labels, causal alignment, loss normalization, equivalence between unequal-length
accumulated microbatches and one combined optimizer update, evaluation, saving,
resumption, and rejection of an unintended objective change on resume.

The tiny Transformer is randomly initialized. Its successful execution is a
pipeline check, not evidence of useful language-model improvement.

## Not established by these checks

- CUDA kernel execution or throughput.
- Ripple-65B native-model compatibility and gradient parity.
- Long-context GPU memory feasibility.
- Multi-process GPU training or another model-partitioning system.
- Model-quality superiority over SFT, DPO, or another objective.
- Higher-order differentiation or torch.compile compatibility.

## Reproduce

From an installed checkout with the test dependencies available:

```bash
python -m pytest -q
python examples/tiny_cpu.py --steps 3
python -m build
```

Live CI results are recorded in [GitHub Actions](https://github.com/schonsense/NLPO/actions).
The results above describe the local validation performed before publication.
