"""Chosen-only token preference training.

The loss requires only PyTorch. The Trainer adapter is imported lazily, so
Transformers and Accelerate are optional for an existing custom training loop.
"""

from .config import PreferenceConfig
from .data import ChosenDataCollator, prepare_dataset, prepare_example
from .loss import GatedUnlikelihoodLoss, loss_statistics

__version__ = "0.1.0"
__all__ = [
    "PreferenceConfig",
    "GatedUnlikelihoodLoss",
    "loss_statistics",
    "PreferenceTrainer",
    "ChosenDataCollator",
    "prepare_example",
    "prepare_dataset",
]


def __getattr__(name):
    if name == "PreferenceTrainer":
        from .trainer import PreferenceTrainer

        return PreferenceTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
