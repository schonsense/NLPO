"""Configuration for the snapshot-based token preference objective."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Mapping


@dataclass(frozen=True)
class PreferenceConfig:
    """Loss settings, independent of a tokenizer, model, or training framework.

    A position is active when the chosen probability is strictly below
    ``probability_floor`` OR the strongest other token has probability strictly
    above ``alternative_ratio * chosen_probability``. Zero component weights are
    useful for ablations; at least one component must have positive weight.
    """

    probability_floor: float = 0.05
    alternative_ratio: float = 3.0
    negative_weight: float = 1.0
    floor_weight: float = 1.0
    ignore_index: int = -100
    chunk_size: int = 128

    def __post_init__(self) -> None:
        for name in (
            "probability_floor",
            "alternative_ratio",
            "negative_weight",
            "floor_weight",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a finite real number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0.0 < self.probability_floor < 1.0:
            raise ValueError("probability_floor must be strictly between 0 and 1")
        if self.alternative_ratio <= 0.0:
            raise ValueError("alternative_ratio must be positive")
        if self.negative_weight < 0.0 or self.floor_weight < 0.0:
            raise ValueError("loss weights must be nonnegative")
        if self.negative_weight == 0.0 and self.floor_weight == 0.0:
            raise ValueError("at least one loss weight must be positive")
        if isinstance(self.ignore_index, bool) or not isinstance(self.ignore_index, int):
            raise TypeError("ignore_index must be an integer")
        if isinstance(self.chunk_size, bool) or not isinstance(self.chunk_size, int):
            raise TypeError("chunk_size must be an integer")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible settings for experiment/checkpoint metadata."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "PreferenceConfig":
        """Construct settings, rejecting unknown fields rather than ignoring them."""
        if not isinstance(values, Mapping):
            raise TypeError("settings must be a mapping")
        return cls(**dict(values))
