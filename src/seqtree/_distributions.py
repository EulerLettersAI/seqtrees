from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any


class EmpiricalDistribution:
    """A finite empirical distribution over observed values."""

    def __init__(self, values: list[Any]):
        if not values:
            raise ValueError("EmpiricalDistribution requires at least one value.")
        counts = Counter(values)
        self.values_ = list(counts.keys())
        self.weights_ = [counts[value] for value in self.values_]
        self.n_samples_ = len(values)

    def entropy(self) -> float:
        total = sum(self.weights_)
        score = 0.0
        for weight in self.weights_:
            probability = weight / total
            score -= probability * math.log2(probability)
        return score

    def sample(self, rng: random.Random) -> Any:
        return rng.choices(self.values_, weights=self.weights_, k=1)[0]

    def probability(self, value: Any, alpha: float = 1.0) -> float:
        total = sum(self.weights_)
        size = len(self.values_)
        for candidate, weight in zip(self.values_, self.weights_):
            if candidate == value:
                return (weight + alpha) / (total + alpha * size)
        return alpha / (total + alpha * (size + 1))


def entropy(values: list[Any]) -> float:
    return EmpiricalDistribution(values).entropy()
