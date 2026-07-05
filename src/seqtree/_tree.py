from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._distributions import EmpiricalDistribution, entropy
from ._utils import is_number


@dataclass
class Split:
    feature: str
    threshold: float | None = None
    category: Any | None = None

    def go_left(self, row: dict[str, Any]) -> bool:
        value = row.get(self.feature)
        if self.threshold is not None:
            return is_number(value) and value <= self.threshold
        return value == self.category

    @property
    def label(self) -> str:
        if self.threshold is not None:
            return f"{self.feature} <= {self.threshold:g}"
        return f"{self.feature} == {self.category!r}"


@dataclass
class TreeNode:
    distribution: EmpiricalDistribution
    split: Split | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.split is None


class ConditionalTree:
    """Small CART-style conditional sampler used internally by SeqTree."""

    def __init__(
        self,
        *,
        max_depth: int = 5,
        min_samples_leaf: int = 5,
        min_impurity_decrease: float = 1e-9,
        max_thresholds: int = 16,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_thresholds = max_thresholds

    def fit(self, records: list[dict[str, Any]], features: list[str], target: str) -> "ConditionalTree":
        self.features_ = list(features)
        self.target_ = target
        self.root_ = self._fit_node(records, depth=0)
        return self

    def sample(self, row: dict[str, Any], rng) -> Any:
        node = self.root_
        while not node.is_leaf:
            node = node.left if node.split.go_left(row) else node.right
        return node.distribution.sample(rng)

    def leaf_probability(self, row: dict[str, Any], value: Any, alpha: float = 1.0) -> float:
        node = self.root_
        while not node.is_leaf:
            node = node.left if node.split.go_left(row) else node.right
        return node.distribution.probability(value, alpha=alpha)

    def conditional_entropy(self, records: list[dict[str, Any]]) -> float:
        if not records:
            return 0.0
        total = 0.0
        for record in records:
            node = self.root_
            while not node.is_leaf:
                node = node.left if node.split.go_left(record) else node.right
            total += node.distribution.entropy()
        return total / len(records)

    def _fit_node(self, records: list[dict[str, Any]], depth: int) -> TreeNode:
        values = [record[self.target_] for record in records]
        distribution = EmpiricalDistribution(values)
        node = TreeNode(distribution=distribution)

        if depth >= self.max_depth or len(records) < 2 * self.min_samples_leaf or distribution.entropy() == 0:
            return node

        split, gain = self._best_split(records)
        if split is None or gain <= self.min_impurity_decrease:
            return node

        left_records, right_records = self._partition(records, split)
        node.split = split
        node.left = self._fit_node(left_records, depth + 1)
        node.right = self._fit_node(right_records, depth + 1)
        return node

    def _best_split(self, records: list[dict[str, Any]]) -> tuple[Split | None, float]:
        parent_entropy = entropy([record[self.target_] for record in records])
        best_split = None
        best_gain = 0.0
        for feature in self.features_:
            for split in self._candidate_splits(records, feature):
                left_records, right_records = self._partition(records, split)
                if len(left_records) < self.min_samples_leaf or len(right_records) < self.min_samples_leaf:
                    continue
                weighted_entropy = (
                    len(left_records) * entropy([record[self.target_] for record in left_records])
                    + len(right_records) * entropy([record[self.target_] for record in right_records])
                ) / len(records)
                gain = parent_entropy - weighted_entropy
                if gain > best_gain:
                    best_split = split
                    best_gain = gain
        return best_split, best_gain

    def _candidate_splits(self, records: list[dict[str, Any]], feature: str) -> list[Split]:
        values = [record.get(feature) for record in records if record.get(feature) is not None]
        unique = sorted(set(values)) if all(is_number(value) for value in values) else list(dict.fromkeys(values))
        if len(unique) <= 1:
            return []

        if all(is_number(value) for value in unique) and len(unique) > 2:
            thresholds = [(unique[i] + unique[i + 1]) / 2 for i in range(len(unique) - 1)]
            if len(thresholds) > self.max_thresholds:
                step = len(thresholds) / self.max_thresholds
                thresholds = [thresholds[int(i * step)] for i in range(self.max_thresholds)]
            return [Split(feature=feature, threshold=threshold) for threshold in thresholds]

        return [Split(feature=feature, category=value) for value in unique[:-1]]

    @staticmethod
    def _partition(records: list[dict[str, Any]], split: Split) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        left_records = []
        right_records = []
        for record in records:
            if split.go_left(record):
                left_records.append(record)
            else:
                right_records.append(record)
        return left_records, right_records
