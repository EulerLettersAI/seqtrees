from __future__ import annotations

from typing import Any

from ._distributions import EmpiricalDistribution
from ._utils import require_numeric_values


class SklearnConditionalTree:
    """Conditional sampler backed by scikit-learn's compiled regression tree."""

    def __init__(
        self,
        *,
        max_depth: int = 5,
        min_samples_leaf: int = 5,
        min_impurity_decrease: float = 1e-9,
        max_thresholds: int = 16,
        random_state: int | None = None,
    ) -> None:
        del max_thresholds
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state

    def fit(self, records: list[dict[str, Any]], features: list[str], target: str) -> "SklearnConditionalTree":
        try:
            from sklearn.tree import DecisionTreeRegressor
        except ImportError as exc:
            raise ImportError("Install seqtree[sklearn] to use tree_backend='sklearn'.") from exc

        self.features_ = list(features)
        self.target_ = target
        target_values = [record[target] for record in records]
        self.marginal_ = EmpiricalDistribution(target_values)
        require_numeric_values(records, self.features_ + [target], backend="sklearn")

        if self.max_depth == 0 or not self.features_ or len(set(target_values)) <= 1:
            self.model_ = None
            self.leaf_distributions_ = {}
            return self

        x_matrix = [self._encode_features(record) for record in records]

        max_depth = None if self.max_depth is None else self.max_depth
        self.model_ = DecisionTreeRegressor(
            criterion="squared_error",
            max_depth=max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            random_state=self.random_state,
        )
        self.model_.fit(x_matrix, target_values)

        leaves = self.model_.apply(x_matrix)
        grouped_values: dict[int, list[Any]] = {}
        for leaf_id, value in zip(leaves, target_values):
            grouped_values.setdefault(int(leaf_id), []).append(value)
        self.leaf_distributions_ = {
            leaf_id: EmpiricalDistribution(values) for leaf_id, values in grouped_values.items()
        }
        return self

    def sample(self, row: dict[str, Any], rng) -> Any:
        return self._distribution_for_row(row).sample(rng)

    def leaf_probability(self, row: dict[str, Any], value: Any, alpha: float = 1.0) -> float:
        return self._distribution_for_row(row).probability(value, alpha=alpha)

    def conditional_entropy(self, records: list[dict[str, Any]]) -> float:
        if not records:
            return 0.0
        total = 0.0
        for record in records:
            total += self._distribution_for_row(record).entropy()
        return total / len(records)

    def _distribution_for_row(self, row: dict[str, Any]) -> EmpiricalDistribution:
        if self.model_ is None:
            return self.marginal_
        leaf_id = int(self.model_.apply([self._encode_features(row)])[0])
        return self.leaf_distributions_.get(leaf_id, self.marginal_)

    def _encode_features(self, record: dict[str, Any]) -> list[int]:
        return [record.get(feature) for feature in self.features_]
