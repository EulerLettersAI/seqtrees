from __future__ import annotations

from typing import Any

from ._distributions import EmpiricalDistribution
from ._utils import require_numeric_values


class LightGBMConditionalTree:
    """Conditional sampler backed by LightGBM's histogram regression trees."""

    def __init__(
        self,
        *,
        max_depth: int = 5,
        min_samples_leaf: int = 5,
        min_impurity_decrease: float = 1e-9,
        max_thresholds: int = 16,
        random_state: int | None = None,
        n_jobs: int | None = None,
    ) -> None:
        del max_thresholds
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, records: list[dict[str, Any]], features: list[str], target: str) -> "LightGBMConditionalTree":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("Install seqtree[lightgbm] to use tree_backend='lightgbm'.") from exc

        self.features_ = list(features)
        self.target_ = target
        target_values = [record[target] for record in records]
        self.marginal_ = EmpiricalDistribution(target_values)
        require_numeric_values(records, self.features_ + [target], backend="lightgbm")

        if self.max_depth == 0 or not self.features_ or len(set(target_values)) <= 1:
            self.model_ = None
            self.leaf_distributions_ = {}
            return self

        x_matrix = [self._encode_features(record) for record in records]

        max_depth = -1 if self.max_depth is None else self.max_depth
        num_leaves = 31 if max_depth <= 0 else max(2, min(31, 2**max_depth))
        self.model_ = LGBMRegressor(
            boosting_type="gbdt",
            objective="regression",
            n_estimators=32,
            learning_rate=0.2,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=self.min_samples_leaf,
            min_split_gain=self.min_impurity_decrease,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=-1,
        )
        self.model_.fit(x_matrix, target_values)

        grouped_values: dict[tuple[int, ...], list[Any]] = {}
        for leaf_id, value in zip(self._leaf_ids(x_matrix), target_values):
            grouped_values.setdefault(leaf_id, []).append(value)
        self.leaf_distributions_ = {
            leaf_id: EmpiricalDistribution(values) for leaf_id, values in grouped_values.items()
        }
        return self

    def sample(self, row: dict[str, Any], rng, *, interpolate: bool = False) -> Any:
        distribution = self._distribution_for_row(row)
        if interpolate:
            return distribution.sample_interpolated(rng)
        return distribution.sample(rng)

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
        leaf_id = self._leaf_ids([self._encode_features(row)])[0]
        return self.leaf_distributions_.get(leaf_id, self.marginal_)

    def _encode_features(self, record: dict[str, Any]) -> list[int | float]:
        return [float("nan") if record.get(feature) is None else record.get(feature) for feature in self.features_]

    def _leaf_ids(self, x_matrix: list[list[int | float]]) -> list[tuple[int, ...]]:
        leaves = self.model_.predict(x_matrix, pred_leaf=True)
        result = []
        for row in leaves:
            if hasattr(row, "tolist"):
                row = row.tolist()
            if isinstance(row, list):
                result.append(tuple(int(value) for value in row))
            else:
                result.append((int(row),))
        return result
