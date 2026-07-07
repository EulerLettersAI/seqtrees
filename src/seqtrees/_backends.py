from __future__ import annotations

from typing import Any

from ._tree import ConditionalTree


SUPPORTED_TREE_BACKENDS = {"native", "sklearn", "lightgbm", "auto"}


def create_tree_backend(
    backend: str,
    *,
    max_depth: int,
    min_samples_leaf: int,
    min_impurity_decrease: float,
    max_thresholds: int,
    random_state: int | None = None,
    n_jobs: int | None = None,
) -> Any:
    if backend not in SUPPORTED_TREE_BACKENDS:
        raise ValueError(f"tree_backend must be one of {sorted(SUPPORTED_TREE_BACKENDS)}.")

    if backend == "sklearn":
        from ._sklearn_tree import SklearnConditionalTree

        return SklearnConditionalTree(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            max_thresholds=max_thresholds,
            random_state=random_state,
        )

    if backend == "lightgbm":
        from ._lightgbm_tree import LightGBMConditionalTree

        return LightGBMConditionalTree(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            max_thresholds=max_thresholds,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    if backend == "auto":
        try:
            from ._lightgbm_tree import LightGBMConditionalTree
            import lightgbm  # noqa: F401

            return LightGBMConditionalTree(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                min_impurity_decrease=min_impurity_decrease,
                max_thresholds=max_thresholds,
                random_state=random_state,
                n_jobs=n_jobs,
            )
        except ImportError:
            pass
        try:
            from ._sklearn_tree import SklearnConditionalTree
            import sklearn  # noqa: F401

            return SklearnConditionalTree(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                min_impurity_decrease=min_impurity_decrease,
                max_thresholds=max_thresholds,
                random_state=random_state,
            )
        except ImportError:
            pass

    return ConditionalTree(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_impurity_decrease=min_impurity_decrease,
        max_thresholds=max_thresholds,
    )
