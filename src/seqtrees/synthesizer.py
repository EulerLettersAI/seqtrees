from __future__ import annotations

import random
from typing import Any, Sequence

from ._backends import SUPPORTED_TREE_BACKENDS, create_tree_backend
from ._distributions import EmpiricalDistribution, entropy
from ._parallel import parallel_map, resolve_n_jobs
from ._utils import (
    is_float_value,
    is_integer_code,
    normalize_table,
    resolve_column_subset,
    resolve_columns,
    validate_no_nulls,
    validate_variable_types,
)


class SequentialTreeSynthesizer:
    """Sequential tree synthesizer with a scikit-learn-like estimator API.

    The fitted model factorizes a table as:

    ``P(X_1, ..., X_d) = P(X_1) * P(X_2 | X_1) * ... * P(X_d | X_1, ..., X_{d-1})``.

    Each conditional distribution is represented by an empirical tree whose
    leaves store observed target values. Pandas DataFrame input is transformed
    with ifcfill using label encoding for categorical variables. List-based
    input must already be numeric: continuous variables as floats and discrete
    variables as integer category codes.

    Parameters
    ----------
    variable_order:
        Optional sequence of column names or integer positions. When supplied,
        it must contain every input column exactly once.
    optimize_order:
        If ``True`` and ``variable_order`` is not supplied, choose a greedy
        variable order using tree-estimated conditional entropy.
    max_depth:
        Maximum depth for each conditional tree.
    min_samples_leaf:
        Minimum number of records required in each child after a split.
    min_impurity_decrease:
        Minimum impurity decrease required for a tree split.
    max_thresholds:
        Maximum number of numeric threshold candidates considered by the native
        backend for each feature.
    tree_backend:
        Tree backend to use: ``"auto"``, ``"native"``, ``"sklearn"``, or
        ``"lightgbm"``. ``"auto"`` prefers LightGBM when installed, then
        scikit-learn, then the native backend.
    continuous_strategy:
        ``"empirical"`` samples observed values from each leaf. ``"interpolate"``
        can generate interpolated values for columns listed in, or inferred by,
        ``continuous_columns``.
    continuous_columns:
        Optional subset of column names or integer positions to treat as
        continuous float variables. If either ``continuous_columns`` or
        ``discrete_columns`` is supplied, the two lists must classify every
        input column exactly once.
    discrete_columns:
        Optional subset of column names or integer positions that must contain
        integer category codes. Binary variables are treated as ordinary
        discrete variables. One-hot encoded inputs are intentionally out of
        scope.
    n_jobs:
        Number of parallel workers. Use ``-1`` to use all available cores.
    random_state:
        Optional seed for reproducible sampling.

    Attributes
    ----------
    feature_names_in_:
        Column names seen during fitting.
    n_features_in_:
        Number of columns seen during fitting.
    variable_order_:
        Learned or user-supplied variable order.
    continuous_columns_:
        Set of columns treated as continuous for interpolation.
    integer_columns_:
        Set of original DataFrame columns identified by ifcfill as integer
        variables. These columns are numeric, not categorical labels.
    categorical_columns_:
        Set of original DataFrame columns identified by ifcfill as categorical
        variables and label-encoded for modelling.
    discrete_columns_:
        Set of columns treated as discrete integer category codes.
        For raw DataFrame input, this is the set of categorical source columns.
    preprocessed_data_:
        DataFrame used internally to fit the sequential tree model after any
        ifcfill transformation.
    marginal_:
        Empirical distribution for the first variable in ``variable_order_``.
    trees_:
        Mapping from later variable names to fitted conditional tree backends.
    n_jobs_:
        Resolved worker count used during fitting.
    """

    def __init__(
        self,
        *,
        variable_order: Sequence[str | int] | None = None,
        optimize_order: bool = False,
        max_depth: int = 5,
        min_samples_leaf: int = 5,
        min_impurity_decrease: float = 1e-9,
        max_thresholds: int = 16,
        tree_backend: str = "auto",
        continuous_strategy: str = "empirical",
        continuous_columns: Sequence[str | int] | None = None,
        discrete_columns: Sequence[str | int] | None = None,
        n_jobs: int | None = 1,
        random_state: int | None = None,
    ) -> None:
        self.variable_order = variable_order
        self.optimize_order = optimize_order
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_thresholds = max_thresholds
        self.tree_backend = tree_backend
        self.continuous_strategy = continuous_strategy
        self.continuous_columns = continuous_columns
        self.discrete_columns = discrete_columns
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> "SequentialTreeSynthesizer":
        """Fit the sequential tree model.

        Parameters
        ----------
        X:
            A pandas DataFrame, list of dictionaries, or list of row sequences.
            DataFrames are transformed with ifcfill. List-based input must
            contain floats for continuous variables and integers for discrete
            variables. Null values are not accepted for list-based input.
        y:
            Ignored. Present for compatibility with estimator conventions.

        Returns
        -------
        SequentialTreeSynthesizer
            The fitted estimator.
        """
        del y
        self._validate_params()
        prepared_X, dataframe_input = self._prepare_input(X)
        records, columns, _ = normalize_table(prepared_X)
        if not columns:
            raise ValueError("X must contain at least one column.")
        self.preprocessed_data_ = self._preprocessed_dataframe(prepared_X, records, columns)
        self.n_jobs_ = resolve_n_jobs(self.n_jobs)
        validate_no_nulls(records, columns)
        continuous_columns, discrete_columns, integer_columns = self._resolve_variable_types(records, columns)
        validate_variable_types(
            records,
            continuous_columns=continuous_columns,
            discrete_columns=discrete_columns,
            integer_columns=integer_columns,
        )

        if self.variable_order is not None:
            order = resolve_columns(columns, self.variable_order)
        elif self.optimize_order:
            order = self._optimize_order(records, columns)
        else:
            order = list(columns)

        self.feature_names_in_ = list(columns)
        self.n_features_in_ = len(columns)
        self.variable_order_ = order
        self.continuous_columns_ = continuous_columns
        self.discrete_columns_ = discrete_columns
        self.integer_columns_ = integer_columns
        self.dataframe_input_ = dataframe_input
        self.marginal_ = EmpiricalDistribution([record[order[0]] for record in records])
        self.trees_: dict[str, Any] = {}
        self._backend_n_jobs_ = 1 if self.n_jobs_ > 1 and len(order[1:]) > 1 else self.n_jobs_

        fitted_trees = parallel_map(
            lambda item: self._fit_conditional_tree(records, order, item),
            list(enumerate(order[1:], start=1)),
            self.n_jobs_,
        )
        self.trees_ = {target: tree for target, tree in fitted_trees}

        self.is_fitted_ = True
        return self

    def sample(
        self,
        n_samples: int = 1,
        *,
        random_state: int | None = None,
        as_dataframe: bool | None = None,
        n_jobs: int | None = None,
    ) -> Any:
        """Generate synthetic rows from the fitted model.

        Parameters
        ----------
        n_samples:
            Number of synthetic rows to generate.
        random_state:
            Optional seed for this sampling call. If omitted, the estimator's
            ``random_state`` is used.
        as_dataframe:
            If ``True``, return a pandas DataFrame. If ``False``, return a list
            of dictionaries. If omitted, SeqTrees returns a DataFrame when the
            model was fitted with DataFrame-like input.
        n_jobs:
            Optional worker count for row generation. If omitted, the fitted
            model's worker count is reused.

        Returns
        -------
        Any
            Synthetic rows as a list of dictionaries or pandas DataFrame.
        """
        self._check_is_fitted()
        if n_samples < 0:
            raise ValueError("n_samples must be non-negative.")

        rng = random.Random(self.random_state if random_state is None else random_state)
        seeds = [rng.randrange(0, 2**63) for _ in range(n_samples)]
        sample_jobs = self.n_jobs_ if n_jobs is None else resolve_n_jobs(n_jobs)
        rows = parallel_map(self._sample_one, seeds, sample_jobs)
        rows = self._restore_rows(rows, random_state=random_state)

        if as_dataframe is None:
            as_dataframe = self.dataframe_input_
        if as_dataframe:
            try:
                import pandas as pd
            except ImportError as exc:
                raise ImportError("Install seqtrees[pandas] to return pandas DataFrames.") from exc
            return pd.DataFrame(rows, columns=self.feature_names_in_)
        return rows

    def fit_sample(self, X: Any, n_samples: int, *, random_state: int | None = None) -> Any:
        """Fit the model and immediately sample synthetic rows.

        Parameters
        ----------
        X:
            Training data accepted by :meth:`fit`.
        n_samples:
            Number of synthetic rows to generate.
        random_state:
            Optional seed for the sampling call.

        Returns
        -------
        Any
            Synthetic rows as a list of dictionaries or pandas DataFrame.
        """
        return self.fit(X).sample(n_samples, random_state=random_state)

    def get_variable_order(self) -> list[str]:
        """Return the learned or user-specified variable order.

        Returns
        -------
        list[str]
            Variable names in the order used by the sequential model.
        """
        self._check_is_fitted()
        return list(self.variable_order_)

    def get_preprocessed_data(self, *, copy: bool = True) -> Any:
        """Return the model-ready DataFrame used during fitting.

        For pandas DataFrame input, this is the output of SeqTrees' internal
        ifcfill transformation. For list-based input, this is the normalized
        numeric table used by the tree model.

        Parameters
        ----------
        copy:
            If ``True``, return a defensive copy. If ``False``, return the
            stored fitted DataFrame directly.

        Returns
        -------
        Any
            The preprocessed pandas DataFrame used to fit the sequential tree
            model.
        """
        self._check_is_fitted()
        if copy:
            return self.preprocessed_data_.copy()
        return self.preprocessed_data_

    def to_puml(self) -> str:
        """Render the fitted sequential dependency chain as PlantUML.

        Returns
        -------
        str
            PlantUML source for the fitted variable-order dependency chain.
        """
        self._check_is_fitted()
        lines = ["@startuml", "title SeqTrees learned sequential model", "left to right direction"]
        for column in self.variable_order_:
            lines.append(f'node "{column}" as {self._puml_id(column)}')
        for left, right in zip(self.variable_order_, self.variable_order_[1:]):
            lines.append(f"{self._puml_id(left)} --> {self._puml_id(right)}")
        lines.append("@enduml")
        return "\n".join(lines)

    def _optimize_order(self, records: list[dict[str, Any]], columns: list[str]) -> list[str]:
        remaining = set(columns)
        order = [min(columns, key=lambda column: entropy([record[column] for record in records]))]
        remaining.remove(order[0])

        while remaining:
            self._backend_n_jobs_ = 1 if self.n_jobs_ > 1 and len(remaining) > 1 else self.n_jobs_
            scores = parallel_map(
                lambda candidate: (candidate, self._order_candidate_score(records, order, candidate)),
                sorted(remaining),
                self.n_jobs,
            )
            best_column, _ = min(scores, key=lambda item: item[1])
            order.append(best_column)
            remaining.remove(best_column)
        return order

    def _fit_conditional_tree(
        self,
        records: list[dict[str, Any]],
        order: list[str],
        item: tuple[int, str],
    ) -> tuple[str, Any]:
        index, target = item
        tree = create_tree_backend(
            self.tree_backend,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_thresholds=self.max_thresholds,
            random_state=None if self.random_state is None else self.random_state + index,
            n_jobs=self._backend_n_jobs(),
        )
        tree.fit(records, order[:index], target)
        return target, tree

    def _order_candidate_score(self, records: list[dict[str, Any]], order: list[str], candidate: str) -> float:
        tree = create_tree_backend(
            self.tree_backend,
            max_depth=min(self.max_depth, 3),
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_thresholds=self.max_thresholds,
            random_state=self.random_state,
            n_jobs=self._backend_n_jobs(),
        ).fit(records, order, candidate)
        return tree.conditional_entropy(records)

    def _sample_one(self, seed: int) -> dict[str, Any]:
        rng = random.Random(seed)
        row = {}
        first = self.variable_order_[0]
        if self._should_interpolate(first):
            row[first] = self.marginal_.sample_interpolated(rng)
        else:
            row[first] = self.marginal_.sample(rng)
        for target in self.variable_order_[1:]:
            row[target] = self.trees_[target].sample(row, rng, interpolate=self._should_interpolate(target))
        return {column: row[column] for column in self.feature_names_in_}

    def _backend_n_jobs(self) -> int:
        return getattr(self, "_backend_n_jobs_", getattr(self, "n_jobs_", 1))

    def _should_interpolate(self, column: str) -> bool:
        numeric_columns = self.continuous_columns_ | self.integer_columns_
        return self.continuous_strategy == "interpolate" and column in numeric_columns

    @staticmethod
    def _preprocessed_dataframe(prepared_X: Any, records: list[dict[str, Any]], columns: list[str]) -> Any:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("Install pandas to inspect preprocessed SeqTrees data.") from exc

        if hasattr(prepared_X, "columns") and hasattr(prepared_X, "copy"):
            frame = prepared_X.copy()
            frame.columns = [str(column) for column in frame.columns]
            return frame
        return pd.DataFrame(records, columns=columns)

    def _prepare_input(self, X: Any) -> tuple[Any, bool]:
        if not self._should_use_ifcfill(X):
            self.ifc_transformer_ = None
            self.ifc_column_types_ = {}
            self.categorical_columns_ = set()
            return X, False

        try:
            from ifcfill import IFCTransformer
        except ImportError as exc:
            raise ImportError("Install ifcfill>=0.3.4 to fit raw pandas DataFrames.") from exc

        dataframe = X.copy()
        dataframe.columns = [str(column) for column in dataframe.columns]
        transformer = IFCTransformer(cat_encoding="label", n_jobs=self.n_jobs)
        transformed = transformer.fit_transform(dataframe)
        transformed.columns = [str(column) for column in transformed.columns]

        self.ifc_transformer_ = transformer
        self.ifc_column_types_ = {
            str(column): str(column_type)
            for column, column_type in transformer.column_types_.items()
        }
        self.categorical_columns_ = {
            column for column, column_type in self.ifc_column_types_.items() if column_type == "categorical"
        }
        return transformed, True

    @staticmethod
    def _should_use_ifcfill(X: Any) -> bool:
        return hasattr(X, "columns") and hasattr(X, "to_dict") and hasattr(X, "__len__")

    def _restore_rows(self, rows: list[dict[str, Any]], *, random_state: int | None) -> list[dict[str, Any]]:
        if self.ifc_transformer_ is None:
            return rows

        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("Install pandas to restore ifcfill-transformed samples.") from exc

        frame = pd.DataFrame(rows, columns=self.feature_names_in_)
        for column in self.integer_columns_ | self.discrete_columns_:
            if column in frame:
                frame[column] = frame[column].round().astype("int64")

        restored = self.ifc_transformer_.inverse_transform(
            frame,
            random_state=self.random_state if random_state is None else random_state,
        )
        restored.columns = [str(column) for column in restored.columns]
        return restored.to_dict("records")

    def _resolve_variable_types(
        self,
        records: list[dict[str, Any]],
        columns: list[str],
    ) -> tuple[set[str], set[str], set[str]]:
        if self.ifc_transformer_ is not None:
            if self.continuous_columns is not None or self.discrete_columns is not None:
                return (*self._resolve_declared_variable_types(columns), set())
            continuous_columns = {
                column for column in columns if self.ifc_column_types_.get(column) == "float"
            }
            integer_columns = {
                column for column in columns if self.ifc_column_types_.get(column) in {"integer", "datetime"}
            }
            discrete_columns = {
                column for column in columns if self.ifc_column_types_.get(column) == "categorical"
            }
            unknown = set(columns) - continuous_columns - integer_columns - discrete_columns
            if unknown:
                continuous_columns.update(unknown)
            return continuous_columns, discrete_columns, integer_columns

        if self.continuous_columns is None and self.discrete_columns is None:
            continuous_columns, discrete_columns = self._infer_variable_types(records, columns)
            return continuous_columns, discrete_columns, set()

        continuous_columns, discrete_columns = self._resolve_declared_variable_types(columns)
        return continuous_columns, discrete_columns, set()

    def _resolve_declared_variable_types(self, columns: list[str]) -> tuple[set[str], set[str]]:
        if self.continuous_columns is None and self.discrete_columns is None:
            return set(), set()

        continuous_columns = (
            set(resolve_column_subset(columns, self.continuous_columns, parameter="continuous_columns"))
            if self.continuous_columns is not None
            else set()
        )
        discrete_columns = (
            set(resolve_column_subset(columns, self.discrete_columns, parameter="discrete_columns"))
            if self.discrete_columns is not None
            else set()
        )

        overlap = continuous_columns & discrete_columns
        if overlap:
            raise ValueError(f"Columns cannot be both continuous and discrete: {sorted(overlap)}")

        declared = continuous_columns | discrete_columns
        missing = set(columns) - declared
        extra = declared - set(columns)
        if missing or extra:
            raise ValueError(
                "continuous_columns and discrete_columns must classify every column exactly once; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        return continuous_columns, discrete_columns

    @staticmethod
    def _infer_variable_types(records: list[dict[str, Any]], columns: list[str]) -> tuple[set[str], set[str]]:
        continuous_columns = set()
        discrete_columns = set()
        for column in columns:
            values = [record[column] for record in records]
            if all(is_float_value(value) for value in values):
                continuous_columns.add(column)
            elif all(is_integer_code(value) for value in values):
                discrete_columns.add(column)
            else:
                raise TypeError(
                    f"Could not infer variable type for column {column!r}. "
                    "SeqTrees accepts only float continuous variables and integer-coded discrete variables; "
                    "pass continuous_columns and discrete_columns explicitly if needed."
                )
        return continuous_columns, discrete_columns

    def _validate_params(self) -> None:
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative.")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1.")
        if self.max_thresholds < 1:
            raise ValueError("max_thresholds must be at least 1.")
        if self.tree_backend not in SUPPORTED_TREE_BACKENDS:
            raise ValueError(f"tree_backend must be one of {sorted(SUPPORTED_TREE_BACKENDS)}.")
        if self.continuous_strategy not in {"empirical", "interpolate"}:
            raise ValueError("continuous_strategy must be 'empirical' or 'interpolate'.")
        resolve_n_jobs(self.n_jobs)

    def _check_is_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise ValueError("This SequentialTreeSynthesizer instance is not fitted yet. Call fit first.")

    @staticmethod
    def _puml_id(name: str) -> str:
        return "v_" + "".join(char if char.isalnum() else "_" for char in str(name))
