# API Reference

## `SequentialTreeSynthesizer`

```python
SequentialTreeSynthesizer(
    variable_order=None,
    optimize_order=False,
    max_depth=5,
    min_samples_leaf=5,
    min_impurity_decrease=1e-9,
    max_thresholds=16,
    tree_backend="auto",
    n_jobs=1,
    random_state=None,
)
```

### Parameters

- `variable_order`: optional sequence of column names or integer positions.
  When provided, every input column must appear exactly once.
- `optimize_order`: when `True`, learn a greedy variable order. Ignored when
  `variable_order` is provided.
- `max_depth`: maximum depth for each conditional tree.
- `min_samples_leaf`: minimum samples required in each child after a split.
- `min_impurity_decrease`: minimum entropy reduction required to split.
- `max_thresholds`: maximum candidate thresholds per numeric feature.
- `tree_backend`: `"auto"`, `"native"`, `"sklearn"`, or `"lightgbm"`.
  `"auto"` uses LightGBM when installed, then scikit-learn, and otherwise
  falls back to the pure-Python backend.
- `n_jobs`: parallel workers for fitting conditional trees and scoring greedy
  ordering candidates. Use `-1` for all available cores.
- `random_state`: seed used for reproducible sampling.

### Methods

- `fit(X, y=None)`: fit the sequential tree model and return `self`.
- `sample(n_samples=1, random_state=None, as_dataframe=None, n_jobs=None)`:
  generate synthetic rows. `n_jobs=None` reuses the fitted model's worker count.
- `fit_sample(X, n_samples, random_state=None)`: fit and sample in one call.
- `get_variable_order()`: return the learned order.
- `to_puml()`: render the learned dependency chain as PlantUML.

### Learned Attributes

- `feature_names_in_`: input column names.
- `n_features_in_`: number of input columns.
- `variable_order_`: final variable order used by the model.
- `marginal_`: empirical distribution for the first variable.
- `trees_`: mapping from later variables to fitted conditional trees.
- `n_jobs_`: resolved worker count used during fitting.
