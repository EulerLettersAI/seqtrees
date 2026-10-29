# SeqTrees Documentation

SeqTrees builds synthetic tabular rows by learning an ordered product of
conditional distributions. Given tabular data with variables
$X_1, X_2, ..., X_d$, it estimates:

$$
P(X_1, X_2, ..., X_d) = P(X_1) P(X_2 | X_1) ... P(X_d | X_1, ..., X_{d-1})
$$

This implementation follows the sequential synthesis idea shown in [@khaled_el_mosquera_zheng_2020]: generate the first variable, append it to
the synthetic row, then use the generated prefix to synthesize the next
variable until the row is complete.

## Model

`SequentialTreeSynthesizer` uses:

- an empirical marginal distribution for the first variable;
- one conditional decision tree for each later variable;
- empirical leaf distributions that interpolate continuous numeric values by
  default while preserving observed codes for discrete variables.

SeqTrees accepts pandas DataFrames and performs no internal preprocessing by
default. Missing values are rejected and categorical columns must already be
integer encoded. Set `preprocessor="ifcfill"` to opt into
[ifcfill](https://github.com/EulerLettersAI/ifcfill) imputation and categorical
label encoding. IFCFill is the only supported internal preprocessor.
With IFCFill 0.5.1, generated missingness flags are modelled as discrete
variables so missingness can be learned jointly with the data.
Use `model.get_preprocessed_data()` after fitting to inspect the exact
model-ready DataFrame used by the sequential trees.

Without IFCFill, inputs must be model-ready DataFrames with no missing values:
continuous variables are floats and discrete variables are integer category
codes. Binary variables are ordinary discrete variables with codes `0` and `1`.

## Continuous Variables

SeqTrees' default sampling strategy is `"interpolate"`. Continuous float
variables, and numeric integer variables detected by IFCFill, can generate
interpolated values inside the reached distribution. Discrete variables still
sample observed category codes:

```python
model = SequentialTreeSynthesizer(
    continuous_columns=["age", "bmi"],
    discrete_columns=["sex_code", "risk_code"],
)
```

!!! warning "Training-record exposure with empirical sampling"

    Setting `continuous_strategy="empirical"` samples exact observed values and
    can reproduce complete training records. This exposes training data; do not
    use the empirical strategy when exact-record disclosure is unacceptable.

SeqTrees can infer variable types when both `continuous_columns` and
`discrete_columns` are omitted: all-float columns become continuous and all-int
integer-valued columns become discrete. With IFCFill enabled, SeqTrees uses
`IFCTransformer.column_types_`: original categorical columns stay discrete and
never generate unseen labels, while original integer columns are treated as
numeric integer variables. Prefer declaring both lists when inference is not
appropriate:

```python
model = SequentialTreeSynthesizer(
    continuous_columns=["age", "bmi"],
    discrete_columns=["sex_code", "income_bin", "risk_code"],
)
```

When either type list is supplied, the two lists must classify every input
column exactly once. One-hot encoded inputs are intentionally out of scope; keep
each categorical field as a single integer-coded variable before fitting
SeqTrees.

## Performance

SeqTrees can use either:

- `tree_backend="native"`: the pure-Python fallback with no required
  dependencies;
- `tree_backend="lightgbm"`: LightGBM's histogram-based gradient boosting
  regressor, using leaf-index buckets for empirical or interpolated sampling;
- `tree_backend="sklearn"`: scikit-learn's compiled `DecisionTreeRegressor`
  with empirical or interpolated leaf sampling;
- `tree_backend="auto"`: use LightGBM when installed, then scikit-learn, and
  otherwise fall back to the native backend.

Install the faster optional dependencies with:

```bash
python -m pip install "seqtrees[fast]"
```

Use LightGBM explicitly with:

```python
model = SequentialTreeSynthesizer(tree_backend="lightgbm", n_jobs=-1)
```

Set `n_jobs` to control parallelism. `n_jobs=1` runs serially, a positive
integer sets the worker count, and `n_jobs=-1` uses all detected CPU cores.
SeqTrees parallelizes conditional tree fitting across variables once the order is
known, parallelizes candidate scoring inside greedy order optimization, and can
parallelize synthetic row generation through `sample(..., n_jobs=...)`.
When fitting many conditional models in parallel, SeqTrees gives each LightGBM
model one internal thread to avoid oversubscribing CPU cores.

## Variable Ordering

You can provide an exact order:

```python
model = SequentialTreeSynthesizer(variable_order=["age", "sex_code", "risk_code"])
```

Or ask SeqTrees to choose a greedy order:

```python
model = SequentialTreeSynthesizer(optimize_order=True)
```

The optimizer starts with the lowest-entropy variable and repeatedly picks the
remaining variable with the lowest tree-estimated conditional entropy given the
variables already selected.

For a full explanation of sequential factorization and variable ordering, see
[Theoretical Background](theoretical_background.md).

## API

See [API](api.md) for constructor parameters and learned attributes.

## Comparison

For comparison with [Synthpop](https://cran.r-project.org/web/packages/synthpop/synthpop.pdf) ,  please check [Comparison](comparison.md).

## Diagrams

Both activity and class diagrams are provided in [Diagrams](diagrams.md) 
