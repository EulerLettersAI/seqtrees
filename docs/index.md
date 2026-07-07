# SeqTrees Documentation

SeqTrees builds synthetic tabular rows by learning an ordered product of
conditional distributions. Given preprocessed data with variables
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
- empirical leaf distributions, so sampling preserves values seen in the
  preprocessed training data.

The current implementation assumes preprocessing has already happened. SeqTrees
expects model-ready data with no null values. It accepts only continuous
variables represented as floats and discrete variables represented as integer
category codes. Binary variables are ordinary discrete variables with codes
`0` and `1`.

SeqTrees intentionally does not include encoders, imputers, scalers, or category
mapping utilities. If the source data contains categories such as `"female"`,
`"male"`, `"low"`, or `"high"`, convert them before calling `fit`. We recommed using [ifcfill](https://github.com/EulerLettersAI/ifcfill)
for transforming your raw data into SeqTrees-ready data.

## Continuous Variables

SeqTrees' default sampling strategy is empirical: every generated value is one
of the values observed in the training data. For continuous float variables,
you can opt into interpolation inside the reached leaf distribution:

```python
model = SequentialTreeSynthesizer(
    continuous_strategy="interpolate",
    continuous_columns=["age", "bmi"],
    discrete_columns=["sex_code", "risk_code"],
)
```

SeqTrees can infer variable types when both `continuous_columns` and
`discrete_columns` are omitted: all-float columns become continuous and all-int
columns become discrete. For production use, prefer declaring both lists:

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
  regressor, using leaf-index buckets for empirical sampling;
- `tree_backend="sklearn"`: scikit-learn's compiled `DecisionTreeRegressor`
  with empirical leaf sampling;
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
