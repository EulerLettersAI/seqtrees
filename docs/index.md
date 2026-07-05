# SeqTree Documentation

SeqTree builds synthetic tabular rows by learning an ordered product of
conditional distributions. Given preprocessed data with variables
`X1, X2, ..., Xd`, it estimates:

```text
P(X1, X2, ..., Xd) = P(X1) P(X2 | X1) ... P(Xd | X1, ..., Xd-1)
```

This implementation follows the sequential synthesis idea shown in Figure 1 of
the attached paper (`ocaa249.pdf`): generate the first variable, append it to
the synthetic row, then use the generated prefix to synthesize the next
variable until the row is complete.

## Model

`SequentialTreeSynthesizer` uses:

- an empirical marginal distribution for the first variable;
- one conditional decision tree for each later variable;
- empirical leaf distributions, so sampling preserves values seen in the
  preprocessed training data.

The current implementation assumes preprocessing has already happened. SeqTree
expects model-ready data with no null values. It accepts only continuous
variables represented as floats and discrete variables represented as integer
category codes. Binary variables are ordinary discrete variables with codes
`0` and `1`.

SeqTree intentionally does not include encoders, imputers, scalers, or category
mapping utilities. If the source data contains categories such as `"female"`,
`"male"`, `"low"`, or `"high"`, convert them before calling `fit`. For SeqTree,
that usually means label encoding categorical variables to integer codes and
passing naturally continuous variables as floats.

## Continuous Variables

SeqTree's default sampling strategy is empirical: every generated value is one
of the values observed in the training data. For continuous float variables,
you can opt into interpolation inside the reached leaf distribution:

```python
model = SequentialTreeSynthesizer(
    continuous_strategy="interpolate",
    continuous_columns=["age", "bmi"],
    discrete_columns=["sex_code", "risk_code"],
)
```

SeqTree can infer variable types when both `continuous_columns` and
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
SeqTree.

## Performance

SeqTree can use either:

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
python -m pip install "seqtree[fast]"
```

Use LightGBM explicitly with:

```python
model = SequentialTreeSynthesizer(tree_backend="lightgbm", n_jobs=-1)
```

Set `n_jobs` to control parallelism. `n_jobs=1` runs serially, a positive
integer sets the worker count, and `n_jobs=-1` uses all detected CPU cores.
SeqTree parallelizes conditional tree fitting across variables once the order is
known, parallelizes candidate scoring inside greedy order optimization, and can
parallelize synthetic row generation through `sample(..., n_jobs=...)`.
When fitting many conditional models in parallel, SeqTree gives each LightGBM
model one internal thread to avoid oversubscribing CPU cores.

## Variable Ordering

You can provide an exact order:

```python
model = SequentialTreeSynthesizer(variable_order=["age", "sex_code", "risk_code"])
```

Or ask SeqTree to choose a greedy order:

```python
model = SequentialTreeSynthesizer(optimize_order=True)
```

The optimizer starts with the lowest-entropy variable and repeatedly picks the
remaining variable with the lowest tree-estimated conditional entropy given the
variables already selected.

## API

See [api.md](api.md) for constructor parameters and learned attributes.

## Diagrams

PlantUML sources:

- [diagrams/sequential_synthesis.puml](diagrams/sequential_synthesis.puml)
- [diagrams/class_overview.puml](diagrams/class_overview.puml)

## References

- Attached paper: `ocaa249.pdf`, Figure 1 sequential data synthesis workflow.
- Breiman, Friedman, Olshen, and Stone. *Classification and Regression Trees*.
  Wadsworth, 1984.
- Quinlan. "Induction of Decision Trees." *Machine Learning*, 1986.
- Ke et al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree."
  NeurIPS, 2017.
