# seqtrees

Sequential tree models for tabular synthetic data generation.

`seqtrees` is a small open source Python library for generating synthetic tabular
data with a sequential tree model. Its API follows familiar estimator naming
conventions:

```python
from seqtrees import SequentialTreeSynthesizer

data = [
    {"age": 24.5, "sex_code": 0, "income_bin": 1, "risk_code": 0},
    {"age": 31.2, "sex_code": 1, "income_bin": 1, "risk_code": 0},
    {"age": 67.4, "sex_code": 0, "income_bin": 3, "risk_code": 2},
    {"age": 73.0, "sex_code": 1, "income_bin": 3, "risk_code": 2},
]

model = SequentialTreeSynthesizer(
    variable_order=["age", "sex_code", "income_bin", "risk_code"],
    tree_backend="auto",
    continuous_strategy="empirical",
    continuous_columns=["age"],
    discrete_columns=["sex_code", "income_bin", "risk_code"],
    n_jobs=-1,
    random_state=7,
)
model.fit(data)

synthetic = model.sample(10)
```

The model factorizes a table into a sequence of conditional distributions:

```text
P(X1, X2, ..., Xd) = P(X1) P(X2 | X1) ... P(Xd | X1, ..., Xd-1)
```

The first variable is sampled from its empirical marginal distribution. Each
later variable is sampled from a conditional decision tree trained on the
previous variables in the sequence. This follows the sequential tree synthesis
approach described by El Emam, Mosquera, and Zheng (2020).

SeqTrees accepts raw pandas DataFrames and uses `ifcfill` with label encoding to
prepare them for the tree model. The fitted transformer is used again when
sampling, so generated DataFrame output is restored to the original labels and
column types.

After fitting, inspect the exact model-ready table with
`model.get_preprocessed_data()` or the fitted `model.preprocessed_data_`
attribute.

List-based inputs still need to be model-ready, preprocessed data with no null
values. They accept only:

- continuous variables as floats;
- discrete variables as integer category codes, including binary 0/1 variables.

For DataFrame input, original categorical columns are label-encoded and sampled
only from labels observed during fitting. Original integer columns are treated
as numeric integer variables, not category labels; with interpolation enabled,
they may generate integer values that were not present in the input.

By default, all generated values are sampled from observed training values.
For continuous float columns, you can opt into within-leaf interpolation:

```python
model = SequentialTreeSynthesizer(
    continuous_strategy="interpolate",
    continuous_columns=["age", "bmi"],
    discrete_columns=["sex_code", "income_bin", "risk_code"],
)
```

When type lists are supplied, `continuous_columns` and `discrete_columns` must
classify every input column exactly once.

## Features

- `fit`, `sample`, and `fit_sample` methods.
- Access to the model-ready fitted table with `get_preprocessed_data()`.
- User-specified variable ordering with `variable_order`.
- Greedy learned ordering with `optimize_order=True`.
- Parallel fitting and order-search candidates with `n_jobs`.
- Parallel row synthesis with `sample(..., n_jobs=...)`.
- Optional LightGBM and scikit-learn backends with `tree_backend="lightgbm"`,
  `tree_backend="sklearn"`, or `tree_backend="auto"`.
- Optional within-leaf interpolation for continuous float and DataFrame integer
  variables.
- Supports any number of variables.
- Accepts lists of dictionaries, lists of row sequences, and pandas DataFrames
  when pandas is installed.
- Uses ifcfill for DataFrame encoding, imputation, and restoration.

## Installation

```bash
python -m pip install seqtrees
```

Install optional backends only when you need them:

```bash
python -m pip install "seqtrees[sklearn]"
python -m pip install "seqtrees[lightgbm]"
python -m pip install "seqtrees[fast]"
```

## Install For Development

Use an editable install when working on the source code locally:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

For faster compiled tree fitting from a local checkout:

```bash
python -m pip install -e ".[fast]"
```

For LightGBM only from a local checkout:

```bash
python -m pip install -e ".[lightgbm]"
```

To run the notebooks in `examples/`:

```bash
python -m pip install -e ".[examples]"
jupyter lab examples
```

Use all available cores:

```python
model = SequentialTreeSynthesizer(tree_backend="lightgbm", n_jobs=-1)
```

## Documentation

Read the documentation at https://eulerlettersai.github.io/seqtrees/ for
examples, API details, diagrams, and theoretical background.

Build the documentation locally with MkDocs:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[docs]"
mkdocs serve
```

## References

- El Emam, Mosquera, and Zheng. "Optimizing the synthesis of clinical trial data
  using sequential trees." *Journal of the American Medical Informatics
  Association*, 2020. https://academic.oup.com/jamia/article/28/1/3/5981525
- Breiman, Friedman, Olshen, and Stone. *Classification and Regression Trees*.
  Wadsworth, 1984.
- Quinlan. "Induction of Decision Trees." *Machine Learning*, 1986.
