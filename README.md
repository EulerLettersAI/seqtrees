# seqtree

Sequential tree models for tabular synthetic data generation.

`seqtree` is a small open source Python library for generating synthetic tabular
data with a sequential tree model. Its API follows familiar estimator naming
conventions:

```python
from seqtree import SequentialTreeSynthesizer

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
previous variables in the sequence. This mirrors the sequential data synthesis
workflow shown in Figure 1 of the attached paper (`ocaa249.pdf`).

SeqTree expects model-ready, preprocessed data with no null values. It accepts
only:

- continuous variables as floats;
- discrete variables as integer category codes, including binary 0/1 variables.

It does not encode raw categorical columns, and it is not intended for one-hot
encoded inputs. Categorical variables should be mapped before `fit`, for
example by your preprocessing library, and passed as stable integer codes or
bins such as `sex_code`, `income_bin`, and `risk_code`.

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
- User-specified variable ordering with `variable_order`.
- Greedy learned ordering with `optimize_order=True`.
- Parallel fitting and order-search candidates with `n_jobs`.
- Parallel row synthesis with `sample(..., n_jobs=...)`.
- Optional LightGBM and scikit-learn backends with `tree_backend="lightgbm"`,
  `tree_backend="sklearn"`, or `tree_backend="auto"`.
- Optional within-leaf interpolation for continuous float variables.
- Supports any number of preprocessed variables.
- Accepts lists of dictionaries, lists of row sequences, and pandas DataFrames
  when pandas is installed.
- Does not provide encoders, imputers, scalers, or category mapping utilities.
- Pure-Python core with no mandatory runtime dependencies.
- PlantUML diagrams in `docs/diagrams`.

## Install For Development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

For faster compiled tree fitting:

```bash
python -m pip install -e ".[fast]"
```

For LightGBM only:

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

Start with [docs/index.md](docs/index.md). The package includes PlantUML source
files for the training and sampling flow:

- [docs/diagrams/sequential_synthesis.puml](docs/diagrams/sequential_synthesis.puml)
- [docs/diagrams/class_overview.puml](docs/diagrams/class_overview.puml)

Build the documentation locally with MkDocs:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[docs]"
mkdocs serve
```

Read the Docs builds are configured by `.readthedocs.yaml` and install
`docs/requirements.txt`.

## References

- The implementation is based on the sequential synthetic data workflow
  described in the attached paper (`ocaa249.pdf`), especially the Figure 1
  sequential data synthesis process.
- Breiman, Friedman, Olshen, and Stone. *Classification and Regression Trees*.
  Wadsworth, 1984.
- Quinlan. "Induction of Decision Trees." *Machine Learning*, 1986.
