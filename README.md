# seqtree

Sequential tree models for tabular synthetic data generation.

`seqtree` is a small open source Python library for generating synthetic tabular
data with a sequential tree model. Its API follows familiar estimator naming
conventions:

```python
from seqtree import SequentialTreeSynthesizer

data = [
    {"age": 24, "sex_code": 0, "income_bin": 1, "risk_code": 0},
    {"age": 31, "sex_code": 1, "income_bin": 1, "risk_code": 0},
    {"age": 67, "sex_code": 0, "income_bin": 3, "risk_code": 2},
    {"age": 73, "sex_code": 1, "income_bin": 3, "risk_code": 2},
]

model = SequentialTreeSynthesizer(
    variable_order=["age", "sex_code", "income_bin", "risk_code"],
    tree_backend="auto",
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

SeqTree expects model-ready, preprocessed data. It does not encode raw
categorical columns. Categorical variables should be converted before `fit`,
for example by your preprocessing library, and passed as stable numeric codes
or bins such as `sex_code`, `income_bin`, and `risk_code`.

## Features

- `fit`, `sample`, and `fit_sample` methods.
- User-specified variable ordering with `variable_order`.
- Greedy learned ordering with `optimize_order=True`.
- Parallel fitting and order-search candidates with `n_jobs`.
- Parallel row synthesis with `sample(..., n_jobs=...)`.
- Optional LightGBM and scikit-learn backends with `tree_backend="lightgbm"`,
  `tree_backend="sklearn"`, or `tree_backend="auto"`.
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
