# Comparison To Synthpop

SeqTree is inspired by the same broad sequential synthesis idea used by
Synthpop, but it makes a different design choice at the data boundary.

## Synthpop

Synthpop is an R package for creating synthetic microdata. Its documentation
describes variables as categorical or continuous and states that variables are
synthesised one by one using sequential modelling. Replacements are drawn from
conditional distributions fitted to the original data using parametric models
or classification and regression trees.

Synthpop is designed to work directly with R data frames and their data types.
It includes many synthesis methods, utility/disclosure tools, and data-type
aware behavior. For example, low-cardinality numeric variables can be changed
to factors depending on settings such as `minnumlevels`.

## SeqTree

SeqTree is deliberately narrower. It expects preprocessing to happen before
`fit`. The input table must contain no null values and every variable must be
one of:

- continuous: represented as `float`;
- discrete: represented as `int` category codes.

SeqTree does not encode raw categorical values, does not impute, does not scale,
and does not accept one-hot encoding as the intended workflow. Binary variables
are simply discrete integer-coded variables with values such as `0` and `1`.

Users should declare the modelling role of every variable:

```python
model = SequentialTreeSynthesizer(
    continuous_columns=["age", "bmi"],
    discrete_columns=["sex_code", "income_bin", "risk_code"],
)
```

If both lists are omitted, SeqTree can infer all-float columns as continuous and
all-integer columns as discrete, but explicit declaration is recommended for
production workflows.

## Practical Difference

| Topic | Synthpop | SeqTree |
| --- | --- | --- |
| Language | R | Python |
| Input data | R data frames with type-aware behavior | Preprocessed numeric tables |
| Categorical handling | Can use factor/categorical-aware methods | External label encoding required |
| Continuous handling | Parametric and tree-based numeric methods | Empirical or interpolated leaf sampling |
| Variable roles | Largely driven by data type and method settings | Explicit `continuous_columns` and `discrete_columns` |
| One-hot input | Not the core design point | Not intended; use one integer-coded categorical column |
| API style | R functions such as `syn()` | Estimator-style `fit()` / `sample()` |
| Backends | R modelling ecosystem | Native, scikit-learn, LightGBM |

SeqTree is therefore closer to a Python estimator for already-prepared tabular
data, while Synthpop is a broader R toolkit with more built-in data-type aware
synthetic data machinery.

## References

- Synthpop CRAN manual: <https://cran.r-project.org/web/packages/synthpop/synthpop.pdf>
- Nowok, Raab, and Dibben. "synthpop: Bespoke Creation of Synthetic Data in R."
  *Journal of Statistical Software*, 2016.
