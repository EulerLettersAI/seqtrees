# Comparison To Synthpop

SeqTrees is inspired by the same broad sequential synthesis idea used by
Synthpop, but it makes a different design choice at the data boundary.

## Synthpop

Synthpop[@nowok2016synthpop] is an R package for creating synthetic microdata. Its documentation
describes variables as categorical or continuous and states that variables are
synthesised one by one using sequential modelling. Replacements are drawn from
conditional distributions fitted to the original data using parametric models
or classification and regression trees.

Synthpop is designed to work directly with R data frames and their data types.
It includes many synthesis methods, utility/disclosure tools, and data-type
aware behavior. For example, low-cardinality numeric variables can be changed
to factors depending on settings such as `minnumlevels`.

## SeqTrees

SeqTrees is deliberately narrower. For pandas DataFrame input, it uses ifcfill
to impute and label-encode values before fitting. List-based input must be
preprocessed before `fit`, contain no null values, and every variable must be
one of:

- continuous: represented as `float`;
- discrete: represented as `int` category codes.

SeqTrees does not accept one-hot encoding as the intended workflow. Binary
variables are simply discrete integer-coded variables with values such as `0`
and `1`.

Users should declare the modelling role of every variable:

```python
model = SequentialTreeSynthesizer(
    continuous_columns=["age", "bmi"],
    discrete_columns=["sex_code", "income_bin", "risk_code"],
)
```

If both lists are omitted for list-based input, SeqTrees can infer all-float
columns as continuous and all-integer columns as discrete. For DataFrames,
SeqTrees uses `IFCTransformer.column_types_` so original categorical columns do
not generate unseen labels while original integer columns remain numeric.

## Practical Difference

| Topic                | Synthpop                                        | SeqTrees                                                 |
| -------------------- | ----------------------------------------------- | ------------------------------------------------------- |
| Language             | R                                               | Python                                                  |
| Input data           | R data frames with type-aware behavior          | DataFrames via ifcfill or preprocessed numeric tables   |
| Categorical handling | Can use factor/categorical-aware methods        | ifcfill label encoding for DataFrame input              |
| Continuous handling  | Parametric and tree-based numeric methods       | Empirical or interpolated leaf sampling                 |
| Variable roles       | Largely driven by data type and method settings | Explicit`continuous_columns` and `discrete_columns` |
| One-hot input        | Not the core design point                       | Not intended; use one integer-coded categorical column  |
| API style            | R functions such as`syn()`                    | Estimator-style`fit()` / `sample()`                 |
| Backends             | R modelling ecosystem                           | Native, scikit-learn, LightGBM                          |

SeqTrees is therefore closer to a Python estimator with a focused DataFrame
preparation layer, while Synthpop is a broader R toolkit with more built-in
data-type aware synthetic data machinery.
