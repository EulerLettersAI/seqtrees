# Examples

The examples are provided as notebooks so users can run the workflow step by
step and inspect each intermediate table.

Install the optional example dependencies with:

```bash
python -m pip install "seqtrees[examples]"
```

- [Simple SeqTrees example](https://github.com/EulerLettersAI/seqtrees/blob/main/examples/simple_seqtrees.ipynb): the small
  documentation-style table with `age`, `sex_code`, `income_bin`, and
  `risk_code`.
- [Medical data with ifcfill](https://github.com/EulerLettersAI/seqtrees/blob/main/examples/medical_ifcfill_seqtrees.ipynb): loads a
  public Synthea healthcare CSV zip from a URL, preprocesses missing values,
  dates, and categorical values with `ifcfill`, fits SeqTrees, samples the same
  number of rows, and transforms the synthetic table back through `ifcfill`.
