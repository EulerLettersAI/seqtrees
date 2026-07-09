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
- [Medical DataFrame synthesis](https://github.com/EulerLettersAI/seqtrees/blob/main/examples/medical_dataframe_seqtrees.ipynb): loads a
  public Synthea healthcare CSV zip from a URL, fits SeqTrees directly on a
  mixed patient DataFrame, samples the same number of rows, and restores
  patient-like values through SeqTrees' internal `ifcfill` transformer.
