# Changelog

## Unreleased

- Updated the runtime dependency to `ifcfill==0.3.6` after `0.3.5` was yanked.

## 0.2.0 - 2026-07-09

- Added Python 3.11 support metadata alongside Python 3.9 and 3.10.
- Added `ifcfill==0.3.4` as a runtime dependency.
- Added direct pandas DataFrame fitting with built-in `ifcfill` preprocessing and inverse transformation for sampled output.
- Used `IFCTransformer.column_types_` to distinguish original categorical columns from original integer columns after label encoding.
- Added `preprocessed_data_` and `get_preprocessed_data()` for inspecting the model-ready table used during fitting.
- Updated the medical notebook to fit SeqTrees directly on a mixed DataFrame and renamed it to `medical_dataframe_seqtrees.ipynb`.
- Updated documentation and tests for the DataFrame workflow.

## 0.1.0 - 2026-07-07

- Initial release of the sequential tree synthesizer.
