# Changelog

## 0.4.3 - 2026-10-29

- Relaxed the runtime dependency from `ifcfill==0.5.1` to
  `ifcfill>=0.5.2,<0.6`, so IFCFill patch releases install without a new
  SeqTrees release. IFCFill 0.5.2 rounds reconstructed dates to whole units.

## 0.4.2 - 2026-10-29

- Fixed the LightGBM backend, the default when LightGBM is installed,
  reproducing complete training records. It grouped records by their leaves
  across 32 boosted trees, which left most groups with a single record, so
  sampling replayed that record. The backend now grows one tree, so every leaf
  holds at least `min_samples_leaf` records.

## 0.4.1 - 2026-10-29

- Fixed `seqtrees.__version__` to read the installed package metadata instead
  of reporting the stale hard-coded version `0.2.0`.

## 0.4.0 - 2026-10-29

- Changed the default `continuous_strategy` from `"empirical"` to
  `"interpolate"`.
- Added documentation warnings that empirical sampling can expose complete
  training records.
- Updated the runtime dependency to `ifcfill==0.5.1` and modelled its generated
  missingness flags as discrete variables.
- Changed the package development status from Alpha to Beta.

## 0.3.0 - 2026-10-21

- Updated the runtime dependency to `ifcfill==0.4.0`.
- Limited supported Python versions to `>=3.9,<3.13` to match `ifcfill==0.4.0`, and added Python 3.12 metadata.
- Sampled output from `preprocessor="ifcfill"` now reproduces missing values at the training rates instead of imputing them. This follows the `ifcfill` 0.4.0 inverse transform, which always rebuilds missingness.

## 0.2.2 - 2026-07-25

- Restricted fitting input to pandas DataFrames.
- Disabled internal preprocessing by default and added the explicit
  `preprocessor="ifcfill"` option.
- Added missing-value validation with guidance for opting into IFCFill.

## 0.2.1 - 2026-07-20

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
