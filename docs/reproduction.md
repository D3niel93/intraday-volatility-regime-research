# Data dictionary and reproducibility boundary

- `predictions_*.csv`: disjoint chunks of all 7,783 stored predictions. year/local_date/bucket uniquely identify a next-bar evaluation; actual_expansive is its binary observed label; probability_expansive is the frozen-model output.
- `verified_numeric_range_accuracy.csv`: numeric forecast evaluation, distinct from binary classification.
- `audit_summary.csv`: chronology reconstruction evidence; mismatches were found, not hidden.
- Walk-forward fold, bootstrap, cost and strategy CSVs relate to the **separate five-minute reversal** research. They are not confidence intervals for the 15-minute classifier.

The public evaluator requires no market feed. It reproduces the stored prediction-level metrics, not data cleaning, model fitting or tick replay. The numerical forecasts use next-bar opening prices as their percentage-to-points scale; a completed-bar signal available beforehand must use a known-price anchor for tradable orders, not a future opening price.

Archived scripts preserve originals with private absolute paths redacted. Local imports, private event caches and original directory layout are needed for raw-data reconstruction. Source hashes in the manifest identify the pre-sanitization originals. Use completed current-bar features only, never the target bar's values.
