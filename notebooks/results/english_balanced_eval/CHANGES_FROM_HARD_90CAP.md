# Changes vs hard 90-per-class

- **Removed** the fixed **90 rows per class** rule (and the implicit tie to the smallest class as a universal cap).
- **New rule:** after the same text dedupe, compute the **median** of the eight per-class deduped counts. Any class **at or below** that median keeps **all** deduped rows (minorities unchanged). Any class **above** the median is reduced to **median + 0.38 × (deduped − median)** (rounded down to int) — large classes stay larger than small ones, but extremes are softened.
- **Continuity:** If the on-disk `english_balanced_8class_eval.csv` was still the old **720-row / 90-per-class** file, it is copied to **`english_eval_hard_90cap_archive.csv`** before overwrite. Building the soft slice **keeps prior rows first** when they still exist in the deduped pool, then **adds** stratified rows to reach the soft target.
- **9 vs 8 classes:** Project ontology is 9 supercategories; English mapped data currently has **8** (`sales_account` missing). **No synthetic sales labels** are added.

Tune **SOFT_PULL** in the first code cell (higher = closer to full counts; lower = more trimming of heavy classes).
