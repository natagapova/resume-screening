# Reproducibility guide

This document supplements the [README](README.md) with notebook maps, gitignore rules, and regeneration instructions for artifacts that are not stored in git.

## What is versioned vs. local-only

| Category | In git | Local only |
|----------|--------|------------|
| Notebooks | Yes (except `00_eda.ipynb`) | — |
| Summary metrics & TeX tables | Yes (`notebooks/results/unified_comparison/`, `table_export/`, etc.) | — |
| Figures | Yes (`figures/`) | — |
| Thesis PDF | `thesis/thesis.pdf` only | LaTeX sources under `thesis/` |
| Raw / processed data | — | `data/` |
| Model checkpoints | — | `notebooks/models/`, `models/`, `*.pt`, `*.safetensors` |
| Row-level predictions | — | `**/predictions_*.csv`, city-swap counterfactual CSVs |

A clean clone contains notebooks, summary tables, and figures. You must supply data and train (or copy) checkpoints to reproduce training from scratch.

## Regenerating excluded artifacts

| Artifact | Notebook |
|----------|----------|
| English city-swap pair predictions | `notebooks/english/e05_english_city_swap_existing_models.ipynb` |
| English eval predictions | `notebooks/english/e01_*`, `e02_*`, `e03_*` |
| Integrated Gradients wide predictions | `notebooks/41_1_integrated_gradients_extended.ipynb` |
| City-swap counterfactual CSVs | `notebooks/32_english_dataset_alignment.ipynb`, `02_english_dataset_transfer_pilot.ipynb` |

## Notebook map

Numbered files follow an informal pipeline order. Prefix **`c`** = challenger hyperparameter studies; **`p`** = audit / plumbing notebooks.

### English evaluation (`notebooks/english/`)

- `e00_english_eval_overview.ipynb` — overview of the English eval stack
- `e01_eval_bert_9classes_final.ipynb`, `e02_eval_bert_scrubbing.ipynb`, `e03_eval_label_smoothing_eps01_2ep.ipynb` — eval-only runs for Russian-trained checkpoints
- `e04_english_benchmark_aggregate.ipynb` — aggregated English benchmark tables
- `e05_english_city_swap_existing_models.ipynb` — city-swap stress tests on English
- `e07_english_eval_common.py` — shared helpers

### Exploration and English transfer (root-level)

- `01_english_dataset_eda.ipynb`, `02_english_dataset_transfer_pilot.ipynb` — English résumé EDA and transfer pilot
- `32_english_dataset_alignment.ipynb` — align English labels with the main task setup

### Baselines and data prep

- `10_baseline_tfidf.ipynb` — TF–IDF baseline
- `30_data_cleaning.ipynb`, `31_supercategories_clustering_clean.ipynb` — cleaning and supercategory clustering

### BERT training and debiasing

- `20_combined_scrubbing_groupdro.ipynb` — combined scrubbing + GroupDRO
- `24_finetuned_bert.ipynb` — core BERT fine-tuning
- `25_debiasing_combo.ipynb`, `26_oversampling_only.ipynb`, `27_groupdro_eta01_2ep.ipynb`, `29_focal_loss.ipynb`
- `60_label_smoothing.ipynb`, `61_adversarial_debiasing.ipynb`, `62_scrubbing.ipynb`, `63_attribution_regularization_v2.ipynb`

### Fairness, errors, interpretability

- `40_fairness_and_error_analysis.ipynb`, `42_fairness_comparison.ipynb`
- `41_integrated_gradients.ipynb`, `41_1_integrated_gradients_extended.ipynb`

### Figures

- `50_paper_figures_v3.ipynb` — consolidated paper figures
- `51_slides_figure_generation.ipynb` — slide graphics
- `52_table_export.ipynb` — TeX/CSV table exports

### City-swap and unified comparison

- `70_city_swap_batch_eval.ipynb`, `71_unified_models_comparison.ipynb`
- `72_restored_models_city_swap.ipynb`, `73_cross_run_city_swap_comparison.ipynb`
- `city_swap_model_diagnostics.ipynb`

### Challengers (`notebooks/challengers/`)

Tuning notebooks for alternative losses and training tricks (GroupDRO, focal loss, label smoothing, class-balanced losses, R-Drop, etc.). Start from `c00_compare_challengers_and_all_models.ipynb`.

### Audit / maintenance

- `p01_adversarial_attr_reg_recovery_audit.ipynb`, `p02_restore_split_checkpoints.ipynb`

## Results subfolders (`notebooks/results/`)

Exported CSV, JSON, TXT, TeX, and some PNG files from runs:

- `table_export/` — `T*.tex` / `T*.csv` fragments for papers and appendices
- `model_diagnostics/` — per-model diagnostic CSVs
- `unified_comparison/` — unified comparison table and companion plots
- `english_*` — English alignment, eval, and city-swap outputs
- `challengers_city_swap/`, `city_swap_all/`, `cross_run_city_swap_comparison/`
- `two-models-restore/` — restore audit and verification tables

## Figures (`figures/`)

- `paper_figures/` — export-quality PDF/PNG for write-ups
- `unified_comparison/` — accuracy–fairness trade-off plots
- `challengers/` — challenger-method sweep summaries
- `slides_figures/` — slide-sized explanatory figures
- `integrated_gradients_extended/` — extended IG comparison grids

Match experiment ids when tracing provenance (e.g. `50_*` for paper figures, `e0*` for English eval).

## Thesis build

LaTeX sources under `thesis/` are gitignored; only `thesis/thesis.pdf` is published.

```bash
cd thesis
latexmk -xelatex thesis.tex
latexmk -xelatex annotation.tex
```

Requires XeLaTeX + biber. See local `thesis/README.md` if present.

## Suggested notebook order

**Russian main path:** `24_finetuned_bert` → debiasing notebooks → `40`/`42` fairness → `41` IG → `50`/`51` figures → `70`–`73` city-swap.

**English path:** `notebooks/english/e00_*` → `32_*` / `02_*` as needed.

## Gitignored paths (reference)

See [`.gitignore`](.gitignore) for the authoritative list. Highlights:

- `data/`, `notebooks/models/`, `models/`, `*.pt`, `*.safetensors`
- `**/predictions_*.csv`, `**/*city_swap_counterfactuals.csv`
- `notebooks/00_eda.ipynb`
- `icml2026/`, `archive/`, `scripts/`, `venv/`, `.venv/`
