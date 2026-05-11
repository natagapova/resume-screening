This document was last updated on **May 11, 2026**.

# Resume screening (BERT sequence classification)

This repository contains **Jupyter notebooks**, **exported figures**, and **tabular result artifacts** for research on **automated resume screening** with **BERT-style sequence classification** (9 job categories). The focus is on **accuracy–fairness trade-offs**, **debiasing methods** (GroupDRO, scrubbing, focal loss, label smoothing, adversarial and attribution-regularized training, etc.), **interpretability** (Integrated Gradients), and **robustness checks** such as **city-swap** counterfactual evaluation (Russian in-domain and English transfer).

---

## Where everything lives

### Top level

| Path | What it is |
|------|------------|
| **`notebooks/`** | Experiment and analysis notebooks, shared helpers, and reproducible outputs under `notebooks/results/`. |
| **`figures/`** | Paper-, slide-, and report-ready plots (PNG/PDF) plus companion CSV/JSON summaries produced from notebooks. |
| **`README.md`** | This map + what GitHub hides via `.gitignore`. |
| **`.gitignore`** | Paths that stay **local only**. See [Not on GitHub (gitignored)](#not-on-github-gitignored). |

There is **no** separate `src/` package: most logic lives in the notebooks (research / thesis artifact layout).

---

### `notebooks/` — notebook map

Numbered files follow an informal pipeline order. Prefix **`c`** = “challenger” hyperparameter or ablation studies; **`p`** = audit / plumbing notebooks.

**English evaluation (frozen checkpoints, transfer, city-swap)**

- **`notebooks/english/e00_english_eval_overview.ipynb`** — Overview of the English evaluation stack.
- **`e01_eval_bert_9classes_final.ipynb`**, **`e02_eval_bert_scrubbing.ipynb`**, **`e03_eval_label_smoothing_eps01_2ep.ipynb`** — Eval-only runs for specific Russian-trained checkpoints on English slices.
- **`e04_english_benchmark_aggregate.ipynb`** — Aggregated English benchmark tables.
- **`e05_english_city_swap_existing_models.ipynb`** — City-swap style stress tests on English (regenerates large pair exports; see gitignore note below).
- **`e07_english_eval_common.py`** — Shared helpers imported by the English notebooks.

**Exploration and English transfer (root-level notebooks)**

- **`01_english_dataset_eda.ipynb`**, **`02_english_dataset_transfer_pilot.ipynb`** — English résumé data: EDA and transfer pilot.
- **`32_english_dataset_alignment.ipynb`** — Aligning English labels/categories with the main task setup.

**Baselines and data prep**

- **`10_baseline_tfidf.ipynb`** — Non-neural baseline (TF–IDF).
- **`30_data_cleaning.ipynb`**, **`31_supercategories_clustering_clean.ipynb`** — Cleaning and supercategory clustering.

**BERT training and debiasing (main numbered runs)**

- **`20_combined_scrubbing_groupdro.ipynb`** — Combined scrubbing + GroupDRO-style setup.
- **`24_finetuned_bert.ipynb`** — Core BERT fine-tuning workflow.
- **`25_debiasing_combo.ipynb`**, **`26_oversampling_only.ipynb`**, **`27_groupdro_eta01_2ep.ipynb`**, **`29_focal_loss.ipynb`** — Debiasing and class-imbalance variants.
- **`60_label_smoothing.ipynb`**, **`61_adversarial_debiasing.ipynb`**, **`62_scrubbing.ipynb`**, **`63_attribution_regularization_v2.ipynb`** — Additional training objectives and regularizers.

**Fairness, errors, interpretability**

- **`40_fairness_and_error_analysis.ipynb`**, **`42_fairness_comparison.ipynb`** — Fairness metrics and comparisons across runs.
- **`41_integrated_gradients.ipynb`** — Integrated Gradients attributions for qualitative analysis.

**Figures for writing and slides**

- **`50_paper_figures_v3.ipynb`** — Consolidated paper figures.
- **`51_slides_figure_generation.ipynb`** — Slide-oriented graphics.

**City-swap and multi-model comparison**

- **`70_city_swap_batch_eval.ipynb`**, **`71_unified_models_comparison.ipynb`** — Batch evaluation and a unified comparison table across model families.
- **`72_restored_models_city_swap.ipynb`**, **`73_cross_run_city_swap_comparison.ipynb`** — Restored checkpoints and cross-run city-swap analysis.
- **`city_swap_model_diagnostics.ipynb`** — Diagnostics around city-swap experiments.

**`notebooks/challengers/`**

Tuning and comparison notebooks for alternative losses and training tricks, e.g. GroupDRO tuning, focal loss, label smoothing (several ε schedules), class-balanced losses, logit adjustment, R-Drop, re-seed comparisons, and challenger city-swap evaluation (`c10_*.ipynb`). Start from **`c00_compare_challengers_and_all_models.ipynb`** for an overview workflow.

**Audit / maintenance**

- **`p01_adversarial_attr_reg_recovery_audit.ipynb`**, **`p02_restore_split_checkpoints.ipynb`** — Recovery and checkpoint/split audits.

---

### `notebooks/results/` — saved metrics and tables

Exported **CSV**, **JSON**, **TXT**, **TeX**, and some **PNG** files from runs so comparisons do not depend on re-executing every notebook. Subfolders mirror experiment themes, for example:

- **`table_export/`** — Machine-generated **`T*.tex`** / **`T*.csv`** fragments (unified ledgers, city-swap, English transfer, dataset splits, etc.) meant for papers and appendices.
- **`model_diagnostics/`** — Per-model diagnostic CSVs (baseline, GroupDRO, scrubbing, oversampling, focal, adversarial, label smoothing, attribution regularization, combined best, etc.) plus **`model_diagnostics_summary.csv`**.
- **`english_dataset_alignment/`**, **`english_dataset_transfer_pilot/`**, **`english_eval/`**, **`english_transfer_eval/`**, **`english_city_swap_eval/`**, **`english_balanced_eval/`** — Outputs from the English alignment / eval / city-swap workflows.
- **`challengers_city_swap/`**, **`city_swap_all/`**, **`cross_run_city_swap_comparison/`** — City-swap summaries and charts.
- **`two-models-restore/`** — Restore audit (`model_recovery_audit/`), restored-model city-swap summaries, and **`restored_models/`** verification tables (large binary checkpoints under specific subtrees are gitignored).
- **`unified_comparison/`** — Unified comparison table, report, and companion plots (e.g. **`c71_accuracy_vs_gap.png`**).

Paths under `notebooks/results/` are **versioned** when they are small text/tabular artifacts; regenerate them by running the corresponding notebooks if you change code or data.

---

### `figures/` — graphics for the thesis or paper

High-level outputs used in documents and talks:

- **Root** — Accuracy/fairness comparison plots, trade-off scatters, Integrated Gradients figure sets, city TPR heatmaps, etc.
- **`figures/challengers/`** — Summaries and scatter plots for challenger-method sweeps (CSVs + key PNGs/PDFs).
- **`figures/slides_figures/`** — Slide-sized explanatory figures (distributions, TPR gaps, methods overview, debiasing outcomes).
- **`figures/paper_figures/`** — Export-quality PDF/PNG pairs for write-ups (unified comparison panels, fairness vs. city-swap scatters, etc.).
- **`figures/integrated_gradients_extended/`** — Extended IG comparison grids and heatmaps.
- **`figures/english_dataset_alignment/`**, **`figures/english_dataset_transfer_pilot/`**, **`figures/english_eval/`**, **`figures/english_transfer_eval/`**, **`figures/english_city_swap_eval/`** — Plots tied to English dataset and eval work.
- **`figures/cross_run_city_swap_comparison/`**, **`figures/cross_run_city_swap/`** — Cross-run city-swap visualization.
- **`figures/dataset_audit/`**, **`figures/adversarial_eval_design/`** — Dataset and adversarial-eval schematics.

If you need the **exact** notebook that produced a file, open the notebook with the matching experiment id (e.g. `50_*` for many paper figures, `51_*` for slides, `e0*` for English eval).

---

## Not on GitHub (gitignored)

These paths are **intentionally excluded** (see `.gitignore`). A **clean GitHub clone** will not contain them unless you create or copy them locally:

| Category | Paths / patterns |
|----------|------------------|
| **LaTeX / ICML drafts** | **`icml2026/`** (entire directory: style file, main `.tex`, appendix snippets, local PDFs, etc.) — **not visible on GitHub** with the current ignore rule. |
| **Primary EDA notebook** | **`notebooks/00_eda.ipynb`** — listed in `.gitignore` (keep a private copy locally if you use it). |
| **Data & weights** | **`data/`**, **`models/`**, **`*.pt`**, **`*.pkl`**, **`*.joblib`**, checkpoint dirs such as **`bert_9classes/`**, **`bert_9classes_sqrt_rw_2ep/`**, **`bert_9classes_gdro/`** |
| **Environment & IDE** | **`venv/`**, **`.venv/`**, **`.vscode/`**, **`.DS_Store`** |
| **Experiment noise** | **`runs/`**, **`archive/`**, **`drafts/`**, **`local_bert/`**, **`output/combined_scrubbing_groupdro/`**, **`notebooks/notebooks/`** (duplicate tree guard) |
| **Large / binary artifacts** | Partial trees under **`notebooks/results/two-models-restore/restored_models/adversarial/`** and **`.../attr_reg/`**; **`**/predictions_english_city_swap_pairs.csv`** (very large; regenerate via **`e05_english_city_swap_existing_models.ipynb`**) |
| **Caches** | **`__pycache__/`**, **`*.pyc`**, **`*.ipynb_checkpoints`** |
| **Local tool logs** | **`.brew-texlive-install.log`** |

You must supply **your own data** and **train or copy checkpoints** locally to reproduce training from scratch. Many **metrics, TeX table exports, and figures** are still available from `notebooks/results/` and `figures/` without rerunning GPU training.

To **publish LaTeX sources** or share `icml2026/` on GitHub, narrow or remove the corresponding line in `.gitignore` and commit only what you intend to license.

---

## Getting started

1. **Clone the repository**

   ```bash
   git clone https://github.com/natagapova/resume-screening.git
   cd resume-screening
   ```

2. **Create a virtual environment** and install dependencies. This repo does **not** ship a pinned `requirements.txt`; install a standard PyTorch + NLP stack used by the notebooks, for example:

   - Python 3.9+ (adjust for your PyTorch build)
   - `torch`, `transformers`, `datasets`, `accelerate`
   - `jupyter`, `ipykernel`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

   Add any extra packages as imports fail when you open a notebook.

3. **Add data** under `data/` (ignored by git) following the paths expected in the notebooks (e.g. processed CSV splits if you preprocess locally).

4. **Run notebooks** in order of interest; for a high-level path: English quick path → `notebooks/english/e00_*` → root `32_*` / `02_*` as needed; Russian path → `24_finetuned_bert` → debiasing notebooks → `40`/`42` fairness → `41` IG → `50`/`51` figures → `70`–`73` city-swap.

---

## Model & configuration (summary)

- **Task:** BERT **sequence classification**, **9 classes** (job categories).
- **Checkpoints:** Not committed; train locally or point notebooks to your saved weights.

---

## Reference metrics (private data)

Illustrative numbers from a **BERT + sqrt_rw** baseline on **non-public** data (for context only; your numbers will differ):

| Metric | Value |
|--------|------:|
| Accuracy | 0.609 |
| Macro F1 | 0.621 |
| TPR gap (worst robust) | 0.329 |
| TPR gap (macro robust) | 0.116 |

---

## Notes

- This project is intended for **research and thesis reproducibility**, not production deployment.
- If something is missing after clone, check **`.gitignore`** first—it is usually deliberate.
- **GitHub’s file view** will not list ignored paths; use this README + `.gitignore` as the source of truth for “what exists locally vs. on the remote.”
