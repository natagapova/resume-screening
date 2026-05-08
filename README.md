This document was last updated on **May 8, 2026**.

# Resume screening (BERT sequence classification)

This repository contains **Jupyter notebooks**, **exported figures**, and **tabular result artifacts** for research on **automated resume screening** with **BERT-style sequence classification** (9 job categories). The focus is on **accuracy–fairness trade-offs**, **debiasing methods** (GroupDRO, scrubbing, focal loss, label smoothing, adversarial and attribution-regularized training, etc.), **interpretability** (Integrated Gradients), and **robustness checks** such as **city-swap** counterfactual evaluation.

---

## Where everything lives

### Top level

| Path | What it is |
|------|------------|
| **`notebooks/`** | All experiment and analysis notebooks, plus reproducible outputs under `notebooks/results/`. |
| **`figures/`** | Paper-, slide-, and report-ready plots (PNG/PDF) and companion CSV/JSON summaries produced from the notebooks. |
| **`.gitignore`** | Defines paths that stay **local only** (datasets, checkpoints, virtualenvs, large logs). See [Not in this repository](#not-in-this-repository). |

There is **no** separate `src/` package: logic lives in the notebooks (typical for a thesis / research artifact repo).

---

### `notebooks/` — notebook map

Numbered files follow an informal pipeline order. Prefix **`c`** = “challenger” hyperparameter or ablation studies; **`p`** = audit / plumbing notebooks.

**Exploration and English transfer**

- **`00_eda.ipynb`** — Exploratory data analysis on the main (structured) dataset.
- **`01_english_dataset_eda.ipynb`**, **`02_english_dataset_transfer_pilot.ipynb`** — English résumé data: EDA and a small transfer pilot.
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

Tuning and comparison notebooks for alternative losses and training tricks, e.g. GroupDRO tuning, focal loss, label smoothing (several eps schedules), class-balanced losses, logit adjustment, R-Drop, re-seed comparisons, and challenger city-swap evaluation (`c10_*.ipynb`). Start from **`c00_compare_challengers_and_all_models.ipynb`** for an overview workflow.

**Audit / maintenance**

- **`p01_adversarial_attr_reg_recovery_audit.ipynb`**, **`p02_restore_split_checkpoints.ipynb`** — Recovery and checkpoint/split audits.

---

### `notebooks/results/` — saved metrics and tables

Exported **CSV**, **JSON**, **TXT**, and some **PNG** files from runs so comparisons do not depend on re-executing every notebook. Subfolders mirror experiment themes, for example:

- **`model_diagnostics/`** — Per-model diagnostic CSVs (baseline, GroupDRO, scrubbing, oversampling, focal, adversarial, label smoothing, attribution regularization, combined best, etc.) plus **`model_diagnostics_summary.csv`**.
- **`english_dataset_alignment/`**, **`english_dataset_transfer_pilot/`** — Outputs from notebooks `32` and `02`.
- **`challengers_city_swap/`**, **`city_swap_all/`**, **`cross_run_city_swap_comparison/`** — City-swap summaries and charts.
- **`two-models-restore/`** — Restore audit (`model_recovery_audit/`), restored-model city-swap summaries, and **`restored_models/`** verification tables (large binary checkpoints themselves are gitignored).
- **`unified_comparison/`** — Unified comparison table, report, and **`c71_accuracy_vs_gap.png`**.

Paths under `notebooks/results/` are **versioned** when they are small text/tabular artifacts; regenerate them by running the corresponding notebooks if you change code or data.

---

### `figures/` — graphics for the thesis or paper

High-level outputs used in documents and talks:

- **Root** — Accuracy/fairness comparison plots, trade-off scatters, Integrated Gradients figure sets, city TPR heatmaps, etc.
- **`figures/challengers/`** — Summaries and scatter plots for challenger-method sweeps (CSVs + key PNGs/PDFs).
- **`figures/slides_figures/`** — Slide-sized explanatory figures (distributions, TPR gaps, methods overview, debiasing outcomes).
- **`figures/english_dataset_alignment/`**, **`figures/english_dataset_transfer_pilot/`** — Plots tied to English dataset work.
- **`figures/cross_run_city_swap_comparison/`** — Cross-run city-swap visualization.

If you need the **exact** notebook that produced a file, open the notebook with the matching experiment id (e.g. `50_*` for many paper figures, `51_*` for slides).

---

## Not in this repository

The following are **intentionally excluded** (see `.gitignore`) and will not appear in a clean clone unless you create them locally:

- **Raw and processed datasets** (`data/`)
- **Trained weights** (`models/`, `*.pt`, `*.pkl`, `*.joblib`, and named checkpoint dirs such as `bert_9classes/`, …)
- **Experiment dumps** (`runs/`, local `archive/`, draft folders)
- **Virtual environments** (`venv/`, `.venv/`)
- **Large restored checkpoint trees** under some `notebooks/results/two-models-restore/restored_models/...` paths

You must supply **your own data** and **train or copy checkpoints** locally to reproduce training from scratch. Many **metrics and figures** are still available from `notebooks/results/` and `figures/` without rerunning GPU training.

---

## Getting started

1. **Clone the repository** (replace the URL with yours):

   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. **Create a virtual environment** and install dependencies. This repo does **not** ship a pinned `requirements.txt`; install a standard PyTorch + NLP stack used by the notebooks, for example:

   - Python 3.9+ (adjust for your PyTorch build)
   - `torch`, `transformers`, `datasets`, `accelerate`
   - `jupyter`, `ipykernel`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

   Add any extra packages as imports fail when you open a notebook.

3. **Add data** under `data/` (ignored by git) following the paths expected in the notebooks (e.g. processed CSV splits if you preprocess locally).

4. **Run notebooks** in order of interest; for a high-level path: EDA → `24_finetuned_bert` → debiasing notebooks → `40`/`42` fairness → `41` IG → `50`/`51` figures.

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
