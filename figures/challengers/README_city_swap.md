# City-swap evaluation policy

**All** challenger runs get accuracy + robust TPR-gap metrics. **City-swap flip rate** is run only for a **selected subset** (best per family or explicitly strong combinations), not for every hyperparameter trial. Empty `overall_flip_rate` in `c00_all_models_overview.csv` means “not evaluated,” not “zero flips.”

| Notebook / file | Models |
|-----------------|--------|
| `c10_challenger_city_swap_eval.ipynb` | 5 best-per-family singles (after tuning) |
| `c15_tier1_combo_city_swap_eval.ipynb` | Combos C11–C14 (tier-1, already strong on F1/gap) |
| `c18_scrubbing_groupdro_city_swap_eval.ipynb` | Scrub + GroupDRO (η=0.05, 0.10) — flip 0.00\% / 0.15\% |
| `c16_logit_adjustment_posthoc_ls_eps01.ipynb` | Post-hoc logit on best label smoothing |
| Legacy `city_swap_all` / fairness_06 | Earlier baseline BERT checkpoint |

Summary CSVs: `c15_…`, `c16_…`, `c18_…` in this folder; per-run JSON under `notebooks/results/challenger_city_swap/`.

Thesis: Section `subsec:city_swap_scope` in `thesis/chapters/chapter3.tex`.
