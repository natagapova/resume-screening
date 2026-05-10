"""
Shared helpers for adversarial pair evaluation (notebook 58).

Imports model utilities from notebooks/english/e07_english_eval_common.py.
Each model run writes only under:
  notebooks/results/adversarial_model_eval/<model_run_id>/
  figures/adversarial_model_eval/<model_run_id>/
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

# e07_english_eval_common lives in notebooks/english/
import sys

_EN = Path(__file__).resolve().parent.parent / "english"
if str(_EN) not in sys.path:
    sys.path.insert(0, str(_EN))

from e07_english_eval_common import (  # noqa: E402
    MODEL_CHECKPOINT_CANDIDATES,
    filter_to_encoder_labels,
    load_model_bundle,
    predict_logits_batch,
    resolve_model_dir,
    resolve_repo_root,
)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def run_adversarial_pair_model_eval(
    model_run_id: str,
    *,
    pairs_csv: Path | None = None,
    batch_size: int = 8,
    max_length: int = 256,
    seed: int = 42,
    results_subdir: str = "adversarial_model_eval",
    figures_subdir: str = "adversarial_model_eval",
) -> Dict[str, Any]:
    """
    Evaluate one checkpoint on anchor vs counterfactual texts from notebook 56 CSV.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    repo_root = resolve_repo_root()
    if pairs_csv is None:
        pairs_csv = repo_root / "notebooks" / "results" / "adversarial_resume_pairs" / "adversarial_resume_pairs.csv"
    pairs_csv = Path(pairs_csv)
    if not pairs_csv.is_file():
        raise FileNotFoundError(f"Missing adversarial pairs CSV: {pairs_csv}")

    model_dir, tried_paths = resolve_model_dir(repo_root, model_run_id)
    results_root = repo_root / "notebooks" / "results" / results_subdir / model_run_id
    figures_root = repo_root / "figures" / figures_subdir / model_run_id
    results_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    df = pd.read_csv(pairs_csv)
    req = [
        "pair_id",
        "anchor_resume_text",
        "counterfactual_resume_text",
        "supercategory",
        "attribute_edited",
    ]
    for c in req:
        if c not in df.columns:
            raise KeyError(f"pairs CSV missing column {c!r}: {pairs_csv}")

    bundle = load_model_bundle(model_dir, device)
    le = bundle.label_encoder

    work = df.copy()
    work["anchor_resume_text"] = work["anchor_resume_text"].fillna("").astype(str)
    work["counterfactual_resume_text"] = work["counterfactual_resume_text"].fillna("").astype(str)
    work, n_drop = filter_to_encoder_labels(work, "supercategory", le)
    if len(work) == 0:
        raise ValueError("No rows left after filtering to encoder labels")

    y_idx = le.transform(work["supercategory"].astype(str))
    t_a = work["anchor_resume_text"].tolist()
    t_c = work["counterfactual_resume_text"].tolist()

    logits_a = predict_logits_batch(t_a, bundle, device, batch_size=batch_size, max_length=max_length)
    logits_c = predict_logits_batch(t_c, bundle, device, batch_size=batch_size, max_length=max_length)

    pred_a = logits_a.argmax(axis=-1)
    pred_c = logits_c.argmax(axis=-1)
    probs_a = _softmax(logits_a)
    probs_c = _softmax(logits_c)

    n = len(work)
    idx = np.arange(n)
    p_true_a = probs_a[idx, y_idx]
    p_true_c = probs_c[idx, y_idx]
    flip = pred_a != pred_c

    out = work.copy()
    out["y_true_idx"] = y_idx
    out["pred_anchor_idx"] = pred_a
    out["pred_counterfactual_idx"] = pred_c
    out["pred_anchor_label"] = le.inverse_transform(pred_a)
    out["pred_counterfactual_label"] = le.inverse_transform(pred_c)
    out["prob_true_class_anchor"] = p_true_a
    out["prob_true_class_counterfactual"] = p_true_c
    out["prediction_flipped"] = flip.astype(bool)
    out["delta_logit_true_class"] = logits_c[idx, y_idx] - logits_a[idx, y_idx]

    out["delta_prob_true_class"] = p_true_c - p_true_a

    pred_path = results_root / "predictions_adversarial_pairs.csv"
    out.to_csv(pred_path, index=False)

    flip_rate = float(flip.mean())
    delta_prob_mean = float(out["delta_prob_true_class"].mean())

    by_attr = (
        out.groupby("attribute_edited", dropna=False)
        .agg(
            n_pairs=("pair_id", "count"),
            flip_rate=("prediction_flipped", "mean"),
            mean_delta_prob_true=("delta_prob_true_class", "mean"),
        )
        .reset_index()
    )

    by_class = (
        out.groupby("supercategory", dropna=False)
        .agg(
            n_pairs=("pair_id", "count"),
            flip_rate=("prediction_flipped", "mean"),
            mean_delta_prob_true=("delta_prob_true_class", "mean"),
        )
        .reset_index()
    )

    summary: Dict[str, Any] = {
        "model_run_id": model_run_id,
        "n_pairs_evaluated": int(len(out)),
        "rows_dropped_unknown_label": int(n_drop),
        "flip_rate": flip_rate,
        "mean_delta_prob_true_class": delta_prob_mean,
        "by_attribute_edited": by_attr.to_dict(orient="records"),
        "by_true_class": by_class.to_dict(orient="records"),
        "pairs_csv": str(pairs_csv.relative_to(repo_root)),
        "predictions_csv": str(pred_path.relative_to(repo_root)),
        "resolved_model_dir": str(model_dir),
        "model_resolution_tried": tried_paths,
        "device": str(device),
    }

    (results_root / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "model_run_id": model_run_id,
        "pid": os.getpid(),
        "results_dir": str(results_root),
        "figures_dir": str(figures_root),
        "metrics_json": str((results_root / "metrics.json").relative_to(repo_root)),
    }
    (results_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        attrs = by_attr["attribute_edited"].astype(str).tolist()
        ax.bar(attrs, by_attr["flip_rate"].tolist(), color="#4c78a8")
        ax.set_ylabel("Flip rate")
        ax.set_title(f"{model_run_id} — prediction flip by attribute edit")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fp = figures_root / "flip_rate_by_attribute.png"
        fig.savefig(fp, dpi=160, bbox_inches="tight")
        plt.close(fig)
        summary["figure_flip_rate_by_attribute"] = str(fp.relative_to(repo_root))
    except Exception as exc:  # noqa: BLE001
        summary["figure_error"] = str(exc)

    del bundle, logits_a, logits_c
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary
