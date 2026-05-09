"""
Shared English evaluation helpers for notebooks under notebooks/english/.

Designed for parallel runs: each model uses a dedicated subdirectory under
notebooks/results/english_eval/<model_run_id>/ and figures/english_eval/<model_run_id>/.

Do not import from training code; this module is evaluation-only.
"""

from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------


def resolve_repo_root() -> Path:
    cwd = Path.cwd().resolve()
    candidates = [
        cwd,
        cwd.parent,
        cwd.parent.parent,
    ]
    for root in candidates:
        if (root / "data" / "processed").is_dir() and (root / "notebooks").is_dir():
            return root
    return cwd.parent if (cwd / "english_eval_common.py").exists() else cwd


def english_notebook_dir() -> Path:
    """Directory containing this file (notebooks/english/)."""
    return Path(__file__).resolve().parent


def default_english_eval_datasets(repo_root: Path) -> List[Dict[str, Any]]:
    """Canonical English artifacts from notebooks 32 / 54."""
    align_results = repo_root / "notebooks" / "results" / "english_dataset_alignment"
    balanced_results = repo_root / "notebooks" / "results" / "english_balanced_eval"
    processed = repo_root / "data" / "processed" / "english_dataset_v1"
    rows = [
        {
            "dataset_id": "english_train",
            "csv_path": processed / "train.csv",
            "text_col": "resume_text",
            "label_col": "supercategory",
        },
        {
            "dataset_id": "english_val",
            "csv_path": processed / "val.csv",
            "text_col": "resume_text",
            "label_col": "supercategory",
        },
        {
            "dataset_id": "english_test",
            "csv_path": processed / "test.csv",
            "text_col": "resume_text",
            "label_col": "supercategory",
        },
        {
            "dataset_id": "english_soft_eval_slice",
            "csv_path": balanced_results / "english_balanced_8class_eval.csv",
            "text_col": "resume_text",
            "label_col": "supercategory",
        },
    ]
    # Optional: base eval slice from 32 (city rows)
    rows.append(
        {
            "dataset_id": "english_base_eval_slice",
            "csv_path": align_results / "32_english_base_eval_slice.csv",
            "text_col": "resume_text",
            "label_col": "label",
        }
    )
    return rows


# Model run id -> ordered list of relative paths (first existing wins).
# Optional first hop: notebooks/models/english/<run_id> (symlink or copy — user-maintained).
MODEL_CHECKPOINT_CANDIDATES: Dict[str, List[str]] = {
    "bert_9classes_final": [
        "notebooks/models/english/bert_9classes_final",
        "models/bert_9classes_final",
        "notebooks/models/bert_9classes_final",
    ],
    "bert_scrubbing": [
        "notebooks/models/english/bert_scrubbing",
        "models/bert_scrubbing",
        "notebooks/models/bert_scrubbing",
    ],
    "label_smoothing_eps01_2ep": [
        "notebooks/models/english/label_smoothing_eps01_2ep",
        "notebooks/models/challengers/label_smoothing_eps01_2ep",
        "models/challengers/label_smoothing_eps01_2ep",
    ],
}


def resolve_model_dir(repo_root: Path, model_run_id: str) -> Tuple[Path, List[str]]:
    rels = MODEL_CHECKPOINT_CANDIDATES.get(model_run_id)
    if not rels:
        raise KeyError(f"Unknown model_run_id={model_run_id!r}. Known: {sorted(MODEL_CHECKPOINT_CANDIDATES)}")
    tried: List[str] = []
    for rel in rels:
        p = (repo_root / rel).resolve()
        tried.append(str(p))
        if p.is_dir() and _directory_looks_like_model(p):
            return p, tried
    raise FileNotFoundError(
        f"No checkpoint directory found for {model_run_id}. Tried:\n  - " + "\n  - ".join(tried)
    )


def _directory_looks_like_model(path: Path) -> bool:
    if (path / "config.json").exists():
        return True
    if (path / "bert" / "config.json").exists():
        return True
    for sub in path.iterdir() if path.is_dir() else []:
        if sub.is_dir() and (sub / "config.json").exists():
            return True
    return False


# -----------------------------------------------------------------------------
# Model loading (same patterns as notebooks/70_city_swap_batch_eval.ipynb)
# -----------------------------------------------------------------------------


def find_model_files(model_dir: Path) -> Tuple[Optional[str], Optional[Path]]:
    model_dir = Path(model_dir)
    if (model_dir / "config.json").exists():
        return "hf", model_dir
    if (model_dir / "bert" / "config.json").exists() and (
        (model_dir / "classifier.pt").exists() or (model_dir / "model.pt").exists()
    ):
        return "split_classifier", model_dir
    candidates = [sub for sub in model_dir.iterdir() if sub.is_dir() and (sub / "config.json").exists()]
    if candidates:
        finals = [c for c in candidates if c.name == "final"]
        if finals:
            return "hf", finals[0]
        return "hf", sorted(candidates, key=lambda p: str(p))[-1]
    return None, None


def _extract_state_dict(obj: Any) -> Optional[dict]:
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    if isinstance(obj, dict):
        for key in ["state_dict", "model_state_dict", "classifier_state_dict"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        return obj
    return None


def _load_split_classifier_weights(model: nn.Module, model_dir: Path) -> bool:
    model_dir = Path(model_dir)
    classifier_paths = [model_dir / "classifier.pt", model_dir / "model.pt"]
    target_state = model.classifier.state_dict()

    for candidate in classifier_paths:
        if not candidate.exists():
            continue
        try:
            obj = torch.load(candidate, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(candidate, map_location="cpu")
        state = _extract_state_dict(obj)
        if not isinstance(state, dict):
            continue

        remapped = {}
        for key, value in state.items():
            norm_key = key.replace("module.", "")
            if norm_key.startswith("classifier."):
                norm_key = norm_key[len("classifier.") :]
            if norm_key in target_state and getattr(value, "shape", None) == target_state[norm_key].shape:
                remapped[norm_key] = value

        if set(remapped.keys()) == set(target_state.keys()):
            model.classifier.load_state_dict(remapped, strict=True)
            return True

    return False


@dataclass
class ModelBundle:
    model: nn.Module
    tokenizer: Any
    label_encoder: LabelEncoder
    hf_path: Path
    fmt: str


def load_model_bundle(model_dir: Path, device: torch.device) -> ModelBundle:
    fmt, path = find_model_files(model_dir)
    if fmt is None or path is None:
        raise FileNotFoundError(f"No model files under {model_dir}")

    model_dir = Path(model_dir)
    if fmt == "split_classifier":
        bert_path = path / "bert"
        tokenizer_path = path if (path / "tokenizer.json").exists() else bert_path
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
        model, loading_info = AutoModelForSequenceClassification.from_pretrained(
            str(bert_path), local_files_only=True, output_loading_info=True
        )
        missing_keys = set(loading_info.get("missing_keys", []))
        if any(k.startswith("classifier.") for k in missing_keys) and not _load_split_classifier_weights(model, path):
            raise RuntimeError(f"Could not restore classifier head from {path}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        model, loading_info = AutoModelForSequenceClassification.from_pretrained(
            str(path), local_files_only=True, output_loading_info=True
        )
        missing_keys = set(loading_info.get("missing_keys", []))
        if any(k.startswith("classifier.") for k in missing_keys):
            raise RuntimeError(f"Missing classifier weights: {sorted(missing_keys)}")

    model = model.to(device).eval()

    le = None
    for le_path in [
        path / "label_encoder.joblib",
        path.parent / "label_encoder.joblib",
        model_dir / "label_encoder.joblib",
    ]:
        if le_path.exists():
            le = joblib.load(le_path)
            break
    if le is None:
        raise FileNotFoundError(f"label_encoder.joblib not found near {path}")

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        label_encoder=le,
        hf_path=path,
        fmt=fmt,
    )


# -----------------------------------------------------------------------------
# Inference + metrics
# -----------------------------------------------------------------------------


def filter_to_encoder_labels(df: pd.DataFrame, label_col: str, le: LabelEncoder) -> Tuple[pd.DataFrame, int]:
    known = set(le.classes_)
    mask = df[label_col].astype(str).isin(known)
    return df.loc[mask].reset_index(drop=True), int((~mask).sum())


def predict_logits_batch(
    texts: List[str],
    bundle: ModelBundle,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """Return logits as float32 array of shape (n_samples, num_labels)."""
    model = bundle.model
    tokenizer = bundle.tokenizer
    chunks: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="predict", leave=False):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits.float().cpu().numpy()
        chunks.append(logits)
        del enc, logits
    if not chunks:
        return np.zeros((0, len(bundle.label_encoder.classes_)), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32)


def predict_texts(
    texts: List[str],
    bundle: ModelBundle,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    logits = predict_logits_batch(texts, bundle, device, batch_size=batch_size, max_length=max_length)
    return logits.argmax(axis=-1).astype(int)


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, le: LabelEncoder) -> Dict[str, Any]:
    labels = list(range(len(le.classes_)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "n_evaluated": int(len(y_true)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=list(le.classes_),
            zero_division=0,
            output_dict=True,
        ),
    }


def run_english_eval_pipeline(
    model_run_id: str,
    *,
    batch_size: int = 16,
    max_length: int = 128,
    seed: int = 42,
    skip_missing_csv: bool = True,
    results_subdir: str = "english_eval",
    figures_subdir: str = "english_eval",
) -> Dict[str, Any]:
    """
    Load one checkpoint and evaluate on all configured English CSVs.
    Writes only under results/<results_subdir>/<model_run_id>/ and figures/<figures_subdir>/<model_run_id>/.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    repo_root = resolve_repo_root()
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

    bundle = load_model_bundle(model_dir, device)
    le = bundle.label_encoder

    run_manifest: Dict[str, Any] = {
        "model_run_id": model_run_id,
        "repo_root": str(repo_root),
        "resolved_model_dir": str(model_dir),
        "model_resolution_tried": tried_paths,
        "device": str(device),
        "batch_size": batch_size,
        "max_length": max_length,
        "seed": seed,
        "results_dir": str(results_root),
        "figures_dir": str(figures_root),
        "pid": os.getpid(),
        "english_pipeline_note": "Data from 32 (alignment) / 54 (soft slice); see canonical manifest in english_dataset_alignment.",
    }

    per_dataset: Dict[str, Any] = {}
    datasets_meta: List[Dict[str, Any]] = []

    for spec in default_english_eval_datasets(repo_root):
        ds_id = spec["dataset_id"]
        csv_path: Path = spec["csv_path"]
        text_col = spec["text_col"]
        label_col = spec["label_col"]

        if not csv_path.is_file():
            meta = {"dataset_id": ds_id, "status": "skipped_missing_csv", "csv_path": str(csv_path)}
            datasets_meta.append(meta)
            if not skip_missing_csv:
                raise FileNotFoundError(f"Missing CSV for {ds_id}: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        if text_col not in df.columns:
            raise KeyError(f"{ds_id}: column {text_col!r} not in {list(df.columns)}")
        if label_col not in df.columns:
            raise KeyError(f"{ds_id}: column {label_col!r} not in {list(df.columns)}")

        work = df[[text_col, label_col]].copy()
        work = work.rename(columns={text_col: "resume_text", label_col: "supercategory"})
        work["resume_text"] = work["resume_text"].fillna("").astype(str)
        work, n_dropped = filter_to_encoder_labels(work, "supercategory", le)
        if len(work) == 0:
            datasets_meta.append(
                {
                    "dataset_id": ds_id,
                    "status": "skipped_no_rows_after_label_filter",
                    "rows_dropped_unknown_label": n_dropped,
                }
            )
            continue

        y_true = le.transform(work["supercategory"].astype(str))
        texts = work["resume_text"].tolist()

        y_pred = predict_texts(texts, bundle, device, batch_size=batch_size, max_length=max_length)

        out_rows = work.copy()
        out_rows["y_true_idx"] = y_true
        out_rows["y_pred_idx"] = y_pred
        out_rows["y_true_label"] = le.inverse_transform(y_true)
        out_rows["y_pred_label"] = le.inverse_transform(y_pred)

        pred_path = results_root / f"predictions_{ds_id}.csv"
        out_rows.to_csv(pred_path, index=False)

        m = metrics_dict(y_true, y_pred, le)
        m["rows_total_in_csv"] = int(len(df))
        m["rows_dropped_unknown_label"] = n_dropped
        m["predictions_csv"] = str(pred_path.relative_to(repo_root))
        per_dataset[ds_id] = m
        datasets_meta.append({"dataset_id": ds_id, "status": "ok", "predictions_csv": str(pred_path)})

        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import ConfusionMatrixDisplay

            fig, ax = plt.subplots(figsize=(10, 10))
            ConfusionMatrixDisplay.from_predictions(
                y_true,
                y_pred,
                display_labels=list(le.classes_),
                xticks_rotation=90,
                ax=ax,
                include_values=False,
                cmap="Blues",
            )
            ax.set_title(f"{model_run_id} — {ds_id}")
            fig.tight_layout()
            fig_path = figures_root / f"confusion_matrix_{ds_id}.png"
            fig.savefig(fig_path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            datasets_meta[-1]["confusion_matrix_png"] = str(fig_path.relative_to(repo_root))
        except Exception as exc:  # noqa: BLE001
            datasets_meta[-1]["confusion_matrix_error"] = str(exc)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Release model before returning (helps parallel notebook runs exit cleanly)
    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary = {
        "model_run_id": model_run_id,
        "per_dataset": per_dataset,
        "datasets": datasets_meta,
    }
    (results_root / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    run_manifest["metrics_json"] = str((results_root / "metrics.json").relative_to(repo_root))
    run_manifest["completed_datasets"] = datasets_meta
    (results_root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return summary


__all__ = [
    "resolve_repo_root",
    "resolve_model_dir",
    "MODEL_CHECKPOINT_CANDIDATES",
    "load_model_bundle",
    "predict_logits_batch",
    "run_english_eval_pipeline",
    "default_english_eval_datasets",
]
