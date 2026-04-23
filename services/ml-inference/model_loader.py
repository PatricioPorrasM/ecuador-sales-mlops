"""
Model loader for the ml-inference service.

Loads the pickle bundle produced by the training pipeline from
MODEL_PATH (default: /app/models/model_production.pkl).

The bundle schema (set by SalesModelTrainer.save_model):
  {
    "model":          trained sklearn/XGBoost estimator,
    "feature_cols":   list[str]  — ordered feature names expected by model.predict(),
    "label_encoder":  LabelEncoder fitted on PROVINCES list,
    "version":        "v1" | "v2",
    "metrics":        dict with train_*/test_* metric values,
    "provinces":      list[str]  — ASCII-normalised province names,
  }

Thread safety: the module-level _bundle is written once (double-checked locking)
and then read-only, so no lock is needed after initialisation.
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import threading
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/app/models/model_production.pkl"))

_bundle: dict | None = None
_load_lock = threading.Lock()


# ─────────────────────────────────────────────────────────
# Province name normalisation (mirrors trainer.py logic)
# ─────────────────────────────────────────────────────────


def _ascii_upper(s: str) -> str:
    """Strip diacritics and uppercase for consistent province matching."""
    return (
        unicodedata.normalize("NFKD", s)
        .encode("ascii", "ignore")
        .decode("ascii")
        .strip()
        .upper()
    )


# ─────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────


def get_bundle() -> dict | None:
    """
    Return the loaded model bundle, attempting a lazy load from disk if needed.
    Returns None if the model file does not exist yet (training not completed).
    """
    global _bundle

    if _bundle is not None:
        return _bundle

    with _load_lock:
        if _bundle is not None:  # another thread loaded it while we waited
            return _bundle

        if not MODEL_PATH.exists():
            logger.warning("Model file not found at %s", MODEL_PATH)
            return None

        try:
            with open(MODEL_PATH, "rb") as fh:
                _bundle = pickle.load(fh)
            logger.info(
                "Model bundle loaded — version=%s  features=%s",
                _bundle.get("version"),
                _bundle.get("feature_cols"),
            )
        except Exception as exc:
            logger.error("Failed to deserialise model bundle: %s", exc)

    return _bundle


def is_model_loaded() -> bool:
    return _bundle is not None


# ─────────────────────────────────────────────────────────
# Feature preparation
# ─────────────────────────────────────────────────────────


def prepare_features(bundle: dict, request_data: dict) -> np.ndarray:
    """
    Build the (1, n_features) numpy array that model.predict() expects.

    For v1 models the required fields are the five export columns.
    For v2 models, mes_sin/mes_cos are computed automatically from 'mes';
    lag_1, lag_2, rolling_mean_3, rolling_std_3 are accepted as optional
    request body fields and default to 0.0 when omitted.  Callers should
    provide them for accurate forecasts (e.g. from a historical lookup).

    Raises ValueError for unknown provinces or missing required fields.
    """
    label_enc = bundle["label_encoder"]
    feature_cols: list[str] = bundle["feature_cols"]
    provinces: list[str] = bundle.get("provinces", list(label_enc.classes_))

    provincia_norm = _ascii_upper(str(request_data.get("provincia", "")))
    if provincia_norm not in provinces:
        raise ValueError(
            f"Unknown province '{request_data.get('provincia')}'. "
            f"Valid values: {sorted(provinces)}"
        )

    province_code = int(label_enc.transform([provincia_norm])[0])
    mes = int(request_data["mes"])

    feature_map: dict[str, float] = {
        "province_code":     float(province_code),
        "mes_fiscal":        float(mes),
        "exp_bienes_pn":     float(request_data.get("exportaciones_bienes_pn", 0.0)),
        "exp_servicios_pn":  float(request_data.get("exportaciones_servicios_pn", 0.0)),
        "exp_bienes_soc":    float(request_data.get("exportaciones_bienes_soc", 0.0)),
        "exp_servicios_soc": float(request_data.get("exportaciones_servicios_soc", 0.0)),
        # v2 temporal features (computed or provided)
        "lag_1":             float(request_data.get("lag_1", 0.0)),
        "lag_2":             float(request_data.get("lag_2", 0.0)),
        "rolling_mean_3":    float(request_data.get("rolling_mean_3", 0.0)),
        "rolling_std_3":     float(request_data.get("rolling_std_3", 0.0)),
        "mes_sin":           math.sin(2 * math.pi * mes / 12),
        "mes_cos":           math.cos(2 * math.pi * mes / 12),
    }

    X = np.array([[feature_map[col] for col in feature_cols]], dtype=np.float64)
    return X


# ─────────────────────────────────────────────────────────
# Confidence estimation
# ─────────────────────────────────────────────────────────


def compute_confidence(bundle: dict, X: np.ndarray) -> float:
    """
    Estimate prediction confidence in [0, 1].

    RandomForest (v1): weighted average of tree consensus (inverse coefficient
    of variation across all tree predictions) and held-out test R².
    High agreement between trees + high R² → high confidence.

    XGBoost (v2): XGBoost's public API does not expose individual tree outputs
    without going through the C++ booster, so test R² is used directly as a
    stable proxy for overall model quality.
    """
    model = bundle["model"]
    test_r2 = max(0.0, float(bundle["metrics"].get("test_r2", 0.5)))

    if bundle["version"] == "v1":
        tree_preds: np.ndarray = np.column_stack(
            [tree.predict(X) for tree in model.estimators_]
        )[0]
        mean_p = float(tree_preds.mean())
        std_p = float(tree_preds.std())
        cv = std_p / (abs(mean_p) + 1e-8)
        tree_confidence = 1.0 / (1.0 + cv)
        confidence = 0.5 * tree_confidence + 0.5 * test_r2
    else:
        confidence = test_r2

    return round(min(1.0, max(0.0, confidence)), 4)
