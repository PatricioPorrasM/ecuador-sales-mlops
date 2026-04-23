"""
Cargador de modelos para el servicio ml-inference.

Carga el bundle pickle producido por el pipeline de entrenamiento desde
MODEL_PATH (predeterminado: /app/models/model_production.pkl).

Esquema del bundle (definido por SalesModelTrainer.save_model):
  {
    "model":          estimador sklearn/XGBoost entrenado,
    "feature_cols":   list[str]  — nombres de features en orden esperado por model.predict(),
    "label_encoder":  LabelEncoder ajustado sobre la lista de provincias,
    "version":        "v1" | "v2",
    "metrics":        dict con valores train_*/test_*,
    "provinces":      list[str]  — nombres de provincias normalizados a ASCII,
  }

Seguridad de hilos: el _bundle a nivel de módulo se escribe una sola vez
(bloqueo de doble verificación) y luego es solo de lectura, por lo que
no se necesita bloqueo después de la inicialización.
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
    """Elimina diacríticos y convierte a mayúsculas para comparación consistente de provincias."""
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
    Retorna el bundle del modelo cargado, intentando carga diferida desde disco si es necesario.
    Retorna None si el archivo del modelo aún no existe (entrenamiento no completado).
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
    """Retorna True si el bundle del modelo ya está cargado en memoria."""
    return _bundle is not None


# ─────────────────────────────────────────────────────────
# Feature preparation
# ─────────────────────────────────────────────────────────


def prepare_features(bundle: dict, request_data: dict) -> np.ndarray:
    """
    Construye el array numpy (1, n_features) que espera model.predict().

    Para modelos v1, los campos requeridos son las cinco columnas de exportación.
    Para modelos v2, mes_sin/mes_cos se calculan automáticamente desde 'mes';
    lag_1, lag_2, rolling_mean_3, rolling_std_3 se aceptan como campos opcionales
    del cuerpo de la solicitud y por defecto son 0.0 si se omiten. Los llamadores
    deben proporcionarlos para predicciones precisas (p.ej. desde una consulta histórica).

    Lanza ValueError para provincias desconocidas o campos obligatorios faltantes.
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
        "ano_fiscal":        float(request_data.get("ano_fiscal", 2025.0)),
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
    Estima la confianza de la predicción en el rango [0, 1].

    RandomForest (v1): promedio ponderado del consenso entre árboles (inverso del
    coeficiente de variación de todas las predicciones de árboles individuales)
    y el R² del conjunto de prueba. Mayor acuerdo + mayor R² → mayor confianza.

    XGBoost (v2): la API pública de XGBoost no expone las salidas de árboles
    individuales sin pasar por el booster C++, por lo que el R² de prueba se usa
    directamente como proxy estable de la calidad general del modelo.
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
