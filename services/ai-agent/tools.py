"""
Implementación de herramientas para el agente ReAct.

Se exponen dos herramientas:

  get_province_data(provincia, mes)
    Lee el CSV del SRI y retorna los valores de features de exportación más
    recientes para el par (provincia, mes) solicitado. Estos son los features
    que el modelo de inferencia espera como entrada.

  call_inference(provincia, mes, exp_bienes_pn, ...)
    HTTP POST al endpoint /predict de ml-inference y retorna la respuesta
    completa de predicción.

El CSV se carga una sola vez en la primera llamada y se almacena en caché
para solicitudes posteriores (caché LRU sobre _load_wide_df).

La coincidencia de nombres de provincias se normaliza a ASCII en ambos lados
para que nombres como "Guayas", "guayas", "GUAYAS" se resuelvan correctamente
independientemente de la codificación usada en las cabeceras del CSV.
"""

from __future__ import annotations

import logging
import math
import os
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DATA_PATH = Path(os.environ.get("DATA_PATH", "/app/data/Bdd_SRI_2025.csv"))
ML_INFERENCE_URL: str = os.environ.get("ML_INFERENCE_URL", "http://ml-inference:5000")
INFERENCE_TIMEOUT: int = int(os.environ.get("INFERENCE_TIMEOUT_SECS", "10"))

# ── Temporal validation ───────────────────────────────────────────────────────
# Dataset covers Jan 2020 (record 1) to Sep 2025 (record 69).
# Year/month derived from row index: year = 2020 + (idx // 12), month = idx % 12 + 1
# First predictable month: October 2025.
CUTOFF_YEAR: int = 2025
CUTOFF_MONTH: int = 9                            # Sep 2025 — last dataset record
PREDICTION_START_YEAR: int = 2025
PREDICTION_START_MONTH: int = 10                 # Oct 2025 — first valid prediction
MAX_PREDICTION_MONTHS: int = int(os.environ.get("MAX_PREDICTION_MONTHS", "12"))

_MESES_ES = {
    1: "enero", 2: "febrero", 3: "marzo", 4: "abril",
    5: "mayo", 6: "junio", 7: "julio", 8: "agosto",
    9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre",
}


def _mes_nombre(mes: int) -> str:
    """Retorna el nombre del mes en español para un número de mes dado."""
    return _MESES_ES.get(mes, str(mes))


def _prediction_window() -> tuple[int, int, int, int]:
    """Retorna (año_inicio, mes_inicio, año_fin, mes_fin) de la ventana de predicción válida."""
    start_total = PREDICTION_START_YEAR * 12 + PREDICTION_START_MONTH
    end_total = start_total + MAX_PREDICTION_MONTHS - 1
    end_year, end_month = divmod(end_total, 12)
    if end_month == 0:          # divmod(N*12, 12) gives remainder 0 → December of prior year
        end_year -= 1
        end_month = 12
    return PREDICTION_START_YEAR, PREDICTION_START_MONTH, end_year, end_month


def validate_prediction_date(ano: int, mes: int) -> str | None:
    """
    Valida que (ano, mes) esté dentro de la ventana de predicción permitida.

    Retorna None si la fecha es válida.
    Retorna un mensaje de error en español si no lo es, para que el agente
    lo transmita al usuario sin llamar nunca al servicio de inferencia ML.
    """
    sy, sm, ey, em = _prediction_window()
    req = ano * 12 + mes
    start = sy * 12 + sm
    end = ey * 12 + em

    limite = f"{_mes_nombre(em)} de {ey}"
    if req < start:
        return (
            f"No es posible predecir {_mes_nombre(mes)} de {ano}. "
            f"El dataset del SRI cubre hasta septiembre 2025 (registro 69/69); "
            f"el primer mes predecible es octubre 2025. "
            f"Solicita una fecha entre octubre 2025 y {limite}."
        )
    if req > end:
        return (
            f"No es posible predecir {_mes_nombre(mes)} de {ano}. "
            f"El horizonte máximo es {limite} "
            f"({MAX_PREDICTION_MONTHS} meses desde octubre 2025). "
            f"Solicita una fecha dentro del rango: octubre 2025 — {limite}."
        )
    return None


# ─────────────────────────────────────────────────────────
# OpenAI-compatible tool schema (passed to litellm.completion)
# ─────────────────────────────────────────────────────────

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_province_data",
            "description": (
                "Fetch the most recent historical export data for a specific province "
                "and month from the Ecuador SRI dataset. Returns the feature values "
                "needed to build the inference request payload."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "provincia": {
                        "type": "string",
                        "description": (
                            "Province name in uppercase Spanish, e.g. 'GUAYAS', "
                            "'PICHINCHA', 'AZUAY'. Must match one of the 25 provinces "
                            "in the SRI dataset."
                        ),
                    },
                    "mes": {
                        "type": "integer",
                        "description": "Month number: 1 = January, 12 = December.",
                        "minimum": 1,
                        "maximum": 12,
                    },
                },
                "required": ["provincia", "mes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_inference",
            "description": (
                "Call the ML inference service to obtain a sales prediction for a "
                "province, month and year. Validates that the requested date falls "
                "within the predictable window (Oct 2025 onwards) before calling "
                "the model. Use the feature values returned by get_province_data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "provincia": {
                        "type": "string",
                        "description": "Province name in uppercase.",
                    },
                    "mes": {
                        "type": "integer",
                        "description": "Month number (1–12).",
                        "minimum": 1,
                        "maximum": 12,
                    },
                    "ano": {
                        "type": "integer",
                        "description": (
                            "Year of the prediction (e.g. 2025, 2026). "
                            "Must be strictly after September 2025."
                        ),
                        "minimum": 2025,
                        "maximum": 2030,
                    },
                    "exportaciones_bienes_pn": {
                        "type": "number",
                        "description": "Personas Naturales — Exportaciones de Bienes.",
                    },
                    "exportaciones_servicios_pn": {
                        "type": "number",
                        "description": "Personas Naturales — Exportaciones de Servicios.",
                    },
                    "exportaciones_bienes_soc": {
                        "type": "number",
                        "description": "Sociedades — Exportaciones de Bienes.",
                    },
                    "exportaciones_servicios_soc": {
                        "type": "number",
                        "description": "Sociedades — Exportaciones de Servicios.",
                    },
                },
                "required": [
                    "provincia", "mes", "ano",
                    "exportaciones_bienes_pn", "exportaciones_servicios_pn",
                    "exportaciones_bienes_soc", "exportaciones_servicios_soc",
                ],
            },
        },
    },
]


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────


def _ascii_upper(s: str) -> str:
    """Elimina diacríticos y convierte a mayúsculas para una comparación robusta de columnas."""
    return (
        unicodedata.normalize("NFKD", s)
        .encode("ascii", "ignore")
        .decode("ascii")
        .strip()
        .upper()
    )


@lru_cache(maxsize=1)
def _load_wide_df() -> pd.DataFrame:
    """Carga el CSV del SRI una sola vez y lo almacena en caché en memoria."""
    for enc in ("utf-8-sig", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(DATA_PATH, encoding=enc)
            df.columns = [_ascii_upper(c) for c in df.columns]
            logger.info("CSV loaded from %s (encoding=%s, rows=%d)", DATA_PATH, enc, len(df))
            return df
        except UnicodeDecodeError:
            continue
    raise RuntimeError(
        f"Could not decode {DATA_PATH} with utf-8-sig, latin-1, or cp1252. "
        "Check that DATA_PATH points to the correct file."
    )


def _find_csv_prefix(df: pd.DataFrame, provincia: str) -> str | None:
    """
    Retorna el prefijo real de columna en el CSV (p.ej. 'GUAYAS') para una provincia.
    La comparación se hace tras normalización ASCII para que los problemas de
    codificación sean invisibles.
    """
    norm = _ascii_upper(provincia)
    seen: set[str] = set()
    for col in df.columns:
        if "/" in col:
            raw_prefix = col.split("/")[0].strip()
            if raw_prefix in seen:
                continue
            seen.add(raw_prefix)
            if _ascii_upper(raw_prefix) == norm:
                return raw_prefix
    return None


def _safe_float(value: Any) -> float:
    """Convierte un valor a float de forma segura; retorna 0.0 si es inválido o NaN."""
    try:
        f = float(value)
        return 0.0 if math.isnan(f) else f
    except (ValueError, TypeError):
        return 0.0


# ─────────────────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────────────────


def get_province_data(provincia: str, mes: int) -> dict[str, Any]:
    """
    Retorna los valores de features de exportación más recientes para (provincia, mes).

    Mapeo índice de fila → fecha (según cadencia de publicación del SRI):
        año   = 2020 + índice_fila // 12
        mes   = índice_fila % 12 + 1

    Para un mes dado, encuentra todas las filas coincidentes, selecciona el año
    más reciente y extrae las cinco columnas de exportación para la provincia solicitada.
    """
    try:
        df = _load_wide_df()
    except RuntimeError as exc:
        return {"error": str(exc)}

    prefix = _find_csv_prefix(df, provincia)
    if prefix is None:
        return {
            "error": (
                f"Province '{provincia}' not found in the dataset. "
                "Use uppercase and check spelling against the 25 valid provinces."
            )
        }

    # Annotate rows with reconstructed year/month
    df = df.copy()
    df["_year"] = 2020 + df.index // 12
    df["_month"] = df.index % 12 + 1

    monthly = df[df["_month"] == mes]
    if monthly.empty:
        return {"error": f"No data found for month {mes}"}

    row = monthly.sort_values("_year", ascending=False).iloc[0]
    year_ref = int(row["_year"])

    result: dict[str, Any] = {
        "provincia": _ascii_upper(provincia),
        "mes": mes,
        "ano_referencia": year_ref,
        "exportaciones_bienes_pn":    _safe_float(row.get(f"{prefix}/PERSONAS NATURALES/EXPORTACIONES DE BIENES (417)")),
        "exportaciones_servicios_pn": _safe_float(row.get(f"{prefix}/PERSONAS NATURALES/EXPORTACIONES DE SERVICIOS (418)")),
        "exportaciones_bienes_soc":   _safe_float(row.get(f"{prefix}/SOCIEDADES/EXPORTACIONES DE BIENES (417)")),
        "exportaciones_servicios_soc":_safe_float(row.get(f"{prefix}/SOCIEDADES/EXPORTACIONES DE SERVICIOS (418)")),
        "total_ventas_referencia":    _safe_float(row.get(f"{prefix}/SOCIEDADES/TOTAL VENTAS Y EXPORTACIONES (419)")),
    }

    logger.info(
        "get_province_data(%s, %d) → year_ref=%d  total_ref=%.0f",
        provincia, mes, year_ref, result["total_ventas_referencia"],
    )
    return result


def call_inference(
    provincia: str,
    mes: int,
    ano: int,
    exportaciones_bienes_pn: float,
    exportaciones_servicios_pn: float,
    exportaciones_bienes_soc: float,
    exportaciones_servicios_soc: float,
) -> dict[str, Any]:
    """
    Valida la fecha solicitada y hace POST a ml-inference /predict.

    La validación temporal se realiza primero — si (ano, mes) está fuera de la
    ventana de predicción válida, el servicio ML nunca es llamado y se retorna
    un mensaje de error en español directamente al agente.
    """
    # ── Date validation (short-circuit before any HTTP call) ─────────────────
    date_error = validate_prediction_date(ano, mes)
    if date_error:
        logger.info(
            "call_inference blocked by temporal validation: ano=%d mes=%d — %s",
            ano, mes, date_error,
        )
        return {"error": date_error, "error_tipo": "fecha_fuera_de_rango"}

    # ── Build payload with ano_fiscal ─────────────────────────────────────────
    payload = {
        "provincia":               provincia,
        "mes":                     mes,
        "ano_fiscal":              ano,
        "exportaciones_bienes_pn":    exportaciones_bienes_pn,
        "exportaciones_servicios_pn": exportaciones_servicios_pn,
        "exportaciones_bienes_soc":   exportaciones_bienes_soc,
        "exportaciones_servicios_soc": exportaciones_servicios_soc,
    }
    url = f"{ML_INFERENCE_URL}/predict"

    try:
        resp = requests.post(url, json=payload, timeout=INFERENCE_TIMEOUT)
        resp.raise_for_status()
        result: dict = resp.json()
        logger.info(
            "call_inference(%s, %d/%d) → prediction=%.2f  model=%s  confidence=%.4f",
            provincia, mes, ano,
            result.get("prediccion_total_ventas", 0),
            result.get("modelo_version", "?"),
            result.get("confianza", 0),
        )
        return result
    except requests.exceptions.Timeout:
        return {"error": f"Inference service timed out after {INFERENCE_TIMEOUT}s"}
    except requests.exceptions.ConnectionError:
        return {"error": f"Cannot connect to inference service at {url}"}
    except requests.exceptions.HTTPError as exc:
        body = exc.response.text[:200] if exc.response else ""
        return {"error": f"Inference service HTTP {exc.response.status_code}: {body}"}
    except (ValueError, KeyError) as exc:
        return {"error": f"Unexpected error calling inference service: {exc}"}
