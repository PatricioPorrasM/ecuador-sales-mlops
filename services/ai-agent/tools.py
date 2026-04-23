"""
Tool implementations for the ReAct agent.

Two tools are exposed:

  get_province_data(provincia, mes)
    Reads the SRI CSV and returns the most recent export feature values
    for the requested (province, month) pair.  These are the features
    the inference model expects as input.

  call_inference(provincia, mes, exp_bienes_pn, ...)
    HTTP POST to the ml-inference /predict endpoint and returns the
    full prediction response.

The CSV is loaded once at first call and cached in memory for subsequent
requests (LRU cache on _load_wide_df).

Province name matching is ASCII-normalised on both sides so that names like
"Guayas", "guayas", "GUAYAS" all resolve correctly regardless of the
encoding used to store the CSV column headers.
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
                "province and month. Use the feature values returned by "
                "get_province_data as arguments."
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
                    "provincia", "mes",
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
    """Strip diacritics and uppercase for robust column matching."""
    return (
        unicodedata.normalize("NFKD", s)
        .encode("ascii", "ignore")
        .decode("ascii")
        .strip()
        .upper()
    )


@lru_cache(maxsize=1)
def _load_wide_df() -> pd.DataFrame:
    """Load the SRI CSV once and cache it in memory."""
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
    Return the actual CSV column prefix (e.g. 'GUAYAS') for a province.
    Matching is done after ASCII-normalisation so encoding issues are invisible.
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
    Return the most recent export feature values for (provincia, mes).

    Row index → date mapping (from the SRI publication cadence):
        year  = 2020 + row_idx // 12
        month = row_idx % 12 + 1

    For a given month we find all matching rows, pick the most recent year,
    and extract the five export columns for the requested province.
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
    exportaciones_bienes_pn: float,
    exportaciones_servicios_pn: float,
    exportaciones_bienes_soc: float,
    exportaciones_servicios_soc: float,
) -> dict[str, Any]:
    """
    POST to ml-inference /predict and return the response payload.
    Returns a dict with an 'error' key on any failure.
    """
    payload = {
        "provincia": provincia,
        "mes": mes,
        "exportaciones_bienes_pn":    exportaciones_bienes_pn,
        "exportaciones_servicios_pn": exportaciones_servicios_pn,
        "exportaciones_bienes_soc":   exportaciones_bienes_soc,
        "exportaciones_servicios_soc":exportaciones_servicios_soc,
    }
    url = f"{ML_INFERENCE_URL}/predict"

    try:
        resp = requests.post(url, json=payload, timeout=INFERENCE_TIMEOUT)
        resp.raise_for_status()
        result: dict = resp.json()
        logger.info(
            "call_inference(%s, %d) → prediction=%.2f  model=%s  confidence=%.4f",
            provincia, mes,
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
    except Exception as exc:
        return {"error": f"Unexpected error calling inference service: {exc}"}
