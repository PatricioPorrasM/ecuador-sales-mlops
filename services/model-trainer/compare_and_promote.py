#!/usr/bin/env python3
"""
Compara los modelos v1 y v2 y promueve el ganador a producción.

Modos de operación
──────────────────
  Modo local (predeterminado):
    Lee las métricas directamente desde los archivos pkl generados por
    train_v1.py y train_v2.py en MODEL_DIR. No requiere W&B.
    El ganador se copia como MODEL_DIR/model_production.pkl.

  Modo W&B (cuando WANDB_API_KEY está configurada):
    Adicionalmente registra el alias "production" en el registry de W&B.
    Si W&B falla, recae automáticamente en el modo local.

Regla de promoción
──────────────────
  v2 se promueve  si  test_rmse_v2 < test_rmse_v1 × PROMOTION_THRESHOLD
                                                      (default 0.95 → ≥5% mejora)
  v1 se promueve  en caso contrario (baseline gana o la mejora es marginal).

Variables de entorno
────────────────────
  WANDB_PROJECT         Nombre del proyecto W&B.     Default: ecuador-sales-mlops
  WANDB_ENTITY          Entidad W&B (usuario/org).   Default: inferido desde API key
  MODEL_DIR             Directorio de salida.        Default: /app/models
  PROMOTION_THRESHOLD   Umbral de ratio RMSE.        Default: 0.95
  WANDB_API_KEY         Opcional — habilita el modo W&B.
"""

from __future__ import annotations

import logging
import os
import pickle
import shutil
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "ecuador-sales-mlops")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models"))
PROMOTION_THRESHOLD = float(os.environ.get("PROMOTION_THRESHOLD", "0.95"))


# ─────────────────────────────────────────────────────────
# Modo local — lee pkl desde disco
# ─────────────────────────────────────────────────────────

def _load_bundle(version: str) -> dict:
    """Carga el bundle pkl de un modelo desde MODEL_DIR."""
    path = MODEL_DIR / f"model_{version}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo {version} en {path}. "
            f"Asegúrate de que train_{version}.py haya terminado correctamente."
        )
    with open(path, "rb") as fh:
        bundle = pickle.load(fh)
    rmse = bundle["metrics"].get("test_rmse")
    logger.info(
        "Cargado %-20s  test_rmse=%s",
        path.name,
        f"{rmse:,.4f}" if rmse is not None else "N/A",
    )
    return bundle


def _install_winner(version: str) -> Path:
    """Copia model_{version}.pkl → model_production.pkl en MODEL_DIR."""
    src = MODEL_DIR / f"model_{version}.pkl"
    dest = MODEL_DIR / "model_production.pkl"
    shutil.copy2(src, dest)
    logger.info("Modelo de producción instalado → %s  (fuente: %s)", dest, src.name)
    return dest


def _local_promote() -> None:
    """Compara métricas locales y promueve el ganador sin necesitar W&B."""
    bundle_v1 = _load_bundle("v1")
    bundle_v2 = _load_bundle("v2")

    rmse_v1 = bundle_v1["metrics"]["test_rmse"]
    rmse_v2 = bundle_v2["metrics"]["test_rmse"]

    logger.info(
        "Comparación — V1 RMSE: %s  |  V2 RMSE: %s  |  umbral: %.0f%%",
        f"{rmse_v1:,.4f}", f"{rmse_v2:,.4f}", (1 - PROMOTION_THRESHOLD) * 100,
    )

    if rmse_v2 < rmse_v1 * PROMOTION_THRESHOLD:
        winner_ver, loser_ver = "v2", "v1"
        winner_label, loser_label = "V2 (XGBoost)", "V1 (RandomForest)"
        winner_rmse, loser_rmse = rmse_v2, rmse_v1
        improvement = (1 - rmse_v2 / rmse_v1) * 100
        logger.info(
            "→ %s GANA — %.1f%% de mejora en RMSE sobre %s",
            winner_label, improvement, loser_label,
        )
    else:
        winner_ver, loser_ver = "v1", "v2"
        winner_label, loser_label = "V1 (RandomForest)", "V2 (XGBoost)"
        winner_rmse, loser_rmse = rmse_v1, rmse_v2
        logger.info(
            "→ %s GANA — V2 no alcanzó ≥%.0f%% de mejora (delta: %.2f%%)",
            winner_label,
            (1 - PROMOTION_THRESHOLD) * 100,
            (rmse_v2 / rmse_v1 - 1) * 100,
        )

    prod_path = _install_winner(winner_ver)

    print("\n" + "─" * 50)
    print(f"  Ganador       : {winner_label}")
    print(f"  RMSE ganador  : {winner_rmse:,.4f}")
    print(f"  RMSE perdedor : {loser_rmse:,.4f}")
    print(f"  Ruta modelo   : {prod_path}")
    print("─" * 50)


# ─────────────────────────────────────────────────────────
# Modo W&B — registra alias "production" en el registry
# ─────────────────────────────────────────────────────────

def _wandb_tag_winner(winner_label: str, winner_version: str) -> None:
    """
    Añade el alias 'production' al artefacto ganador en el registry de W&B.
    Falla silenciosamente si W&B no está disponible.
    """
    try:
        import wandb

        api = wandb.Api()
        slug_parts = [WANDB_ENTITY, WANDB_PROJECT] if WANDB_ENTITY else [WANDB_PROJECT]
        slug = "/".join(slug_parts + [f"ecuador-sales-model-{winner_version}:latest"])

        artifact = api.artifact(slug)
        if "production" not in artifact.aliases:
            artifact.aliases.append("production")
            artifact.save()
            logger.info("W&B alias 'production' añadido a %s", slug)
        else:
            logger.info("Artefacto %s ya tiene el alias 'production'", slug)
    except Exception as exc:
        logger.warning("No se pudo actualizar el alias en W&B: %s", exc)


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main() -> None:
    """Punto de entrada: modo local siempre; W&B si WANDB_API_KEY está disponible."""
    # Siempre ejecutar la comparación local (no depende de red)
    _local_promote()

    # Intentar añadir alias en W&B de forma opcional
    if os.environ.get("WANDB_API_KEY"):
        logger.info("WANDB_API_KEY detectada — intentando actualizar alias en W&B…")
        # Determinar ganador nuevamente para pasar a W&B
        try:
            bundle_v1 = _load_bundle("v1")
            bundle_v2 = _load_bundle("v2")
            rmse_v1 = bundle_v1["metrics"]["test_rmse"]
            rmse_v2 = bundle_v2["metrics"]["test_rmse"]
            winner_ver = "v2" if rmse_v2 < rmse_v1 * PROMOTION_THRESHOLD else "v1"
            winner_label = "V2 (XGBoost)" if winner_ver == "v2" else "V1 (RandomForest)"
            _wandb_tag_winner(winner_label, winner_ver)
        except Exception as exc:
            logger.warning("Fallo al actualizar W&B (no crítico): %s", exc)
    else:
        logger.info("WANDB_API_KEY no configurada — omitiendo actualización de W&B.")


if __name__ == "__main__":
    main()
