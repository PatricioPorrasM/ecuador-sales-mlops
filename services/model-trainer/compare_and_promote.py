#!/usr/bin/env python3
"""
Compare v1 and v2 model artifacts in W&B and promote the winner to production.

Promotion rule
──────────────
  v2 is promoted  if  test_rmse_v2 < test_rmse_v1 × PROMOTION_THRESHOLD
                                                      (default 0.95 → ≥5% gain)
  v1 is promoted  otherwise (baseline wins or improvement is marginal).

The winning artifact receives the "production" alias in the W&B Model Registry
and its .pkl file is copied to $MODEL_DIR/model_production.pkl, which is the
path the ml-inference service reads on startup (shared PersistentVolumeClaim).

Environment variables
─────────────────────
  WANDB_PROJECT         W&B project name.            Default: ecuador-sales-mlops
  WANDB_ENTITY          W&B entity (user/org).       Default: inferred from API key
  MODEL_DIR             Output directory.            Default: /app/models
  PROMOTION_THRESHOLD   RMSE ratio threshold.        Default: 0.95
  WANDB_API_KEY         Required.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

import wandb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "ecuador-sales-mlops")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models"))
PROMOTION_THRESHOLD = float(os.environ.get("PROMOTION_THRESHOLD", "0.95"))

ARTIFACT_V1 = "ecuador-sales-model-v1"
ARTIFACT_V2 = "ecuador-sales-model-v2"


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────


def _artifact_slug(name: str, alias: str = "latest") -> str:
    if WANDB_ENTITY:
        return f"{WANDB_ENTITY}/{WANDB_PROJECT}/{name}:{alias}"
    return f"{WANDB_PROJECT}/{name}:{alias}"


def fetch_artifact(api: wandb.Api, name: str) -> wandb.Artifact:
    slug = _artifact_slug(name)
    try:
        artifact = api.artifact(slug)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot fetch artifact '{slug}'. "
            f"Make sure both train_v1.py and train_v2.py have been run first.\n"
            f"Original error: {exc}"
        ) from exc

    rmse = artifact.metadata.get("test_rmse")
    logger.info("Fetched %-35s  test_rmse=%s", slug, f"{rmse:.4f}" if rmse is not None else "N/A")
    return artifact


def promote(artifact: wandb.Artifact, version_label: str) -> None:
    """Add the 'production' alias to the winning artifact in the W&B registry."""
    if "production" not in artifact.aliases:
        artifact.aliases.append("production")
        artifact.save()
        logger.info("Promoted %s → alias 'production' added", version_label)
    else:
        logger.info("%s already carries the 'production' alias; skipping update", version_label)


def download_and_install(artifact: wandb.Artifact) -> Path:
    """
    Download the artifact to a temp directory, copy the .pkl bundle to
    MODEL_DIR/model_production.pkl, then clean up.
    """
    tmp_dir = MODEL_DIR / "_download_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        artifact.download(root=str(tmp_dir))
        pkl_files = list(tmp_dir.glob("**/*.pkl"))
        if not pkl_files:
            raise RuntimeError(
                f"No .pkl file found inside artifact '{artifact.name}'. "
                "Verify that save_model() ran successfully during training."
            )
        src = pkl_files[0]
        dest = MODEL_DIR / "model_production.pkl"
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        logger.info("Production model installed → %s  (source: %s)", dest, src.name)
        return dest
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────


def main() -> None:
    if not os.environ.get("WANDB_API_KEY"):
        logger.error("WANDB_API_KEY is not set. Cannot access the W&B registry.")
        sys.exit(1)

    api = wandb.Api()

    v1 = fetch_artifact(api, ARTIFACT_V1)
    v2 = fetch_artifact(api, ARTIFACT_V2)

    rmse_v1 = v1.metadata.get("test_rmse")
    rmse_v2 = v2.metadata.get("test_rmse")

    if rmse_v1 is None or rmse_v2 is None:
        raise RuntimeError(
            "test_rmse missing from artifact metadata. "
            "Both models must complete training before running this script."
        )

    logger.info(
        "Comparison — V1 RMSE: %.4f  |  V2 RMSE: %.4f  |  threshold: %.0f%%",
        rmse_v1, rmse_v2, (1 - PROMOTION_THRESHOLD) * 100,
    )

    if rmse_v2 < rmse_v1 * PROMOTION_THRESHOLD:
        winner, loser = v2, v1
        winner_label, loser_label = "V2 (XGBoost)", "V1 (RandomForest)"
        winner_rmse = rmse_v2
        improvement_pct = (1 - rmse_v2 / rmse_v1) * 100
        logger.info(
            "→ %s WINS — %.1f%% RMSE improvement over %s",
            winner_label, improvement_pct, loser_label,
        )
    else:
        winner, loser = v1, v2
        winner_label, loser_label = "V1 (RandomForest)", "V2 (XGBoost)"
        winner_rmse = rmse_v1
        logger.info(
            "→ %s WINS — V2 did not achieve ≥%.0f%% improvement (delta: %.2f%%)",
            winner_label,
            (1 - PROMOTION_THRESHOLD) * 100,
            (rmse_v2 / rmse_v1 - 1) * 100,
        )

    promote(winner, winner_label)
    prod_path = download_and_install(winner)

    print("\n" + "─" * 50)
    print(f"  Winner        : {winner_label}")
    print(f"  Winner RMSE   : {winner_rmse:,.4f}")
    print(f"  Loser RMSE    : {rmse_v2 if winner_label.startswith('V1') else rmse_v1:,.4f}")
    print(f"  W&B alias     : production → {winner.name}:{winner.version}")
    print(f"  Model path    : {prod_path}")
    print("─" * 50)


if __name__ == "__main__":
    main()