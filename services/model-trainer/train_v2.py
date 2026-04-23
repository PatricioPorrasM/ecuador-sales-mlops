#!/usr/bin/env python3
"""
Entry point: train the v2 XGBoost model with temporal feature engineering.

Features added over v1:
  lag_1, lag_2            — target values 1 and 2 months prior (per province)
  rolling_mean_3          — 3-month trailing mean of the target (per province)
  rolling_std_3           — 3-month trailing std  of the target (per province)
  mes_sin, mes_cos        — cyclic month encoding (avoids discontinuity at Dec→Jan)

Environment variables (all optional — defaults work out of the box):
  DATA_PATH      Path to the SRI sales CSV.     Default: ./data/Bdd_SRI_2025.csv
  MODEL_DIR      Directory for model artifacts. Default: /app/models
  WANDB_PROJECT  W&B project name.              Default: ecuador-sales-mlops
  WANDB_API_KEY  Required for W&B logging.
"""
import logging
import os

from trainer import DATA_PATH, MODEL_DIR, SalesModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

if __name__ == "__main__":
    trainer = SalesModelTrainer(version="v2")
    run_id = trainer.run_pipeline(
        csv_path=os.environ.get("DATA_PATH", DATA_PATH),
        output_dir=os.environ.get("MODEL_DIR", str(MODEL_DIR)),
    )
    wandb_info = f"W&B run ID: {run_id}" if run_id else "W&B: omitido (sin API key o error de conexión)"
    print(f"\nEntrenamiento V2 completo.  {wandb_info}")