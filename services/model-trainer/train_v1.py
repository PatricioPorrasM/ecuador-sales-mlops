#!/usr/bin/env python3
"""
Entry point: train the v1 RandomForest baseline model.

Environment variables (all optional — defaults work out of the box):
  DATA_PATH   Path to the SRI sales CSV.       Default: ./data/Bdd_SRI_2025.csv
  MODEL_DIR   Directory for model artifacts.   Default: /app/models
  WANDB_PROJECT  W&B project name.             Default: ecuador-sales-mlops
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
    trainer = SalesModelTrainer(version="v1")
    run_id = trainer.run_pipeline(
        csv_path=os.environ.get("DATA_PATH", DATA_PATH),
        output_dir=os.environ.get("MODEL_DIR", str(MODEL_DIR)),
    )
    print(f"\nV1 training complete.  W&B run ID: {run_id}")
