"""
SalesModelTrainer: reproducible training pipeline for Ecuador SRI sales forecasting.

Architecture: one global model trained on all 25 provinces simultaneously.
Province identity is ordinal-encoded (0–24), giving the model province-aware
predictions while sharing cross-province statistical patterns. Each row in
the training set is a (province, month, year) observation.

Target: SOCIEDADES / TOTAL VENTAS Y EXPORTACIONES (419) for the given province-month.
Split:  chronological — last `test_months` calendar months held out as test set.

Encoding note: the SRI CSV may have encoding issues (e.g. CAÑAR → CA\xffAR).
_ascii_upper() strips diacritics for column matching, so province names in
PROVINCES list are stored ASCII-normalised ("CANAR", not "CAÑAR").
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Configuration (overridable via environment variables)
# ─────────────────────────────────────────────────────────

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "ecuador-sales-mlops")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models"))
DATA_PATH = os.environ.get(
    "DATA_PATH",
    str(Path(__file__).parent / "data" / "Bdd_SRI_2025.csv"),
)

# ─────────────────────────────────────────────────────────
# Province list (ASCII-normalised for column matching)
# ─────────────────────────────────────────────────────────

PROVINCES: list[str] = [
    "AZUAY", "BOLIVAR", "CARCHI", "CANAR", "CHIMBORAZO", "COTOPAXI",
    "EL ORO", "ESMERALDAS", "GALAPAGOS", "GUAYAS", "IMBABURA", "LOJA",
    "LOS RIOS", "MANABI", "MORONA SANTIAGO", "NAPO", "ND", "ORELLANA",
    "PASTAZA", "PICHINCHA", "SANTA ELENA", "SANTO DOMINGO DE LOS TSACHILAS",
    "SUCUMBIOS", "TUNGURAHUA", "ZAMORA CHINCHIPE",
]

# ─────────────────────────────────────────────────────────
# Feature column sets
# ─────────────────────────────────────────────────────────

V1_FEATURES: list[str] = [
    "province_code", "ano_fiscal", "mes_fiscal",
    "exp_bienes_pn", "exp_servicios_pn",
    "exp_bienes_soc", "exp_servicios_soc",
]

V2_EXTRA_FEATURES: list[str] = [
    "lag_1", "lag_2",
    "rolling_mean_3", "rolling_std_3",
    "mes_sin", "mes_cos",
]

# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────


def _ascii_upper(s: str) -> str:
    """Strip diacritics and uppercase — used for robust column matching."""
    return (
        unicodedata.normalize("NFKD", s)
        .encode("ascii", "ignore")
        .decode("ascii")
        .strip()
        .upper()
    )


def _read_csv_robust(path: str) -> pd.DataFrame:
    """Try common encodings in order; raise if all fail."""
    for enc in ("utf-8-sig", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc)
            logger.info("Loaded %s with encoding=%s, shape=%s", path, enc, df.shape)
            return df
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode {path} with utf-8-sig, latin-1, or cp1252")


# ─────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────


class SalesModelTrainer:
    """Trains v1 (RandomForest) or v2 (XGBoost) on Ecuador SRI monthly sales data."""

    def __init__(self, version: str) -> None:
        if version not in ("v1", "v2"):
            raise ValueError("version must be 'v1' or 'v2'")
        self.version = version
        self.model: Any = None
        self.feature_cols: list[str] = []
        self.metrics: dict[str, float] = {}

        self._label_enc = LabelEncoder()
        self._label_enc.fit(PROVINCES)

    # ── Data loading ──────────────────────────────────────

    def load_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load wide-format SRI CSV and reshape to long format.

        Year is reconstructed from sequential row index:
            year  = 2020 + (row_idx // 12)
            month = (row_idx % 12) + 1
        This matches the SRI publication cadence: records 0–11 = 2020 Jan–Dec,
        records 12–23 = 2021 Jan–Dec, etc.
        """
        raw = _read_csv_robust(csv_path)
        raw.columns = [_ascii_upper(c) for c in raw.columns]

        prefix_map = self._build_prefix_map(raw.columns.tolist())

        records: list[dict] = []
        for row_idx, row in raw.iterrows():
            year = 2020 + row_idx // 12
            month = row_idx % 12 + 1

            for prov, pfx in prefix_map.items():
                records.append(
                    {
                        "year": year,
                        "month": month,
                        "province": prov,
                        "ano_fiscal": year,
                        "mes_fiscal": month,
                        "exp_bienes_pn": _safe_float(
                            row.get(f"{pfx}/PERSONAS NATURALES/EXPORTACIONES DE BIENES (417)")
                        ),
                        "exp_servicios_pn": _safe_float(
                            row.get(f"{pfx}/PERSONAS NATURALES/EXPORTACIONES DE SERVICIOS (418)")
                        ),
                        "exp_bienes_soc": _safe_float(row.get(f"{pfx}/SOCIEDADES/EXPORTACIONES DE BIENES (417)")),
                        "exp_servicios_soc": _safe_float(row.get(f"{pfx}/SOCIEDADES/EXPORTACIONES DE SERVICIOS (418)")),
                        "target": _safe_float(row.get(f"{pfx}/SOCIEDADES/TOTAL VENTAS Y EXPORTACIONES (419)")),
                    }
                )

        df = pd.DataFrame(records)
        df["province_code"] = self._label_enc.transform(df["province"])
        df = df.sort_values(["province", "year", "month"]).reset_index(drop=True)
        logger.info(
            "Reshaped dataset: %d rows × %d provinces = %d samples",
            len(raw), len(prefix_map), len(df),
        )
        return df

    @staticmethod
    def _build_prefix_map(columns: list[str]) -> dict[str, str]:
        """
        Return {canonical_province: actual_csv_prefix}.
        Accounts for encoding corruption (e.g. CAÑAR → CANAR in both sides).
        """
        csv_prefixes: dict[str, str] = {}
        for col in columns:
            if "/" in col:
                raw_prefix = col.split("/")[0].strip()
                csv_prefixes[_ascii_upper(raw_prefix)] = raw_prefix

        result: dict[str, str] = {}
        for prov in PROVINCES:
            key = _ascii_upper(prov)
            if key in csv_prefixes:
                result[prov] = csv_prefixes[key]
            else:
                logger.warning("Province '%s' not matched in CSV columns; skipping", prov)
        return result

    # ── Feature engineering ───────────────────────────────

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lag (1, 2) and rolling (mean, std over 3-month window) features,
        computed per province to prevent cross-province leakage.
        All features reference t-1 or earlier — no target leakage.
        """
        df = df.copy()
        for prov in PROVINCES:
            mask = df["province"] == prov
            target = df.loc[mask, "target"]

            df.loc[mask, "lag_1"] = target.shift(1).values
            df.loc[mask, "lag_2"] = target.shift(2).values

            # Rolling window applied to the already-shifted series → window = [t-3, t-2, t-1]
            lagged = target.shift(1)
            df.loc[mask, "rolling_mean_3"] = lagged.rolling(window=3, min_periods=2).mean().values
            df.loc[mask, "rolling_std_3"] = lagged.rolling(window=3, min_periods=2).std().values

        df["mes_sin"] = np.sin(2 * np.pi * df["mes_fiscal"] / 12)
        df["mes_cos"] = np.cos(2 * np.pi * df["mes_fiscal"] / 12)
        return df

    # ── Model construction ────────────────────────────────

    def _build_model(self) -> Any:
        if self.version == "v1":
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
            )
        # v2 — XGBoost tuned for ~1 700 sample dataset
        from xgboost import XGBRegressor  # deferred import; not needed for v1

        return XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=3,
            random_state=42,
            n_jobs=-1,
        )

    def _hparams(self) -> dict[str, Any]:
        if self.version == "v1":
            return {
                "algorithm": "RandomForestRegressor",
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42,
            }
        return {
            "algorithm": "XGBRegressor",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_weight": 3,
            "random_state": 42,
        }

    # ── Evaluation ────────────────────────────────────────

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        nonzero = y_true != 0
        mape = (
            float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
            if nonzero.any()
            else float("nan")
        )
        return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}

    # ── Full pipeline ─────────────────────────────────────

    def train(self, csv_path: str = DATA_PATH, test_months: int = 12) -> dict[str, Any]:
        """
        End-to-end training:
          load → feature engineering → chronological split → fit → evaluate.

        Returns dict with keys 'hparams' and 'metrics' (prefixed train_/test_).
        """
        df = self.load_data(csv_path)

        if self.version == "v2":
            df = self._add_temporal_features(df)

        self.feature_cols = V1_FEATURES if self.version == "v1" else V1_FEATURES + V2_EXTRA_FEATURES
        df = df.dropna(subset=self.feature_cols).reset_index(drop=True)

        # Chronological split: hold out the last `test_months` distinct periods
        periods = (
            df[["year", "month"]]
            .drop_duplicates()
            .sort_values(["year", "month"])
            .reset_index(drop=True)
        )
        cutoff = periods.iloc[-test_months]
        train_mask = (df["year"] < cutoff["year"]) | (
            (df["year"] == cutoff["year"]) & (df["month"] < cutoff["month"])
        )

        X_train = df.loc[train_mask, self.feature_cols].values
        y_train = df.loc[train_mask, "target"].values
        X_test = df.loc[~train_mask, self.feature_cols].values
        y_test = df.loc[~train_mask, "target"].values

        logger.info("Split — train: %d samples | test: %d samples", len(X_train), len(X_test))

        self.model = self._build_model()
        self.model.fit(X_train, y_train)

        train_m = self._compute_metrics(y_train, self.model.predict(X_train))
        test_m = self._compute_metrics(y_test, self.model.predict(X_test))

        self.metrics = {
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"test_{k}": v for k, v in test_m.items()},
        }
        logger.info(
            "[%s] Test  RMSE=%.2f  MAE=%.2f  MAPE=%.2f%%  R²=%.4f",
            self.version.upper(),
            test_m["rmse"], test_m["mae"], test_m["mape"], test_m["r2"],
        )
        return {"hparams": self._hparams(), "metrics": self.metrics}

    def save_model(self, output_dir: Path | str = MODEL_DIR) -> Path:
        """
        Pickle the model bundle to {output_dir}/model_{version}.pkl.
        The bundle includes everything the inference service needs at runtime:
        model, feature column list, label encoder, and evaluation metrics.
        """
        if self.model is None:
            raise RuntimeError("Call train() before save_model()")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"model_{self.version}.pkl"

        bundle = {
            "model": self.model,
            "feature_cols": self.feature_cols,
            "label_encoder": self._label_enc,
            "version": self.version,
            "metrics": self.metrics,
            "provinces": PROVINCES,
        }
        with open(path, "wb") as fh:
            pickle.dump(bundle, fh, protocol=5)

        logger.info("Model bundle saved → %s", path)
        return path

    def log_to_wandb(self, model_path: Path) -> str | None:
        """
        Registra la ejecución en Weights & Biases: hiperparámetros, métricas,
        importancia de features y artefacto del modelo.

        Retorna el W&B run ID, o None si W&B no está disponible o falla.
        El entrenamiento local NO se interrumpe por fallos de W&B.
        """
        if not os.environ.get("WANDB_API_KEY"):
            logger.info("WANDB_API_KEY no configurada — omitiendo logging a W&B.")
            return None

        run_name = "v1-random-forest" if self.version == "v1" else "v2-xgboost"

        try:
            run = wandb.init(
                project=WANDB_PROJECT,
                name=run_name,
                tags=[self.version, "training"],
                config=self._hparams(),
            )

            wandb.log(self.metrics)

            importances = sorted(
                zip(self.feature_cols, self.model.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )
            wandb.log({
                "feature_importance": wandb.Table(
                    columns=["feature", "importance"],
                    data=[[f, round(imp, 6)] for f, imp in importances],
                )
            })

            artifact = wandb.Artifact(
                name=f"ecuador-sales-model-{self.version}",
                type="model",
                description=(
                    f"Ecuador SRI sales model {self.version.upper()} "
                    f"({self._hparams()['algorithm']})"
                ),
                metadata=self.metrics,
            )
            artifact.add_file(str(model_path))
            run.log_artifact(artifact)

            run_id = run.id
            wandb.finish()

            logger.info("W&B run registrado — id=%s  nombre=%s", run_id, run_name)
            return run_id

        except Exception as exc:
            logger.warning(
                "No se pudo registrar en W&B (%s). "
                "El modelo local fue guardado correctamente en %s.",
                exc, model_path,
            )
            return None

    def run_pipeline(
        self,
        csv_path: str = DATA_PATH,
        output_dir: Path | str = MODEL_DIR,
    ) -> str | None:
        """Entrena → guarda → registra en W&B (opcional). Retorna el W&B run ID o None."""
        self.train(csv_path)
        model_path = self.save_model(output_dir)
        return self.log_to_wandb(model_path)


# ─────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────


def _safe_float(value: Any) -> float:
    """Convert a potentially missing or non-numeric cell to float."""
    if value is None:
        return 0.0
    try:
        f = float(value)
        return 0.0 if math.isnan(f) else f
    except (ValueError, TypeError):
        return 0.0
