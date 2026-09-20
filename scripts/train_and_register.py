"""
Train a classifier and register it in MLflow.

Prefers WDBC parquet from the Airflow staging folder (unscaled). Falls back to
sklearn's breast_cancer dataset when staging is missing — so
`docker compose run trainer` works before the first DAG run.

    docker compose run --rm trainer
    # or from the Airflow ML venv:
    /opt/ml-venv/bin/python /opt/airflow/scripts/train_and_register.py --ds 2026-08-25
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EXPERIMENT = "wdbc-serving"
MODEL_NAME = os.getenv("MODEL_NAME", "breast-cancer-classifier")
N_TREES = int(os.getenv("N_TREES", "200"))

# Resolve project root whether we run in Airflow (/opt/airflow) or ML image (/work).
_CANDIDATES = [
    Path("/opt/airflow/data"),
    Path("/work/data"),
    Path(__file__).resolve().parents[1] / "data",
]
DATA_ROOT = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[-1])
STAGING = DATA_ROOT / "staging"

LABEL_MAP = {"B": 1, "M": 0}  # 1 = benign (matches T03 /predict)


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in ("sample_id", "diagnosis")]


def load_wdbc_staging(ds: str | None) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series] | None:
    """Load train/test from Airflow staging (unscaled) when available."""
    if not ds:
        # Newest staging folder that already has the unscaled splits.
        if not STAGING.exists():
            return None
        candidates = sorted(
            (p for p in STAGING.iterdir() if p.is_dir() and (p / "train_unscaled.parquet").exists()),
            reverse=True,
        )
        if not candidates:
            return None
        run = candidates[0]
    else:
        run = STAGING / ds
        if not (run / "train_unscaled.parquet").exists():
            return None

    train = pd.read_parquet(run / "train_unscaled.parquet")
    test = pd.read_parquet(run / "test_unscaled.parquet")
    feats = _feature_columns(train)
    y_train = train["diagnosis"].map(LABEL_MAP)
    y_test = test["diagnosis"].map(LABEL_MAP)
    if y_train.isna().any() or y_test.isna().any():
        raise SystemExit("diagnosis contains labels outside {B, M}")
    return train[feats], y_train.astype(int), test[feats], y_test.astype(int)


def load_sklearn_fallback() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ds", default=os.getenv("TRAIN_DS"), help="Airflow logical date folder under data/staging/")
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(EXPERIMENT)

    loaded = load_wdbc_staging(args.ds)
    if loaded is None:
        print("staging parquet not found — falling back to sklearn breast_cancer")
        X_train, X_test, y_train, y_test = load_sklearn_fallback()
        source = "sklearn_breast_cancer"
    else:
        X_train, y_train, X_test, y_test = loaded
        source = f"wdbc_staging:{args.ds or 'latest'}"
        print(f"training on {source}  ({len(X_train)} train / {len(X_test)} test)")

    with mlflow.start_run() as run:
        model = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=N_TREES, random_state=42),
        )
        model.fit(X_train, y_train)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        mlflow.log_param("n_estimators", N_TREES)
        mlflow.log_param("data_source", source)
        mlflow.log_metric("test_roc_auc", auc)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=X_train.head(1),
        )
        print(f"run {run.info.run_id}  n_estimators={N_TREES}  roc_auc={auc:.4f}")

    latest = mlflow.MlflowClient().get_registered_model(MODEL_NAME).latest_versions
    version = max(int(v.version) for v in latest)
    print(f"registered {MODEL_NAME} version {version}")
    print(f"to serve it, set MODEL_VERSION={version} in .env and restart the API")


if __name__ == "__main__":
    main()
