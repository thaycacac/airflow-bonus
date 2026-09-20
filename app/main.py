"""
Serve whichever model version .env names.

The API has no training code and no model file. It asks the registry for
`models:/<MODEL_NAME>/<MODEL_VERSION>` at start-up and serves what it gets.
"""
import os
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_NAME = os.getenv("MODEL_NAME", "breast-cancer-classifier")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

model = None
load_error: str | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model, load_error
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    try:
        # Downloads the artefact over HTTP from the tracking server, which is
        # why the server runs with --serve-artifacts.
        model = mlflow.sklearn.load_model(MODEL_URI)
        print(f"loaded {MODEL_URI}")
    except Exception as exc:
        model, load_error = None, f"{type(exc).__name__}: {exc}"
        print(f"could not load {MODEL_URI} -- {load_error}")
    yield


app = FastAPI(title="airflow-bonus — serving from the registry", lifespan=lifespan)


class PredictRequest(BaseModel):
    features: list[float] = Field(..., min_length=30, max_length=30)


@app.get("/health")
def health():
    """Which model is actually being served, not just whether the port answers."""
    if model is None:
        return {"status": "degraded", "model_loaded": False,
                "model_uri": MODEL_URI, "error": load_error}
    return {"status": "ok", "model_loaded": True, "model_uri": MODEL_URI}


@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail=f"{MODEL_URI} not loaded")
    row = pd.DataFrame([req.features], columns=model.feature_names_in_)
    proba = float(model.predict_proba(row)[0][1])
    return {
        "prediction": "benign" if proba >= 0.5 else "malignant",
        "probability_benign": round(proba, 4),
        "served_by": MODEL_URI,
    }
