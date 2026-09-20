"""Print one real WDBC row as a /predict body."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_CANDIDATES = [
    Path("/opt/airflow/data/raw/wdbc.csv"),
    Path("/work/data/raw/wdbc.csv"),
    Path(__file__).resolve().parents[1] / "data" / "raw" / "wdbc.csv",
]
RAW = next(p for p in _CANDIDATES if p.exists())

frame = pd.read_csv(RAW)
feats = [c for c in frame.columns if c not in ("sample_id", "diagnosis")]
# First malignant row — the API should agree.
row = frame.loc[frame["diagnosis"] == "M", feats].iloc[0]
print(json.dumps({"features": [float(v) for v in row.tolist()]}))
