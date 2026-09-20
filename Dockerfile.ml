# ML image for mlflow / api / trainer — separate from the Airflow pin set.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /work

COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt

COPY scripts/ ./scripts/
COPY app/ ./app/

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////work/mlflow-data/mlflow.db", \
     "--serve-artifacts", \
     "--artifacts-destination", "file:///work/mlflow-data/mlartifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
