# Airflow image: stock Airflow plus the two libraries the DAG imports.
# ML train deps live in an isolated venv so Airflow's constrained pins stay intact.
FROM apache/airflow:2.8.4-python3.11

USER airflow

ARG AIRFLOW_VERSION=2.8.4
ARG PYTHON_VERSION=3.11
RUN pip install --no-cache-dir \
      --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt" \
      "pandas==2.1.4" \
      "pyarrow==14.0.2"

USER root
COPY requirements-ml.txt /requirements-ml.txt
# PIP_USER is set in the Airflow image and breaks venv installs — clear it here.
RUN python3.11 -m venv /opt/ml-venv \
 && PIP_USER=0 /opt/ml-venv/bin/pip install --no-cache-dir -r /requirements-ml.txt \
 && chown -R airflow:root /opt/ml-venv
USER airflow
