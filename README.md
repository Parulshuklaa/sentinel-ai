# SentinelAI

SentinelAI is a full-stack predictive-maintenance platform that detects risky industrial equipment before it fails. It combines a PyTorch deep autoencoder with an Isolation Forest, exposes inference through a versioned Flask API, stores telemetry and training history in SQLite, and presents the results in a responsive operations dashboard.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-111?logo=flask)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch&logoColor=white)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)

## Why this project stands out

- Real hybrid ML: a neural autoencoder learns normal operating behavior while an Isolation Forest catches statistical outliers.
- End-to-end product: ingestion, validation, training, model persistence, inference, REST endpoints, database history, and UI.
- Interview-ready demo: tune six sensor values in the Prediction Lab and get a live risk score plus maintenance recommendation.
- Production-minded details: app factory, API versioning, request limits, health check, responsive UI, deployment config, and tests.

## Architecture

```text
Telemetry CSV / demo generator
             |
             v
   Validation + SQLite store
             |
      +------+------+
      |             |
      v             v
PyTorch AE    Isolation Forest
      |             |
      +------+------+
             v
       Weighted risk score
             |
      Flask REST API v1
             |
  Dashboard + Prediction Lab
```

The autoencoder compresses six standardized sensor readings through a `6 -> 12 -> 4 -> 12 -> 6` network. Its reconstruction error receives 62% of the ensemble weight; the Isolation Forest anomaly score receives 38%. The decision threshold is calibrated from labeled failure examples in the training split.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
flask --app run.py run --debug
```

Open `http://127.0.0.1:5000`. On first use, SentinelAI generates 720 deterministic telemetry observations, trains both models, stores the artifact, and scores the fleet.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | Service health check |
| `GET` | `/api/v1/overview` | KPIs, assets, trend, and model metrics |
| `GET` | `/api/v1/alerts` | Recent high-risk events |
| `POST` | `/api/v1/predict` | Score one operating state |
| `POST` | `/api/v1/train` | Retrain and rescore all telemetry |
| `POST` | `/api/v1/upload` | Validate CSV, ingest, and retrain |

Example prediction:

```bash
curl -X POST http://127.0.0.1:5000/api/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"temperature":106,"vibration":7.2,"pressure":82,"rpm":3900,"load_pct":96,"hours_since_service":1020}'
```

## CSV schema

Required columns: `asset_id`, `recorded_at`, `temperature`, `vibration`, `pressure`, `rpm`, `load_pct`, `hours_since_service`. `is_failure` is optional for ingestion, but a useful training dataset needs both `0` and `1` labels.

## Test

```bash
pytest -q
```

## Interview talking points

1. Why unsupervised anomaly detection is useful when real failure labels are scarce.
2. How scaling and leakage-safe train/test splitting affect anomaly models.
3. Why the ensemble is more robust than either model alone.
4. How model artifacts and database scores stay consistent after retraining.
5. Where to evolve next: PostgreSQL, background jobs, drift detection, authentication, and streaming with Kafka.

## Responsible-use note

This is an educational portfolio system. Synthetic data and risk scores should not be used to schedule maintenance on real equipment without domain validation, calibration, and monitoring.

