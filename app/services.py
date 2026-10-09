from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from flask import current_app

from .db import get_db, utc_now
from .ml import FEATURES, PredictiveMaintenanceModel, load_model


ASSETS = ["TURB-01", "PUMP-07", "COMP-12", "MOTOR-04", "FAN-09", "PRESS-03"]
MODEL_CACHE = {}


def generate_demo_data(samples_per_asset=120, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    start = datetime.now(timezone.utc) - timedelta(hours=samples_per_asset - 1)
    profiles = {
        "TURB-01": (73, 2.2, 101, 3200, 74, 210),
        "PUMP-07": (66, 3.1, 92, 2800, 68, 560),
        "COMP-12": (82, 4.0, 118, 3500, 84, 740),
        "MOTOR-04": (61, 1.8, 96, 2400, 59, 180),
        "FAN-09": (57, 2.5, 89, 2100, 52, 330),
        "PRESS-03": (78, 3.7, 112, 3100, 80, 690),
    }
    for asset_index, asset in enumerate(ASSETS):
        base = profiles[asset]
        for step in range(samples_per_asset):
            cycle = np.sin(step / 11 + asset_index) * 0.8
            degradation = max(0, step - 82) / 38 if asset in {"COMP-12", "PRESS-03"} else 0
            shock = 1 if rng.random() < (0.02 + degradation * 0.11) else 0
            temperature = base[0] + cycle + rng.normal(0, 1.4) + degradation * 13 + shock * 9
            vibration = base[1] + rng.normal(0, 0.28) + degradation * 2.3 + shock * 1.8
            pressure = base[2] + rng.normal(0, 2.0) - degradation * 8 - shock * 5
            rpm = base[3] + rng.normal(0, 48) + degradation * 135
            load = base[4] + rng.normal(0, 3.2) + degradation * 9
            service = base[5] + step
            failure = int(
                temperature > 91
                or vibration > 5.8
                or (pressure < 101 and load > 84)
                or shock
            )
            rows.append(
                {
                    "asset_id": asset,
                    "recorded_at": (start + timedelta(hours=step)).isoformat(),
                    "temperature": round(float(temperature), 3),
                    "vibration": round(float(vibration), 3),
                    "pressure": round(float(pressure), 3),
                    "rpm": round(float(rpm), 3),
                    "load_pct": round(float(load), 3),
                    "hours_since_service": round(float(service), 3),
                    "is_failure": failure,
                }
            )
    return pd.DataFrame(rows)


def insert_readings(frame):
    db = get_db()
    columns = ["asset_id", "recorded_at", *FEATURES, "is_failure"]
    db.executemany(
        f"INSERT INTO readings ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
        [tuple(row[col] for col in columns) for _, row in frame.iterrows()],
    )
    db.commit()


def ensure_demo_data():
    db = get_db()
    count = db.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
    if count == 0:
        insert_readings(generate_demo_data())


def get_training_frame():
    return pd.read_sql_query(
        "SELECT temperature, vibration, pressure, rpm, load_pct, hours_since_service, is_failure FROM readings",
        get_db(),
    )


def train_and_score(epochs=45):
    frame = get_training_frame()
    if len(frame) < 40:
        raise ValueError("At least 40 telemetry rows are required to train the models")
    if frame["is_failure"].nunique() < 2:
        raise ValueError("Training data needs both normal and failure examples")
    model = PredictiveMaintenanceModel()
    result = model.fit(frame, epochs=epochs)
    model_path = current_app.config["MODEL_PATH"]
    model.save(model_path)
    MODEL_CACHE[model_path] = model

    db = get_db()
    raw = pd.read_sql_query(
        "SELECT id, temperature, vibration, pressure, rpm, load_pct, hours_since_service FROM readings",
        db,
    )
    scores = model.predict_frame(raw)
    db.executemany(
        "UPDATE readings SET risk_score=?, anomaly_score=?, reconstruction_error=? WHERE id=?",
        [
            (float(scores["risk"][i]), float(scores["anomaly"][i]), float(scores["reconstruction"][i]), int(row.id))
            for i, row in raw.iterrows()
        ],
    )
    db.execute(
        """INSERT INTO training_runs
           (trained_at, sample_count, failure_count, accuracy, precision_score, recall, f1, threshold, epochs)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            utc_now(), result.sample_count, result.failure_count, result.accuracy,
            result.precision, result.recall, result.f1, result.threshold, result.epochs,
        ),
    )
    db.commit()
    return result


def get_model():
    path = current_app.config["MODEL_PATH"]
    if path not in MODEL_CACHE:
        MODEL_CACHE[path] = load_model(path)
    return MODEL_CACHE[path]


def bootstrap():
    ensure_demo_data()
    if get_model() is None or get_db().execute("SELECT COUNT(*) FROM training_runs").fetchone()[0] == 0:
        train_and_score(epochs=35)


def validate_upload(frame):
    required = {"asset_id", "recorded_at", *FEATURES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    clean = frame.copy()
    clean["is_failure"] = clean.get("is_failure", 0)
    for feature in FEATURES:
        clean[feature] = pd.to_numeric(clean[feature], errors="raise")
    clean["is_failure"] = pd.to_numeric(clean["is_failure"], errors="raise").astype(int).clip(0, 1)
    clean["asset_id"] = clean["asset_id"].astype(str).str.strip().str.slice(0, 32)
    clean["recorded_at"] = pd.to_datetime(clean["recorded_at"], utc=True).astype(str)
    if clean.empty or len(clean) > 20_000:
        raise ValueError("CSV must contain between 1 and 20,000 rows")
    return clean[["asset_id", "recorded_at", *FEATURES, "is_failure"]]

