import sqlite3
from datetime import datetime, timezone

from flask import current_app, g


SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    temperature REAL NOT NULL,
    vibration REAL NOT NULL,
    pressure REAL NOT NULL,
    rpm REAL NOT NULL,
    load_pct REAL NOT NULL,
    hours_since_service REAL NOT NULL,
    is_failure INTEGER NOT NULL DEFAULT 0,
    risk_score REAL,
    anomaly_score REAL,
    reconstruction_error REAL
);
CREATE INDEX IF NOT EXISTS idx_readings_asset_time
    ON readings(asset_id, recorded_at DESC);
CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trained_at TEXT NOT NULL,
    sample_count INTEGER NOT NULL,
    failure_count INTEGER NOT NULL,
    accuracy REAL NOT NULL,
    precision_score REAL NOT NULL,
    recall REAL NOT NULL,
    f1 REAL NOT NULL,
    threshold REAL NOT NULL,
    epochs INTEGER NOT NULL
);
"""


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(_error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_database():
    db = get_db()
    db.executescript(SCHEMA)
    db.commit()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def init_app(app):
    app.teardown_appcontext(close_db)
    with app.app_context():
        init_database()

