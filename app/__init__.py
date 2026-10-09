from pathlib import Path

from flask import Flask

from .api.routes import api
from .db import init_app as init_db


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=False)
    root = Path(app.root_path).parent
    app.config.from_mapping(
        DATABASE=str(root / "data" / "sentinel.db"),
        MODEL_PATH=str(root / "artifacts" / "ensemble.joblib"),
        MAX_CONTENT_LENGTH=5 * 1024 * 1024,
        JSON_SORT_KEYS=False,
    )
    if test_config:
        app.config.update(test_config)

    Path(app.config["DATABASE"]).parent.mkdir(parents=True, exist_ok=True)
    Path(app.config["MODEL_PATH"]).parent.mkdir(parents=True, exist_ok=True)
    init_db(app)
    app.register_blueprint(api)

    from .views import pages

    app.register_blueprint(pages)
    return app
