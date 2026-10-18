"""API integration tests (PRD Section 13: API integration tests required)."""
from __future__ import annotations

import os
import tempfile

from fastapi.testclient import TestClient

# Use an isolated SQLite file for the test database before importing the app.
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp.close()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp.name}"

from api.database import engine, init_db  # noqa: E402
from api.main import app  # noqa: E402

# TestClient does not fire startup events when not used as a context
# manager, so create the schema explicitly for the isolated test DB.
init_db()

client = TestClient(app)


def teardown_module() -> None:
    engine.dispose()
    try:
        os.unlink(_tmp.name)
    except PermissionError:  # Windows file-lock race with SQLite
        pass


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_metadata() -> None:
    r = client.get("/model-metadata")
    assert r.status_code == 200
    body = r.json()
    assert body["model_name"]
    assert len(body["feature_columns"]) > 10


def test_predict_single() -> None:
    """Prediction with an all-zeros feature map must still return a valid
    probability (missing features default to 0)."""
    r = client.get("/model-metadata")
    cols = r.json()["feature_columns"]
    payload = {"features": {c: 0 for c in cols}, "customer_id": 12345}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_label"] in (0, 1)
    assert body["segment"] >= -1


def test_predict_by_id_and_explain() -> None:
    meta = client.get("/model-metadata")
    assert meta.status_code == 200
    # Pick a customer id that exists in the feature store.
    import pandas as pd

    from src.utils.config import load_config, resolve_path

    cfg = load_config()
    table = pd.read_parquet(resolve_path(cfg.get("paths.processed_features")))
    cid = int(table["customer_id"].iloc[0])

    r = client.get(f"/predict/{cid}")
    assert r.status_code == 200

    e = client.get(f"/explain/{cid}")
    assert e.status_code == 200
    body = e.json()
    assert body["top_contributors"]
    assert "plain_language" in body


def test_predict_by_id_404() -> None:
    r = client.get("/predict/999999999")
    assert r.status_code == 404


def test_batch_csv_scoring() -> None:
    import io

    import pandas as pd

    from src.utils.config import load_config, resolve_path

    cfg = load_config()
    table = pd.read_parquet(resolve_path(cfg.get("paths.processed_features")))
    meta = client.get("/model-metadata").json()
    sample = table.head(5)[["customer_id"] + meta["feature_columns"]]
    buf = io.BytesIO()
    sample.to_csv(buf, index=False)
    buf.seek(0)
    r = client.post(
        "/predict/batch",
        files={"file": ("batch.csv", buf.getvalue(), "text/csv")},
    )
    assert r.status_code == 200
    assert r.json()["scored"] == 5
    assert len(r.json()["predictions"]) == 5


def test_invalid_csv_rejected() -> None:
    # Undecodable UTF-8 makes pandas fail loudly -> 400 with a clear message.
    r = client.post(
        "/predict/batch",
        files={"file": ("bad.csv", b"\xff\xfe\xfa binary garbage \x81\x82", "text/csv")},
    )
    assert r.status_code == 400
