"""FastAPI application entrypoint (PRD Section 12.1).

Endpoints:
* ``GET  /health``            – service health check
* ``POST /predict``           – single-customer prediction (JSON features)
* ``POST /predict/batch``     – batch scoring (CSV upload or JSON records)
* ``GET  /predict/{customer_id}`` – score an existing customer by ID
* ``GET  /explain/{customer_id}`` – SHAP explanation for one customer
* ``GET  /model-metadata``    – model card + comparison table + segments

Swagger auto-docs are enabled at ``/docs`` by FastAPI default.
"""
from __future__ import annotations

import io
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, UploadFile
from sqlalchemy.orm import Session

from api.database import Prediction, SegmentAssignment, SessionLocal, init_db
from api.schemas.predict import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    ExplainResponse,
    ModelMetadata,
    PredictionResponse,
)
from api.service import ModelService
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Vantara Customer Intelligence Platform", version="1.0.0")
service = ModelService()


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def _startup() -> None:
    init_db()
    # Persist segment assignments once so the dashboard reads them from DB.
    db = SessionLocal()
    try:
        if db.query(SegmentAssignment).count() == 0 and service.segment_profiles is not None:
            labels = {}
            for p in service.segment_profiles.to_dict(orient="records"):
                labels[int(p["segment"])] = p["segment_label"]
            if service.segment_labels is not None:
                for _, r in service.segment_labels.iterrows():
                    db.add(
                        SegmentAssignment(
                            customer_id=int(r["customer_id"]),
                            segment=int(r["segment"]),
                            segment_label=labels.get(int(r["segment"]), "unknown"),
                        )
                    )
                db.commit()
                logger.info("Persisted %d segment assignments", len(service.segment_labels))
    finally:
        db.close()


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "model": service.model_name}


@app.get("/model-metadata", response_model=ModelMetadata)
def model_metadata() -> ModelMetadata:
    """Model card for the production churn model."""
    meta = service.metadata()
    return ModelMetadata(
        model_name=meta["model_name"],
        feature_columns=meta["feature_columns"],
        metrics=meta["metrics"],
        training_data_rows=meta["training_data_rows"],
    )


@app.get("/segments")
def segments() -> dict[str, Any]:
    """Segment profiles with business-readable labels."""
    return {"segments": service.metadata()["segment_profiles"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(body: CustomerFeatures) -> PredictionResponse:
    """Score one customer from a feature map."""
    try:
        return PredictionResponse(**service.predict(body.features, body.customer_id))
    except Exception as exc:  # noqa: BLE001 - surfaced as 400 with message
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/predict/{customer_id}", response_model=PredictionResponse)
def predict_by_id(customer_id: int) -> PredictionResponse:
    """Score an existing customer straight from the feature store."""
    try:
        return PredictionResponse(**service.predict_by_id(customer_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    file: UploadFile | None = None,
    body: BatchPredictionRequest | None = None,
    db: Session = Depends(get_db),  # noqa: B008 - FastAPI dependency idiom
) -> BatchPredictionResponse:
    """Batch scoring via CSV upload (feature columns) or JSON records.

    Every scored record is persisted to the predictions store (PRD 12.1).
    """
    if file is not None:
        content = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}") from exc
        records = [
            CustomerFeatures(
                features=row.drop(labels=["customer_id"], errors="ignore").to_dict(),
                customer_id=(
                    int(row["customer_id"])
                    if "customer_id" in row and pd.notna(row["customer_id"])
                    else None
                ),
            )
            for _, row in df.iterrows()
        ]
    elif body is not None:
        records = body.records
    else:
        raise HTTPException(status_code=400, detail="Provide a CSV file or JSON body.")

    predictions: list[PredictionResponse] = []
    for rec in records:
        try:
            result = service.predict(rec.features, rec.customer_id)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        db.add(
            Prediction(
                customer_id=result["customer_id"],
                churn_probability=result["churn_probability"],
                churn_label=result["churn_label"],
                clv_estimate=result["clv_estimate"],
                segment=result["segment"],
                model_name=result["model_name"],
            )
        )
        predictions.append(PredictionResponse(**result))
    db.commit()
    return BatchPredictionResponse(predictions=predictions, scored=len(predictions))


@app.get("/explain/{customer_id}", response_model=ExplainResponse)
def explain(customer_id: int) -> ExplainResponse:
    """SHAP explanation + plain-language rationale for one customer."""
    try:
        return ExplainResponse(**service.explain(customer_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
