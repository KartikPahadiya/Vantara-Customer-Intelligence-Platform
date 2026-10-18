"""Pydantic request/response models (PRD Section 12.1: input validation)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Feature values for a single customer (subset allowed — missing
    features are filled with 0, matching training-time ``fillna`` rules)."""

    features: dict[str, Any] = Field(
        ..., description="Feature name -> value map (see /model-metadata)."
    )
    customer_id: int | None = Field(None, description="Optional customer identifier.")


class PredictionResponse(BaseModel):
    """Single-customer churn + CLV + segment response."""

    customer_id: int | None
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn_label: int = Field(..., description="1 if churn_probability >= 0.5")
    clv_estimate: float
    segment: int
    model_name: str


class BatchPredictionRequest(BaseModel):
    """Batch scoring of already-featured customer records."""

    records: list[CustomerFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    scored: int


class ExplainResponse(BaseModel):
    """Explainability payload for one customer."""

    customer_id: int | None
    churn_probability: float
    top_contributors: list[dict[str, Any]]
    plain_language: str


class ModelMetadata(BaseModel):
    """Model card for the production churn model."""

    model_name: str
    feature_columns: list[str]
    metrics: list[dict[str, Any]]
    training_data_rows: int | None = None
