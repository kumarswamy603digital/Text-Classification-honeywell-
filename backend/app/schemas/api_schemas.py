"""Pydantic request/response schemas - the stable API contract."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------- classification ----------------
class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to classify")
    model_id: str | None = Field(
        None, description="Registry model id; defaults to the selected model"
    )


class ExplanationTerm(BaseModel):
    term: str
    weight: float


class ClassifyResponse(BaseModel):
    prediction_id: str
    model_id: str
    model_name: str
    model_version: str
    predicted_label: str
    label_display: str
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    confidence_type: str | None = None
    confidence_note: str | None = None
    explanation_terms: list[ExplanationTerm] = []
    latency_ms: float
    status: Literal["success"]
    timestamp: str


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=200)
    model_id: str | None = None


class BatchResponse(BaseModel):
    results: list[ClassifyResponse]
    count: int
    model_id: str


# ---------------- models ----------------
class ModelSummary(BaseModel):
    model_id: str
    name: str
    version: str
    supports_predict_proba: bool
    notes: str = ""
    is_selected: bool = False


class ModelDetail(ModelSummary):
    best_params: dict = {}
    validation_macro_f1: float | None = None
    test_metrics: dict | None = None


# ---------------- analytics ----------------
class MetricRow(BaseModel):
    model_id: str
    name: str
    accuracy: float
    precision_spam: float
    recall_spam: float
    f1_spam: float
    macro_f1: float
    weighted_f1: float
    roc_auc_spam: float | None = None
    selected: bool = False


class ModelComparisonResponse(BaseModel):
    protocol: dict
    rows: list[MetricRow]


class ConfusionMatrixResponse(BaseModel):
    model_id: str
    labels: list[str]
    matrix: list[list[int]]
    n_evaluated: int
    positive_class: str


class OverviewResponse(BaseModel):
    task: str
    dataset: dict
    dataset_stats: dict | None = None
    selected_model: ModelSummary | None = None
    selected_metrics: dict | None = None
    n_models_trained: int
    history_enabled: bool
    trained_at: str | None = None
    selection_criterion: str | None = None


# ---------------- history ----------------
class HistoryRecord(BaseModel):
    prediction_id: str
    created_at: str
    model_id: str
    predicted_label: str
    confidence: float | None = None


class HistoryListResponse(BaseModel):
    records: list[HistoryRecord]
    total: int
    enabled: bool


# ---------------- system ----------------
class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    time: str


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    models_loaded: bool
    selected_model_id: str | None = None
    detail: str | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str
    status_code: int
