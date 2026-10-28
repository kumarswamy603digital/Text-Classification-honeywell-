"""API v1 routes. One router per domain, composed in app.main."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.core.config import Settings, get_settings
from app.ml.predict import EmptyTextInputError, PredictionService
from app.ml.registry import ModelRegistry, ModelRegistryError, get_registry
from app.schemas.api_schemas import (
    BatchRequest,
    BatchResponse,
    ClassifyRequest,
    ClassifyResponse,
    ConfusionMatrixResponse,
    HealthResponse,
    HistoryListResponse,
    ModelComparisonResponse,
    ModelDetail,
    ModelSummary,
    OverviewResponse,
    ReadyResponse,
)
from app.services.analytics import AnalyticsService
from app.services.history import HistoryService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


# ---------- dependency providers ----------
def get_prediction_service(request: Request) -> PredictionService:
    return request.app.state.prediction_service


def get_history(request: Request) -> HistoryService:
    return request.app.state.history_service


def get_analytics(request: Request) -> AnalyticsService:
    return request.app.state.analytics_service


def _registry() -> ModelRegistry:
    return get_registry()


# ---------- system ----------
@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        time=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse:
    reg = _registry()
    try:
        reg.reload()
        sel = reg.selected_model_id
        loaded = sel is not None and reg.artifact_path(sel).exists()
        return ReadyResponse(
            status="ready" if loaded else "not_ready",
            models_loaded=loaded,
            selected_model_id=sel,
            detail=None if loaded else "Selected model artifact is not available.",
        )
    except ModelRegistryError as exc:
        return ReadyResponse(status="not_ready", models_loaded=False, detail=str(exc))


# ---------- classification ----------
@router.post("/classify", response_model=ClassifyResponse)
def classify(
    body: ClassifyRequest,
    svc: PredictionService = Depends(get_prediction_service),
    history: HistoryService = Depends(get_history),
) -> ClassifyResponse:
    try:
        results = svc.predict([body.text], body.model_id)
    except EmptyTextInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelRegistryError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    res = results[0]
    history.record({**res, "input": body.text})
    return ClassifyResponse(**res)


@router.post("/classify/batch", response_model=BatchResponse)
def classify_batch(
    body: BatchRequest,
    svc: PredictionService = Depends(get_prediction_service),
    history: HistoryService = Depends(get_history),
    settings: Settings = Depends(get_settings),
) -> BatchResponse:
    if len(body.texts) > settings.max_batch_size:
        raise HTTPException(
            status_code=413,
            detail=f"Batch size {len(body.texts)} exceeds limit of {settings.max_batch_size}",
        )
    try:
        results = svc.predict(body.texts, body.model_id)
    except EmptyTextInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelRegistryError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    for r in results:
        history.record(r)
    return BatchResponse(
        results=[ClassifyResponse(**r) for r in results],
        count=len(results),
        model_id=results[0]["model_id"],
    )


# ---------- models ----------
@router.get("/models", response_model=list[ModelSummary])
def list_models() -> list[ModelSummary]:
    try:
        reg = _registry()
        reg.reload()
    except ModelRegistryError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    sel = reg.selected_model_id
    return [
        ModelSummary(
            model_id=m["model_id"],
            name=m["name"],
            version=m.get("version", "1.0.0"),
            supports_predict_proba=m["supports_predict_proba"],
            notes=m.get("notes", ""),
            is_selected=m["model_id"] == sel,
        )
        for m in reg.available_models()
    ]


@router.get("/models/selected", response_model=ModelSummary)
def selected_model() -> ModelSummary:
    try:
        reg = _registry()
        reg.reload()
        info = reg.get_model_info(reg.selected_model_id)
    except ModelRegistryError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ModelSummary(
        model_id=info["model_id"],
        name=info["name"],
        version=info.get("version", "1.0.0"),
        supports_predict_proba=info["supports_predict_proba"],
        notes=info.get("notes", ""),
        is_selected=True,
    )


@router.get("/models/{model_id}", response_model=ModelDetail)
def model_detail(model_id: str) -> ModelDetail:
    try:
        reg = _registry()
        reg.reload()
        info = reg.get_model_info(model_id)
    except ModelRegistryError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ModelDetail(
        model_id=info["model_id"],
        name=info["name"],
        version=info.get("version", "1.0.0"),
        supports_predict_proba=info["supports_predict_proba"],
        notes=info.get("notes", ""),
        is_selected=info["model_id"] == reg.selected_model_id,
        best_params=info.get("best_params", {}),
        validation_macro_f1=info.get("validation_macro_f1"),
        test_metrics=info.get("test_metrics"),
    )


# ---------- analytics ----------
@router.get("/analytics/overview", response_model=OverviewResponse)
def analytics_overview(
    svc: AnalyticsService = Depends(get_analytics),
    history: HistoryService = Depends(get_history),
):
    data = svc.overview()
    if data is None:
        raise HTTPException(
            status_code=503,
            detail="No trained models available yet. Run scripts/train_models.py.",
        )
    data["history_enabled"] = history.enabled
    return OverviewResponse(**data)


@router.get("/analytics/model-comparison", response_model=ModelComparisonResponse)
def analytics_comparison(svc: AnalyticsService = Depends(get_analytics)):
    data = svc.model_comparison()
    if data is None:
        raise HTTPException(status_code=503, detail="No trained models available yet.")
    return ModelComparisonResponse(**data)


@router.get("/analytics/confusion-matrix", response_model=ConfusionMatrixResponse)
def analytics_confusion(
    svc: AnalyticsService = Depends(get_analytics),
    model_id: str | None = Query(None),
):
    data = svc.confusion_matrix(model_id)
    if data is None:
        raise HTTPException(
            status_code=503 if svc.metadata is None else 404,
            detail="Confusion matrix unavailable - models not trained or id unknown.",
        )
    return ConfusionMatrixResponse(**data)


@router.get("/analytics/dataset-stats")
def analytics_dataset_stats(svc: AnalyticsService = Depends(get_analytics)) -> dict:
    stats = svc.dataset_stats
    if stats is None:
        raise HTTPException(status_code=503, detail="Dataset stats not generated yet.")
    extra = svc.eda_extra or {}
    return {**stats, "top_terms": extra}


# ---------- history ----------
@router.get("/history", response_model=HistoryListResponse)
def list_history(
    history: HistoryService = Depends(get_history),
    limit: int = Query(50, ge=1, le=200),
):
    if not history.enabled:
        return HistoryListResponse(records=[], total=0, enabled=False)
    return HistoryListResponse(
        records=history.list_recent(limit), total=history.total(), enabled=True
    )


@router.get("/history/{prediction_id}")
def get_history_item(prediction_id: str, history: HistoryService = Depends(get_history)):
    rec = history.get(prediction_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return rec


@router.delete("/history/{prediction_id}", status_code=204)
def delete_history_item(prediction_id: str, history: HistoryService = Depends(get_history)):
    if not history.enabled:
        raise HTTPException(status_code=404, detail="History is disabled")
    if not history.delete(prediction_id):
        raise HTTPException(status_code=404, detail="Prediction not found")
