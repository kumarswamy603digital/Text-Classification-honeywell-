"""FastAPI application entrypoint.

Run locally:
    uvicorn app.main:app --reload --port 8000
Docs:
    http://localhost:8000/docs
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.ml.predict import PredictionService
from app.ml.registry import ModelRegistryError, get_registry
from app.services.analytics import AnalyticsService
from app.services.history import HistoryService

logger = logging.getLogger("app")

DESCRIPTION = """
**Text Classification Studio** - SMS spam detection powered by real fitted
scikit-learn pipelines (TF-IDF + Naive Bayes / Logistic Regression / Linear SVM).

All predictions, metrics and confusion matrices come from trained artifacts -
nothing is simulated.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging()
    registry = get_registry()
    try:
        meta = registry.reload()
        logger.info(
            "Registry loaded: %d models, selected=%s",
            len(meta.get("models", [])),
            meta.get("selected_model_id"),
        )
    except ModelRegistryError as exc:
        logger.warning("Models not ready at startup: %s", exc)

    app.state.prediction_service = PredictionService(registry)
    app.state.history_service = HistoryService()
    app.state.analytics_service = AnalyticsService()
    yield
    app.state.prediction_service = None


app = FastAPI(
    title=get_settings().app_name,
    version=get_settings().app_version,
    description=DESCRIPTION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

app.include_router(router)


# ---------- centralized, consistent error handling ----------
@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Consistent 422 shape; never echoes full user input back."""
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "detail": [
                {"field": ".".join(str(p) for p in e["loc"][1:]), "msg": e["msg"]}
                for e in exc.errors()[:10]
            ],
            "status_code": 422,
        },
    )


@app.exception_handler(ModelRegistryError)
async def registry_error_handler(request: Request, exc: ModelRegistryError) -> JSONResponse:
    logger.error("Registry error: %s", exc)
    return JSONResponse(
        status_code=503,
        content={
            "error": "model_unavailable",
            "detail": "The requested model is not available. "
                      "Please ensure models have been trained.",
            "status_code": 503,
        },
    )


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    """No stack traces or local paths leak to clients."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": "An unexpected error occurred. Please try again.",
            "status_code": 500,
        },
    )


@app.get("/")
def root() -> dict:
    return {
        "name": get_settings().app_name,
        "version": get_settings().app_version,
        "docs": "/docs",
        "api": "/api/v1",
    }
