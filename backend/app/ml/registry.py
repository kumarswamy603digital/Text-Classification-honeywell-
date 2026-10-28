"""Model registry: discovers fitted artifacts and serves cached pipelines.

The API layer selects models through this registry, so prediction logic is
never duplicated per endpoint.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class ModelRegistryError(RuntimeError):
    """Raised when a model id is unknown or an artifact is unavailable."""


class ModelRegistry:
    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.metadata_path = self.artifacts_dir / "model_metadata.json"
        self._metadata: dict | None = None
        self._pipelines: dict[str, object] = {}
        self._lock = threading.Lock()

    # ---------- metadata ----------
    @property
    def metadata(self) -> dict:
        if self._metadata is None:
            raise ModelRegistryError(
                "No trained models found. Run `python scripts/train_models.py` first."
            )
        return self._metadata

    def is_ready(self) -> bool:
        try:
            self.reload()
            sel = self.selected_model_id
            return sel is not None and self.artifact_path(sel).exists()
        except ModelRegistryError:
            return False

    def reload(self) -> dict:
        """Load (or reload) metadata from disk. Called at startup and can be
        re-invoked after retraining without restarting the API."""
        if not self.metadata_path.exists():
            raise ModelRegistryError(
                f"Registry metadata missing at {self.metadata_path.name}. "
                "Train models first: python scripts/train_models.py"
            )
        try:
            meta = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ModelRegistryError(f"Corrupt registry metadata: {exc}") from exc
        with self._lock:
            self._metadata = meta
            self._pipelines.clear()  # force lazy re-load from new metadata
        return meta

    # ---------- models ----------
    @property
    def selected_model_id(self) -> str | None:
        return self.metadata.get("selected_model_id")

    def available_models(self) -> list[dict]:
        return self.metadata.get("models", [])

    def get_model_info(self, model_id: str) -> dict:
        for m in self.available_models():
            if m["model_id"] == model_id:
                return m
        raise ModelRegistryError(
            f"Unknown model id '{model_id}'. Available: "
            + ", ".join(self.metadata.get("available_model_ids", []))
        )

    def artifact_path(self, model_id: str) -> Path:
        self.get_model_info(model_id)  # validates id
        return self.artifacts_dir / f"{model_id}.joblib"

    def get_pipeline(self, model_id: str | None = None):
        """Return the fitted pipeline (lazy-loaded, thread-safe, cached)."""
        mid = model_id or self.selected_model_id
        if mid is None:
            raise ModelRegistryError("No model selected in the registry")
        with self._lock:
            if mid in self._pipelines:
                return self._pipelines[mid]
        path = self.artifact_path(mid)
        if not path.exists():
            raise ModelRegistryError(
                f"Model artifact '{mid}.joblib' is missing from the artifacts directory."
            )
        import joblib

        try:
            pipe = joblib.load(path)
        except Exception as exc:  # noqa: BLE001
            raise ModelRegistryError(f"Failed to load artifact for '{mid}': {exc}") from exc
        with self._lock:
            self._pipelines[mid] = pipe
        logger.info("Loaded pipeline '%s' into registry cache", mid)
        return pipe


_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry(get_settings().artifacts_path)
    return _registry
