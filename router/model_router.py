import os
import time
import logging
from typing import Any, Dict, Tuple

import joblib
import pandas as pd

from s3_artifacts import download_file

log = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

MODEL_PATH = os.getenv("ROUTER_MODEL_PATH", "/data/router_model/router_model.joblib")
RELOAD_EVERY_S = int(os.getenv("ROUTER_MODEL_RELOAD_S", "30"))

S3_FETCH_ENABLED = os.getenv("ROUTER_MODEL_S3_FETCH", "0") == "1"
S3_BUCKET = os.getenv("S3_BUCKET", "").strip()
S3_KEY = os.getenv("S3_KEY", "").strip()

_state: Dict[str, Any] = {
    "loaded_at": 0.0,
    "obj": None,
    "model_path": None,
    "last_fetch_ok": None,
    "last_fetch_error": None,
}


def _norm_dataset(ds: str) -> str:
    return (ds or "").strip().lower()


def _safe_int(x, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _fetch_from_s3_if_configured() -> None:
    if not S3_FETCH_ENABLED:
        return
    if not S3_BUCKET or not S3_KEY:
        _state["last_fetch_ok"] = False
        _state["last_fetch_error"] = "S3_BUCKET/S3_KEY not set"
        return

    try:
        download_file(MODEL_PATH, S3_BUCKET, S3_KEY)
        _state["last_fetch_ok"] = True
        _state["last_fetch_error"] = None
    except Exception as e:
        _state["last_fetch_ok"] = False
        _state["last_fetch_error"] = repr(e)
        log.warning("S3 fetch failed: %s", e)


def _load_local_model() -> None:
    if os.path.exists(MODEL_PATH):
        try:
            _state["obj"] = joblib.load(MODEL_PATH)
            _state["model_path"] = MODEL_PATH
        except Exception as e:
            log.warning("Failed to load local model: %s", e)


def _maybe_reload() -> None:
    now = time.time()
    if _state["obj"] is None or (now - _state["loaded_at"]) > RELOAD_EVERY_S:
        _fetch_from_s3_if_configured()
        _load_local_model()
        _state["loaded_at"] = now


def _heuristic_route(ds: str, default_variant: str) -> Tuple[str, Dict[str, Any]]:
    if "hotpot" in ds:
        return "llama3", {"route": "heuristic", "rule": "dataset_hotpot"}
    if "narrative" in ds:
        return "mistral", {"route": "heuristic", "rule": "dataset_narrative"}
    return default_variant, {"route": "default", "rule": "no_model"}


def pick_backend(
    dataset: str,
    question: str,
    context: str,
    retrieval_k: int,
    default_variant: str,
) -> Tuple[str, Dict[str, Any]]:
    _maybe_reload()

    ds = _norm_dataset(dataset)
    rk = _safe_int(retrieval_k, _safe_int(os.getenv("TOP_K", "0"), 0))

    if not _state["obj"]:
        chosen, info = _heuristic_route(ds, default_variant)
        info.update(
            {
                "s3_fetch_enabled": S3_FETCH_ENABLED,
                "s3_bucket": S3_BUCKET or None,
                "s3_key": S3_KEY or None,
                "last_fetch_ok": _state.get("last_fetch_ok"),
                "last_fetch_error": _state.get("last_fetch_error"),
            }
        )
        return chosen, info

    obj = _state["obj"]
    if not isinstance(obj, dict):
        obj = {"router_type": "ml", "ml_model": obj}

    router_type = str(obj.get("router_type", "ml")).strip().lower()

    X = {
        "dataset": ds,
        "prompt_chars": len(question or ""),
        "context_chars": len(context or ""),
        "retrieval_k": rk,
    }

    try:
        if router_type == "dataset":
            mapping = obj.get("dataset_router", {}) or {}
            chosen = mapping.get(ds, default_variant)
            return str(chosen), {
                "route": "dataset",
                "router_type": router_type,
                "model_path": _state.get("model_path"),
                "dataset": ds,
                "chosen": chosen,
                "last_fetch_ok": _state.get("last_fetch_ok"),
            }

        if router_type == "ml":
            pipe = obj.get("ml_model")
            if pipe is None:
                raise RuntimeError("ml_model missing from artifact")

            pred = pipe.predict(pd.DataFrame([X]))[0]
            return str(pred), {
                "route": "ml",
                "router_type": router_type,
                "model_path": _state.get("model_path"),
                "features": X,
                "last_fetch_ok": _state.get("last_fetch_ok"),
            }

        raise RuntimeError(f"Unknown router_type: {router_type}")

    except Exception as e:
        log.warning("Routing failed, falling back. Error: %s", e)
        chosen, info = _heuristic_route(ds, default_variant)
        info.update({
            "route": "fallback",
            "router_type": router_type,
            "error": repr(e),
            "features": X,
            "model_path": _state.get("model_path"),
        })
        return chosen, info
