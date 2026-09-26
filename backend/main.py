"""
FastAPI backend for the NEXUS AI dashboard.

Every endpoint either reads a file the ML pipeline actually produced
(results/*.json, data/processed/*/dataset_info.json) or runs a real
preprocessed record through a real trained detector loaded from models/.
Nothing here is a hardcoded or invented number.

NSL-KDD and UNSW-NB15 are flow-feature datasets: they have no real IP
addresses or timestamps. Endpoints that need something to display for those
fields synthesize a stable, deterministic value from the record's index --
clearly separated from genuine model output (prediction/confidence) and
genuine dataset fields (protocol, service, attack category), and never
treated as captured network data anywhere in the pipeline.

Run with:

    uv run uvicorn backend.main:app --reload --port 8000
"""

import json
import random
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import (
    ACTIVE_DETECTOR_VARIANT,
    CORS_ORIGINS,
    DATASET_DISPLAY_NAMES,
    DATASET_NAMES,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    logger,
    vae_model_path,
)
from src.data import load_processed
from src.detector import load_detector, predict_single
from src.vae import load_synthetic, load_vae

_context_cache: dict[str, dict] = {}


def _load_context(dataset_name: str) -> dict:
    """Load and cache everything an endpoint might need for a dataset: processed arrays, both detectors, and the VAE."""
    logger.info(f"Loading pipeline artifacts for '{dataset_name}' into memory")
    data = load_processed(dataset_name)
    models = {
        "baseline": load_detector(dataset_name, augmented=False),
        "augmented": load_detector(dataset_name, augmented=True),
    }
    vae = load_vae(dataset_name)
    return {"data": data, "models": models, "vae": vae}


@asynccontextmanager
async def lifespan(app: FastAPI):
    for name in DATASET_NAMES:
        try:
            _context_cache[name] = _load_context(name)
        except FileNotFoundError as exc:
            logger.warning(f"Could not preload '{name}': {exc}. Endpoints for it will 503 until the pipeline is run.")
    logger.info("Backend startup complete")
    yield
    _context_cache.clear()


app = FastAPI(
    title="NEXUS AI Backend",
    description="Serves VAE/detector research results and safe replay-based simulation for the dashboard.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def _validate_dataset(dataset: str) -> None:
    if dataset not in DATASET_NAMES:
        raise HTTPException(status_code=404, detail=f"Unknown dataset '{dataset}'. Expected one of {DATASET_NAMES}.")


def _validate_variant(variant: str) -> None:
    if variant not in ("baseline", "augmented"):
        raise HTTPException(status_code=400, detail="variant must be 'baseline' or 'augmented'")


def _get_context(dataset_name: str) -> dict:
    _validate_dataset(dataset_name)
    if dataset_name not in _context_cache:
        try:
            _context_cache[dataset_name] = _load_context(dataset_name)
        except FileNotFoundError as exc:
            logger.error(f"Missing pipeline artifacts for '{dataset_name}': {exc}")
            raise HTTPException(
                status_code=503,
                detail=f"Pipeline artifacts missing for '{dataset_name}'. Run the src pipeline first (see README).",
            )
    return _context_cache[dataset_name]


def _load_result_json(filename: str) -> dict:
    path = RESULTS_DIR / filename
    if not path.exists():
        raise HTTPException(
            status_code=503, detail=f"'{filename}' not found in results/. Run the evaluation pipeline first."
        )
    with open(path) as f:
        return json.load(f)


def _load_dataset_info(dataset_name: str) -> dict:
    path = PROCESSED_DATA_DIR / dataset_name / "dataset_info.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"dataset_info.json missing for '{dataset_name}'.")
    with open(path) as f:
        return json.load(f)


def _synthetic_ip(index: int, block: int) -> str:
    """Deterministic, display-only pseudo-IP derived from a record's index (see module docstring)."""
    b = (block * 37 + 11) % 200 + 1
    c = (index // 256) % 256
    d = index % 256
    return f"10.{b}.{c}.{d}"


def _raw_preview(dataset_name: str, index: int) -> dict:
    """Real dataset fields (protocol/service/attack category) for one test-set row."""
    preview = _get_context(dataset_name)["data"]["test_raw_preview"].iloc[index]
    protocol_col = "protocol_type" if "protocol_type" in preview else "proto"
    return {
        "protocol": preview[protocol_col],
        "service": preview["service"],
        "attack_category": preview["attack_category"],
    }


def _timestamp(offset_seconds: float = 0.0) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=offset_seconds)).isoformat()


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "datasets_loaded": list(_context_cache.keys())}


# --------------------------------------------------------------------------
# Dataset info
# --------------------------------------------------------------------------
@app.get("/api/datasets")
def list_datasets():
    return {
        "datasets": [
            {"id": name, "display_name": DATASET_DISPLAY_NAMES[name], "loaded": name in _context_cache}
            for name in DATASET_NAMES
        ]
    }


@app.get("/api/datasets/{dataset}")
def get_dataset(dataset: str):
    _validate_dataset(dataset)
    info = _load_dataset_info(dataset)
    return {**info, "display_name": DATASET_DISPLAY_NAMES[dataset]}


@app.get("/api/model/{dataset}")
def get_model_info(dataset: str):
    """VAE + detector metadata for the 'AI Model' page."""
    _get_context(dataset)  # ensures dataset is valid and loaded
    checkpoint = torch.load(vae_model_path(dataset), map_location="cpu", weights_only=False)
    history = checkpoint["history"]

    baseline_metrics = _load_result_json(f"{dataset}_baseline_metrics.json")
    augmented_metrics = _load_result_json(f"{dataset}_augmented_metrics.json")
    synthetic_validation = _load_result_json(f"{dataset}_synthetic_validation.json")
    dataset_info = _load_dataset_info(dataset)
    synthetic_samples = synthetic_validation["synthetic_samples"]
    # dataset_info's train_records is the raw count before dedup; the
    # detector actually trained on the deduplicated set (see src/data.py).
    real_training_samples = dataset_info["train_records"] - dataset_info["duplicate_rows_in_train"]

    return {
        "dataset": dataset,
        "vae": {
            "input_dim": checkpoint["input_dim"],
            "latent_dim": checkpoint["latent_dim"],
            "hidden_dims": checkpoint["hidden_dims"],
            "num_training_samples": checkpoint["num_training_samples"],
            "epochs_trained": len(history),
            "final_loss": history[-1]["loss"],
            "final_reconstruction_loss": history[-1]["recon_loss"],
            "final_kl_loss": history[-1]["kl_loss"],
            "loss_history": history,
        },
        "synthetic_validation_summary": {
            "synthetic_samples": synthetic_samples,
            "mean_ks_statistic": synthetic_validation["numeric_features"]["mean_ks_statistic"],
            "mean_categorical_tv_distance": synthetic_validation["categorical_features"]["mean_total_variation_distance"],
        },
        "detector": {
            "model_type": "random_forest",
            "baseline": {
                "training_samples": real_training_samples,
                "accuracy": baseline_metrics["accuracy"],
                "f1_score": baseline_metrics["f1_score"],
            },
            "augmented": {
                "training_samples": real_training_samples + synthetic_samples,
                "accuracy": augmented_metrics["accuracy"],
                "f1_score": augmented_metrics["f1_score"],
            },
        },
    }


@app.get("/api/latent-projection")
def get_latent_projection(dataset: str = "nsl_kdd", n_per_group: int = 200):
    """
    A genuine 2D view of the VAE's latent space: real normal traffic, VAE-
    generated synthetic normal traffic (re-encoded), and real attack traffic
    the VAE never trained on, all encoded through the same trained encoder
    and projected to 2D with PCA fit across all three groups together.

    Nothing here is decorative -- every point is a real record's actual
    encoded position, just dimensionality-reduced for plotting.
    """
    from sklearn.decomposition import PCA

    ctx = _get_context(dataset)
    data = ctx["data"]
    vae = ctx["vae"]
    rng = np.random.default_rng(0)

    def encode(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            mu, _ = vae.encode(torch.tensor(X, dtype=torch.float32))
        return mu.numpy()

    real_normal = data["X_train_normal"]
    real_normal_sample = real_normal[rng.choice(len(real_normal), size=min(n_per_group, len(real_normal)), replace=False)]

    synthetic = load_synthetic(dataset)
    synthetic_sample = synthetic[rng.choice(len(synthetic), size=min(n_per_group, len(synthetic)), replace=False)]

    attack_indices = np.where(data["y_test"] == 1)[0]
    attack_sample = data["X_test"][rng.choice(attack_indices, size=min(n_per_group, len(attack_indices)), replace=False)]

    mu_real = encode(real_normal_sample)
    mu_synthetic = encode(synthetic_sample)
    mu_attack = encode(attack_sample)

    combined = np.vstack([mu_real, mu_synthetic, mu_attack])
    projected = PCA(n_components=2, random_state=0).fit_transform(combined)

    n1, n2 = len(mu_real), len(mu_real) + len(mu_synthetic)
    return {
        "dataset": dataset,
        "real_normal": projected[:n1].tolist(),
        "synthetic_normal": projected[n1:n2].tolist(),
        "anomaly": projected[n2:].tolist(),
    }


# --------------------------------------------------------------------------
# Overview / metrics / feature importance
# --------------------------------------------------------------------------
@app.get("/api/summary")
def get_summary(dataset: str = "nsl_kdd", variant: str = ACTIVE_DETECTOR_VARIANT):
    _validate_dataset(dataset)
    _validate_variant(variant)
    metrics = _load_result_json(f"{dataset}_{variant}_metrics.json")
    info = _load_dataset_info(dataset)

    return {
        "dataset": dataset,
        "display_name": DATASET_DISPLAY_NAMES[dataset],
        "detector_variant": variant,
        "total_connections": info["test_records"],
        "normal_traffic": info["class_distribution_test"]["normal"],
        "anomalies": info["class_distribution_test"]["attack"],
        "false_positive_rate": metrics["false_positive_rate"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "roc_auc": metrics["roc_auc"],
    }


@app.get("/api/metrics")
def get_metrics(dataset: str = "nsl_kdd"):
    _validate_dataset(dataset)
    return _load_result_json(f"{dataset}_comparison.json")


@app.get("/api/feature-importance")
def get_feature_importance(dataset: str = "nsl_kdd", variant: str = ACTIVE_DETECTOR_VARIANT):
    _validate_dataset(dataset)
    _validate_variant(variant)
    return _load_result_json(f"{dataset}_feature_importance_{variant}.json")


# --------------------------------------------------------------------------
# Traffic / alerts / connections
# --------------------------------------------------------------------------
@app.get("/api/traffic")
def get_traffic(dataset: str = "nsl_kdd", limit: int = 20, variant: str = ACTIVE_DETECTOR_VARIANT):
    """A rolling window of real test-set records replayed through the detector, for the live activity view."""
    _validate_variant(variant)
    ctx = _get_context(dataset)
    data = ctx["data"]
    model = ctx["models"][variant]
    n_test = data["X_test"].shape[0]
    limit = max(1, min(limit, n_test))

    indices = random.sample(range(n_test), limit)
    events = []
    for i, idx in enumerate(indices):
        result = predict_single(model, data["X_test"][idx])
        preview = _raw_preview(dataset, idx)
        events.append({
            "id": idx,
            "timestamp": _timestamp(offset_seconds=i),
            "source_ip": _synthetic_ip(idx, 1),
            "destination_ip": _synthetic_ip(idx, 2),
            "protocol": preview["protocol"],
            "prediction": result["prediction"],
            "confidence": round(result["confidence"], 4),
        })
    return {"dataset": dataset, "variant": variant, "events": events}


@app.get("/api/alerts")
def get_alerts(dataset: str = "nsl_kdd", limit: int = 20, variant: str = ACTIVE_DETECTOR_VARIANT):
    """Records the detector flags as attacks, split into anomalous/critical by its own confidence."""
    _validate_variant(variant)
    ctx = _get_context(dataset)
    data = ctx["data"]
    model = ctx["models"][variant]

    attack_indices = np.where(data["y_test"] == 1)[0]
    if len(attack_indices) == 0:
        return {"dataset": dataset, "variant": variant, "alerts": []}

    sample_size = min(limit, len(attack_indices))
    chosen = np.random.choice(attack_indices, size=sample_size, replace=False)

    alerts = []
    for i, idx in enumerate(chosen):
        idx = int(idx)
        result = predict_single(model, data["X_test"][idx])
        if result["prediction"] != "attack":
            continue  # only report records the detector itself flagged
        preview = _raw_preview(dataset, idx)
        alerts.append({
            "id": idx,
            "timestamp": _timestamp(offset_seconds=i * 3),
            "source_ip": _synthetic_ip(idx, 3),
            "destination_ip": _synthetic_ip(idx, 4),
            "protocol": preview["protocol"],
            "attack_category": preview["attack_category"],
            "confidence": round(result["confidence"], 4),
            "severity": "critical" if result["confidence"] >= 0.85 else "anomalous",
        })

    alerts.sort(key=lambda a: a["confidence"], reverse=True)
    return {"dataset": dataset, "variant": variant, "alerts": alerts}


@app.get("/api/connections")
def get_connections(
    dataset: str = "nsl_kdd", limit: int = 50, offset: int = 0, variant: str = ACTIVE_DETECTOR_VARIANT
):
    """A stable, paginated view over the real test set (unlike /api/traffic, this doesn't resample randomly)."""
    _validate_variant(variant)
    if limit <= 0 or offset < 0:
        raise HTTPException(status_code=400, detail="limit must be positive and offset non-negative")

    ctx = _get_context(dataset)
    data = ctx["data"]
    model = ctx["models"][variant]
    n_test = data["X_test"].shape[0]
    end = min(offset + limit, n_test)

    rows = []
    for idx in range(offset, end):
        result = predict_single(model, data["X_test"][idx])
        true_label = "normal" if data["y_test"][idx] == 0 else "attack"
        preview = _raw_preview(dataset, idx)
        status = "detected" if result["prediction"] == true_label == "attack" else (
            "false_alarm" if result["prediction"] == "attack" else
            "missed" if true_label == "attack" else "normal"
        )
        rows.append({
            "id": idx,
            "timestamp": _timestamp(offset_seconds=idx),
            "source_ip": _synthetic_ip(idx, 5),
            "destination_ip": _synthetic_ip(idx, 6),
            "protocol": preview["protocol"],
            "traffic_type": preview["attack_category"],
            "prediction": result["prediction"],
            "confidence": round(result["confidence"], 4),
            "status": status,
        })

    return {"dataset": dataset, "variant": variant, "total": n_test, "offset": offset, "limit": limit, "rows": rows}


# --------------------------------------------------------------------------
# Prediction / simulation
# --------------------------------------------------------------------------
class PredictRequest(BaseModel):
    dataset: str = "nsl_kdd"
    record_index: int
    variant: str = ACTIVE_DETECTOR_VARIANT


@app.post("/api/predict")
def predict(req: PredictRequest):
    _validate_variant(req.variant)
    ctx = _get_context(req.dataset)
    data = ctx["data"]
    n_test = data["X_test"].shape[0]
    if not (0 <= req.record_index < n_test):
        raise HTTPException(status_code=400, detail=f"record_index must be between 0 and {n_test - 1}")

    model = ctx["models"][req.variant]
    result = predict_single(model, data["X_test"][req.record_index])
    true_label = "normal" if data["y_test"][req.record_index] == 0 else "attack"

    return {
        "prediction": result["prediction"],
        "confidence": round(result["confidence"], 4),
        "model": f"random_forest_{req.variant}",
        "timestamp": _timestamp(),
        "record_index": req.record_index,
        "true_label": true_label,
    }


class SimulateRequest(BaseModel):
    dataset: str = "nsl_kdd"
    traffic_type: Literal["normal", "anomaly", "attack"] = "normal"
    variant: str = ACTIVE_DETECTOR_VARIANT


@app.post("/api/simulate-attack")
def simulate_attack(req: SimulateRequest):
    """
    Replays a real test-set record through the detector -- no real attacks
    are performed. "normal" draws from real normal-labeled test records;
    "anomaly" and "attack" both draw from real attack-labeled test records,
    differentiated by the detector's own confidence (a clearer-cut case for
    "attack", a more borderline one for "anomaly") rather than by any
    fabricated meaning.
    """
    _validate_variant(req.variant)
    ctx = _get_context(req.dataset)
    data = ctx["data"]
    model = ctx["models"][req.variant]
    y_test = data["y_test"]

    pool = np.where(y_test == (0 if req.traffic_type == "normal" else 1))[0]
    if len(pool) == 0:
        raise HTTPException(status_code=500, detail="No matching test records available for this traffic type")

    idx = int(np.random.choice(pool))
    result = predict_single(model, data["X_test"][idx])

    if req.traffic_type in ("anomaly", "attack"):
        want_high_confidence = req.traffic_type == "attack"
        for _ in range(10):
            if (result["confidence"] >= 0.85) == want_high_confidence:
                break
            alt_idx = int(np.random.choice(pool))
            alt_result = predict_single(model, data["X_test"][alt_idx])
            idx, result = alt_idx, alt_result

    true_label = "normal" if y_test[idx] == 0 else "attack"
    preview = _raw_preview(req.dataset, idx)

    logger.info(
        f"[{req.dataset}] simulated '{req.traffic_type}' -> record {idx}: "
        f"prediction={result['prediction']} confidence={result['confidence']:.4f} latency={result['latency_ms']:.2f}ms"
    )

    return {
        "dataset": req.dataset,
        "traffic_type": req.traffic_type,
        "record_index": idx,
        "protocol": preview["protocol"],
        "service": preview["service"],
        "attack_category": preview["attack_category"],
        "prediction": result["prediction"],
        "confidence": round(result["confidence"], 4),
        "latency_ms": round(result["latency_ms"], 3),
        "true_label": true_label,
        "detected_correctly": result["prediction"] == true_label,
        "timestamp": _timestamp(),
    }
