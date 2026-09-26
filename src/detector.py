"""
Intrusion detector: baseline (real data only) vs VAE-augmented (real +
synthetic normal traffic) experiments.

Random Forest is the model: it's a strong, low-tuning-effort baseline for
tabular intrusion detection, trains quickly even on 100k+ rows, needs no
feature-scale assumptions beyond what preprocessing already did, and exposes
feature importances directly for the "why was this flagged" explanation
used later in the dashboard.

Both experiments use identical hyperparameters and the same untouched real
test set, so any difference in Phase 6's metrics is attributable to the
training data (real vs real+synthetic), not to the model or the evaluation.

Run directly to train both experiments for one or both datasets:

    uv run python -m src.detector --dataset nsl_kdd
"""

import argparse
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from config.settings import (
    DETECTOR_MAX_DEPTH,
    DETECTOR_N_ESTIMATORS,
    RANDOM_SEED,
    detector_model_path,
    logger,
)
from src.data import load_processed
from src.vae import load_synthetic


def build_detector() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=DETECTOR_N_ESTIMATORS,
        max_depth=DETECTOR_MAX_DEPTH,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def train_baseline(dataset_name: str) -> RandomForestClassifier:
    """Experiment A: train on real training data only."""
    data = load_processed(dataset_name)
    X_train, y_train = data["X_train"], data["y_train"]

    logger.info(
        f"[{dataset_name}] training baseline detector on {X_train.shape[0]} real samples "
        f"(normal={int((y_train == 0).sum())}, attack={int((y_train == 1).sum())})"
    )

    model = build_detector()
    model.fit(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(data["X_test"], data["y_test"])
    logger.info(f"[{dataset_name}] baseline: train_accuracy={train_accuracy:.4f} test_accuracy={test_accuracy:.4f}")

    path = detector_model_path(dataset_name, augmented=False)
    joblib.dump(model, path)
    logger.info(f"[{dataset_name}] saved baseline detector to {path}")
    return model


def train_augmented(dataset_name: str) -> RandomForestClassifier:
    """Experiment B: train on real training data + VAE-generated synthetic normal traffic."""
    data = load_processed(dataset_name)
    X_train, y_train = data["X_train"], data["y_train"]
    X_synthetic = load_synthetic(dataset_name)  # prior-sampled, see src/vae.py

    X_train_augmented = np.vstack([X_train, X_synthetic])
    y_synthetic = np.zeros(X_synthetic.shape[0], dtype=y_train.dtype)  # synthetic traffic is always "normal"
    y_train_augmented = np.concatenate([y_train, y_synthetic])

    logger.info(
        f"[{dataset_name}] training augmented detector on {X_train_augmented.shape[0]} samples "
        f"({X_train.shape[0]} real + {X_synthetic.shape[0]} synthetic normal); "
        f"normal={int((y_train_augmented == 0).sum())}, attack={int((y_train_augmented == 1).sum())}"
    )

    model = build_detector()
    model.fit(X_train_augmented, y_train_augmented)

    train_accuracy = model.score(X_train_augmented, y_train_augmented)
    test_accuracy = model.score(data["X_test"], data["y_test"])  # same untouched real test set
    logger.info(f"[{dataset_name}] augmented: train_accuracy={train_accuracy:.4f} test_accuracy={test_accuracy:.4f}")

    path = detector_model_path(dataset_name, augmented=True)
    joblib.dump(model, path)
    logger.info(f"[{dataset_name}] saved augmented detector to {path}")
    return model


def load_detector(dataset_name: str, augmented: bool = False) -> RandomForestClassifier:
    return joblib.load(detector_model_path(dataset_name, augmented=augmented))


def predict_single(model: RandomForestClassifier, x_row: np.ndarray) -> dict:
    """
    Run one already-preprocessed feature row through an already-loaded
    detector. Takes a loaded model (not a dataset name) so callers -- the
    backend in particular -- can cache models in memory and avoid a
    joblib.load() per request.
    """
    start = time.perf_counter()
    proba = model.predict_proba(x_row.reshape(1, -1))[0]
    latency_ms = (time.perf_counter() - start) * 1000

    predicted_class = int(np.argmax(proba))
    return {
        "prediction": "attack" if predicted_class == 1 else "normal",
        "confidence": float(proba[predicted_class]),
        "latency_ms": latency_ms,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline and VAE-augmented intrusion detectors")
    parser.add_argument("--dataset", choices=["nsl_kdd", "unsw_nb15", "all"], default="all")
    args = parser.parse_args()

    datasets = ["nsl_kdd", "unsw_nb15"] if args.dataset == "all" else [args.dataset]
    for name in datasets:
        train_baseline(name)
        train_augmented(name)
