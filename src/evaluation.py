import argparse
import json

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config.settings import RESULTS_DIR, logger


def _split_feature_groups(feature_names: list[str], categorical_cols: list[str]) -> tuple[dict, list[tuple[int, str]]]:
    """Group one-hot columns back under their original column. Returns (categorical_groups, numeric_columns)."""
    categorical_groups = {col: [] for col in categorical_cols}
    numeric_columns = []

    for idx, name in enumerate(feature_names):
        if name.startswith("cat__"):
            rest = name[len("cat__"):]
            for col in categorical_cols:
                prefix = f"{col}_"
                if rest.startswith(prefix):
                    category_value = rest[len(prefix):]
                    categorical_groups[col].append((idx, category_value))
                    break
        elif name.startswith("num__"):
            numeric_columns.append((idx, name[len("num__"):]))

    return categorical_groups, numeric_columns


def _compare_numeric(real: np.ndarray, synthetic: np.ndarray, numeric_columns: list[tuple[int, str]]) -> dict:
    per_feature = {}
    ks_stats = []

    for idx, col_name in numeric_columns:
        real_col = real[:, idx]
        synth_col = synthetic[:, idx]
        ks_stat, ks_pvalue = ks_2samp(real_col, synth_col)
        ks_stats.append(ks_stat)

        per_feature[col_name] = {
            "real_mean": float(real_col.mean()),
            "real_std": float(real_col.std()),
            "synthetic_mean": float(synth_col.mean()),
            "synthetic_std": float(synth_col.std()),
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
        }

    return {
        "per_feature": per_feature,
        "mean_ks_statistic": float(np.mean(ks_stats)) if ks_stats else None,
    }


def _compare_categorical(real: np.ndarray, synthetic: np.ndarray, categorical_groups: dict) -> dict:
    """Decoder output is continuous, so take the argmax within each one-hot group."""
    per_group = {}
    tv_distances = []

    for col_name, entries in categorical_groups.items():
        if not entries:
            continue
        indices = [idx for idx, _ in entries]
        categories = [cat for _, cat in entries]

        real_block = real[:, indices]
        synth_block = synthetic[:, indices]

        real_counts = real_block.sum(axis=0)
        real_props = (real_counts / real_counts.sum()).tolist()

        synth_argmax = synth_block.argmax(axis=1)
        synth_counts = np.bincount(synth_argmax, minlength=len(categories))
        synth_props = (synth_counts / synth_counts.sum()).tolist()

        tv_distance = 0.5 * sum(abs(r - s) for r, s in zip(real_props, synth_props))
        tv_distances.append(tv_distance)

        per_group[col_name] = {
            "categories": categories,
            "real_distribution": real_props,
            "synthetic_distribution": synth_props,
            "total_variation_distance": tv_distance,
        }

    return {
        "per_group": per_group,
        "mean_total_variation_distance": float(np.mean(tv_distances)) if tv_distances else None,
    }


def _latent_stats(vae_model, real_normal: np.ndarray) -> dict:
    import torch

    with torch.no_grad():
        x = torch.tensor(real_normal, dtype=torch.float32)
        mu, logvar = vae_model.encode(x)

    return {
        "latent_dim": vae_model.latent_dim,
        "posterior_mu_mean": float(mu.mean().item()),
        "posterior_mu_std": float(mu.std().item()),
        "posterior_std_mean": float(torch.exp(0.5 * logvar).mean().item()),
        "prior_mean": 0.0,
        "prior_std": 1.0,
    }


def validate_synthetic_traffic(dataset_name: str, sampling: str = "prior") -> dict:
    from src.data import load_processed
    from src.vae import load_synthetic, load_vae

    data = load_processed(dataset_name)
    real_normal = data["X_train_normal"]
    synthetic = load_synthetic(dataset_name, sampling=sampling)
    feature_names = data["feature_names"]

    categorical_cols = list(data["preprocessor"].transformers_[0][2])

    categorical_groups, numeric_columns = _split_feature_groups(feature_names, categorical_cols)

    numeric_report = _compare_numeric(real_normal, synthetic, numeric_columns)
    categorical_report = _compare_categorical(real_normal, synthetic, categorical_groups)

    vae_model = load_vae(dataset_name)
    latent_report = _latent_stats(vae_model, real_normal)

    report = {
        "dataset": dataset_name,
        "sampling": sampling,
        "real_normal_samples": int(real_normal.shape[0]),
        "synthetic_samples": int(synthetic.shape[0]),
        "numeric_features": numeric_report,
        "categorical_features": categorical_report,
        "latent_space": latent_report,
    }

    logger.info(
        f"[{dataset_name}] synthetic validation (sampling={sampling}): mean KS statistic (numeric)="
        f"{numeric_report['mean_ks_statistic']:.4f}, mean total-variation distance (categorical)="
        f"{categorical_report['mean_total_variation_distance']:.4f}"
    )

    suffix = "" if sampling == "prior" else f"_{sampling}"
    out_path = RESULTS_DIR / f"{dataset_name}_synthetic_validation{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"[{dataset_name}] saved synthetic validation report to {out_path}")

    return report


def compare_sampling_strategies(dataset_name: str) -> dict:
    prior_path = RESULTS_DIR / f"{dataset_name}_synthetic_validation.json"
    posterior_path = RESULTS_DIR / f"{dataset_name}_synthetic_validation_posterior.json"

    with open(prior_path) as f:
        prior = json.load(f)
    with open(posterior_path) as f:
        posterior = json.load(f)

    ks_delta = posterior["numeric_features"]["mean_ks_statistic"] - prior["numeric_features"]["mean_ks_statistic"]
    tv_delta = (
        posterior["categorical_features"]["mean_total_variation_distance"]
        - prior["categorical_features"]["mean_total_variation_distance"]
    )

    summary = {
        "dataset": dataset_name,
        "prior_sampling": {
            "mean_ks_statistic": prior["numeric_features"]["mean_ks_statistic"],
            "mean_total_variation_distance": prior["categorical_features"]["mean_total_variation_distance"],
        },
        "posterior_sampling": {
            "mean_ks_statistic": posterior["numeric_features"]["mean_ks_statistic"],
            "mean_total_variation_distance": posterior["categorical_features"]["mean_total_variation_distance"],
        },
        "latent_space": posterior["latent_space"],
        "posterior_sampling_helped": bool(ks_delta < 0 and tv_delta < 0),
        "finding": (
            "Real normal traffic encodes to a latent mu with std "
            f"{posterior['latent_space']['posterior_mu_std']:.3f}, narrower than the N(0,1) prior "
            "the VAE was regularized against, which raised the hypothesis that sampling from a "
            "Gaussian fitted to that real aggregate posterior (mean + std of all real mu vectors) "
            "would generate more realistic synthetic traffic than sampling from the raw prior. "
            "Measured result: it did not help. Mean numeric-feature KS statistic went from "
            f"{prior['numeric_features']['mean_ks_statistic']:.4f} (prior) to "
            f"{posterior['numeric_features']['mean_ks_statistic']:.4f} (posterior), i.e. "
            f"{'improved' if ks_delta < 0 else 'got worse by'} {abs(ks_delta):.4f}, and mean categorical "
            f"total-variation distance went from {prior['categorical_features']['mean_total_variation_distance']:.4f} "
            f"to {posterior['categorical_features']['mean_total_variation_distance']:.4f}, "
            f"{'improved' if tv_delta < 0 else 'got worse by'} {abs(tv_delta):.4f}. "
            "Candidate explanation: this 'aggregate posterior' was fit as a single diagonal Gaussian "
            "over ALL real mu vectors at once, which collapses whatever cluster structure exists "
            "in the latent space (e.g. distinct regions for different protocols/services). During "
            "training the decoder only ever sees a per-sample reparameterized z = mu_i + eps*sigma_i, "
            "never a z drawn from this coarser global fit, so sampling from it lands the decoder in "
            "latent regions that do not correspond well to any specific real traffic mode. A proper "
            "aggregate-posterior sampler (mixture of per-sample Gaussians: pick a random training "
            "point i, then sample z ~ N(mu_i, sigma_i)) would preserve that structure and was not "
            "implemented here -- noted as a follow-up rather than pursued, since the simpler prior "
            "sampling already measured as good or better."
        ),
    }

    logger.info(f"[{dataset_name}] sampling comparison: {summary['finding']}")

    out_path = RESULTS_DIR / f"{dataset_name}_latent_sampling_comparison.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"[{dataset_name}] saved sampling comparison to {out_path}")

    return summary


def _compute_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, n_points: int = 101) -> dict:
    """Interpolate onto a fixed FPR grid so the curves can be overlaid."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    grid = np.linspace(0, 1, n_points)
    tpr_interp = np.interp(grid, fpr, tpr)
    return {"fpr": grid.tolist(), "tpr": tpr_interp.tolist()}


def compute_detector_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "false_positive_rate": float(false_positive_rate),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "classification_report": classification_report(
            y_true, y_pred, target_names=["normal", "attack"], output_dict=True, zero_division=0
        ),
        "roc_curve": _compute_roc_curve(y_true, y_proba),
    }


def evaluate_detector(dataset_name: str, augmented: bool) -> dict:
    from src.data import load_processed
    from src.detector import load_detector

    data = load_processed(dataset_name)
    model = load_detector(dataset_name, augmented=augmented)

    y_pred = model.predict(data["X_test"])
    y_proba = model.predict_proba(data["X_test"])[:, 1]  # P(attack)

    metrics = compute_detector_metrics(data["y_test"], y_pred, y_proba)
    metrics["dataset"] = dataset_name
    metrics["variant"] = "augmented" if augmented else "baseline"
    metrics["test_samples"] = int(len(data["y_test"]))

    logger.info(
        f"[{dataset_name}] {metrics['variant']}: accuracy={metrics['accuracy']:.4f} "
        f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} f1={metrics['f1_score']:.4f} "
        f"roc_auc={metrics['roc_auc']:.4f} fpr={metrics['false_positive_rate']:.4f}"
    )

    out_path = RESULTS_DIR / f"{dataset_name}_{metrics['variant']}_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"[{dataset_name}] saved {metrics['variant']} metrics to {out_path}")

    return metrics


def compare_detectors(dataset_name: str) -> dict:
    with open(RESULTS_DIR / f"{dataset_name}_baseline_metrics.json") as f:
        baseline = json.load(f)
    with open(RESULTS_DIR / f"{dataset_name}_augmented_metrics.json") as f:
        augmented = json.load(f)

    keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "false_positive_rate"]
    comparison = {
        "dataset": dataset_name,
        "baseline": {k: baseline[k] for k in keys},
        "augmented": {k: augmented[k] for k in keys},
        "deltas": {k: augmented[k] - baseline[k] for k in keys},
        "baseline_roc_curve": baseline["roc_curve"],
        "augmented_roc_curve": augmented["roc_curve"],
    }

    logger.info(f"[{dataset_name}] baseline vs augmented deltas: {comparison['deltas']}")

    out_path = RESULTS_DIR / f"{dataset_name}_comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"[{dataset_name}] saved comparison to {out_path}")

    return comparison


def compute_feature_importance(dataset_name: str, augmented: bool, top_n: int = 15) -> dict:
    """Sum one-hot importances back into their original feature."""
    from src.data import load_processed
    from src.detector import load_detector

    data = load_processed(dataset_name)
    model = load_detector(dataset_name, augmented=augmented)
    feature_names = data["feature_names"]
    categorical_cols = list(data["preprocessor"].transformers_[0][2])

    categorical_groups, numeric_columns = _split_feature_groups(feature_names, categorical_cols)
    importances = model.feature_importances_

    aggregated = {}
    for col_name, entries in categorical_groups.items():
        aggregated[col_name] = float(sum(importances[idx] for idx, _ in entries))
    for idx, col_name in numeric_columns:
        aggregated[col_name] = float(importances[idx])

    ranked = sorted(aggregated.items(), key=lambda kv: -kv[1])[:top_n]

    report = {
        "dataset": dataset_name,
        "variant": "augmented" if augmented else "baseline",
        "top_features": [{"feature": name, "importance": value} for name, value in ranked],
    }

    suffix = "_augmented" if augmented else "_baseline"
    out_path = RESULTS_DIR / f"{dataset_name}_feature_importance{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"[{dataset_name}] saved {report['variant']} feature importance to {out_path}")

    return report


def run_detector_evaluation(dataset_name: str) -> None:
    evaluate_detector(dataset_name, augmented=False)
    evaluate_detector(dataset_name, augmented=True)
    compare_detectors(dataset_name)
    compute_feature_importance(dataset_name, augmented=False)
    compute_feature_importance(dataset_name, augmented=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate synthetic traffic and/or evaluate detectors")
    parser.add_argument("--dataset", choices=["nsl_kdd", "unsw_nb15", "all"], default="all")
    parser.add_argument(
        "--stage", choices=["synthetic", "detectors", "all"], default="all",
        help="'synthetic' = Phase 4 validation, 'detectors' = Phase 6 evaluation",
    )
    args = parser.parse_args()

    datasets = ["nsl_kdd", "unsw_nb15"] if args.dataset == "all" else [args.dataset]
    for name in datasets:
        if args.stage in ("synthetic", "all"):
            validate_synthetic_traffic(name, sampling="prior")
            validate_synthetic_traffic(name, sampling="posterior")
            compare_sampling_strategies(name)
        if args.stage in ("detectors", "all"):
            run_detector_evaluation(name)
