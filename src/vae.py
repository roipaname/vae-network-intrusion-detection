"""
Variational Autoencoder trained on real normal traffic only.

The VAE learns the distribution of legitimate network traffic so it can later
generate synthetic normal samples (Phase 4) to augment the intrusion
detector's training data (Phase 5).

Run directly to train the VAE for one or both datasets:

    uv run python -m src.vae --dataset nsl_kdd
"""

import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config.settings import (
    HIDDEN_DIMS,
    KL_WEIGHT,
    LATENT_DIM,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    SYNTHETIC_SAMPLE_COUNT,
    VAE_BATCH_SIZE,
    VAE_EPOCHS,
    VAE_LEARNING_RATE,
    logger,
    vae_model_path,
)
from src.data import load_processed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAE(nn.Module):
    """A small MLP encoder/decoder VAE for tabular traffic features."""

    def __init__(self, input_dim: int, hidden_dims: list[int] = HIDDEN_DIMS, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        # Encoder: input -> hidden layers -> (mu, logvar)
        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            encoder_layers += [nn.Linear(prev_dim, h), nn.ReLU()]
            prev_dim = h
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Decoder mirrors the encoder in reverse. Final layer is linear
        # (no activation) since the inputs are standard-scaled, not [0, 1].
        decoder_layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers += [nn.Linear(prev_dim, h), nn.ReLU()]
            prev_dim = h
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar, kl_weight: float = KL_WEIGHT):
    """Reconstruction (MSE) + KL divergence, both summed per-sample then averaged over the batch."""
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum") / x.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss


def train_vae(
    dataset_name: str,
    epochs: int = VAE_EPOCHS,
    batch_size: int = VAE_BATCH_SIZE,
    lr: float = VAE_LEARNING_RATE,
    kl_weight: float = KL_WEIGHT,
) -> dict:
    """Train the VAE on the real-normal-only subset of a dataset and save it."""
    torch.manual_seed(RANDOM_SEED)

    data = load_processed(dataset_name)
    X_train_normal = data["X_train_normal"]
    input_dim = X_train_normal.shape[1]

    logger.info(
        f"[{dataset_name}] training VAE on {X_train_normal.shape[0]} normal samples "
        f"(input_dim={input_dim}, latent_dim={LATENT_DIM}, epochs={epochs})"
    )

    dataset = TensorDataset(torch.tensor(X_train_normal, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VAE(input_dim=input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_recon, epoch_kl, epoch_total, n_batches = 0.0, 0.0, 0.0, 0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss, recon_loss, kl_loss = vae_loss(recon, batch, mu, logvar, kl_weight)
            loss.backward()
            optimizer.step()

            epoch_total += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            n_batches += 1

        avg_total = epoch_total / n_batches
        avg_recon = epoch_recon / n_batches
        avg_kl = epoch_kl / n_batches
        history.append({"epoch": epoch, "loss": avg_total, "recon_loss": avg_recon, "kl_loss": avg_kl})

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            logger.info(
                f"[{dataset_name}] epoch {epoch}/{epochs} "
                f"loss={avg_total:.4f} recon={avg_recon:.4f} kl={avg_kl:.4f}"
            )

    model_path = vae_model_path(dataset_name)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dims": HIDDEN_DIMS,
            "latent_dim": LATENT_DIM,
            "history": history,
            "num_training_samples": X_train_normal.shape[0],
        },
        model_path,
    )
    logger.info(f"[{dataset_name}] saved trained VAE to {model_path}")

    return {"model": model, "history": history}


def load_vae(dataset_name: str) -> VAE:
    """Load a previously trained VAE for a dataset, ready for inference."""
    checkpoint = torch.load(vae_model_path(dataset_name), map_location=DEVICE, weights_only=False)
    model = VAE(
        input_dim=checkpoint["input_dim"],
        hidden_dims=checkpoint["hidden_dims"],
        latent_dim=checkpoint["latent_dim"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def compute_aggregate_posterior(dataset_name: str) -> dict:
    """
    Encode all real normal training data and fit a diagonal Gaussian (per
    latent dimension, mean + std) to the resulting mu vectors.

    This "aggregate posterior" is where real normal traffic actually lives
    in latent space. Phase 4 validation found it is narrower than the N(0, I)
    prior (mu std ~0.42 vs the prior's 1.0 on both datasets) -- sampling from
    the raw prior therefore draws from a wider region than real data occupies,
    which is the likely cause of the numeric-feature mismatch seen there.
    """
    model = load_vae(dataset_name)
    data = load_processed(dataset_name)
    X = torch.tensor(data["X_train_normal"], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        mu, _ = model.encode(X)
    mu = mu.cpu().numpy()
    return {"mean": mu.mean(axis=0), "std": mu.std(axis=0)}


def generate_synthetic(
    dataset_name: str, n_samples: int = SYNTHETIC_SAMPLE_COUNT, sampling: str = "prior"
) -> np.ndarray:
    """
    Sample from the VAE's latent space and decode into synthetic normal
    traffic.

    sampling="prior" (default) is the textbook VAE approach: z ~ N(0, I).
    sampling="posterior" instead samples z from a single diagonal Gaussian
    fitted to the real data's encoded mu (see compute_aggregate_posterior),
    a standard-looking correction for the prior/aggregate-posterior mismatch
    Phase 4 measured (real mu std ~0.42, narrower than the prior's 1.0).

    We tried it (see results/*_latent_sampling_comparison.json): it did not
    improve fidelity, and made it slightly worse on both datasets. Kept here
    for comparison and documented as a negative result rather than removed,
    since it's an informative finding in its own right. "prior" remains the
    default used for detector augmentation.
    """
    model = load_vae(dataset_name)

    if sampling == "prior":
        z = torch.randn(n_samples, model.latent_dim, device=DEVICE)
    elif sampling == "posterior":
        stats = compute_aggregate_posterior(dataset_name)
        mean = torch.tensor(stats["mean"], dtype=torch.float32, device=DEVICE)
        std = torch.tensor(stats["std"], dtype=torch.float32, device=DEVICE)
        eps = torch.randn(n_samples, model.latent_dim, device=DEVICE)
        z = mean + eps * std
    else:
        raise ValueError(f"Unknown sampling mode: {sampling!r} (expected 'prior' or 'posterior')")

    with torch.no_grad():
        X_synthetic = model.decode(z).cpu().numpy()
    logger.info(f"[{dataset_name}] generated {n_samples} synthetic normal samples (sampling={sampling})")
    return X_synthetic


def _synthetic_path(dataset_name: str, sampling: str):
    out_dir = PROCESSED_DATA_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if sampling == "prior" else f"_{sampling}"
    return out_dir / f"synthetic_normal{suffix}.npz"


def save_synthetic(dataset_name: str, X_synthetic: np.ndarray, sampling: str = "prior") -> None:
    """
    Save synthetic samples separately from real data (never mixed at rest).

    The prior-sampled set is saved as the plain "synthetic_normal.npz" since
    it's the one used for detector augmentation in Phase 5 (it measured at
    least as good as, and slightly better than, posterior sampling -- see
    results/*_latent_sampling_comparison.json). The posterior-sampled set is
    kept alongside it under its own name as a documented comparison.
    """
    path = _synthetic_path(dataset_name, sampling)
    np.savez_compressed(path, X=X_synthetic)
    logger.info(f"[{dataset_name}] saved {sampling}-sampled synthetic normal traffic to {path}")


def load_synthetic(dataset_name: str, sampling: str = "prior") -> np.ndarray:
    return np.load(_synthetic_path(dataset_name, sampling))["X"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the VAE on real normal traffic")
    parser.add_argument("--dataset", choices=["nsl_kdd", "unsw_nb15", "all"], default="all")
    args = parser.parse_args()

    datasets = ["nsl_kdd", "unsw_nb15"] if args.dataset == "all" else [args.dataset]
    for name in datasets:
        train_vae(name)
        for sampling in ("prior", "posterior"):
            synthetic = generate_synthetic(name, sampling=sampling)
            save_synthetic(name, synthetic, sampling=sampling)
