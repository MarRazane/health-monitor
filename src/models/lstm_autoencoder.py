import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    classification_report,
    precision_recall_curve,
    auc
)

MODELS_PATH = Path("models")
MODELS_PATH.mkdir(exist_ok=True)


class CNNAutoencoder(nn.Module):
    def __init__(self, seq_len=256, latent_dim=8):
        super().__init__()

        self._latent_spatial = seq_len // 8

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.bottleneck_enc = nn.Linear(
            128 * self._latent_spatial,
            latent_dim
        )

        self.bottleneck_dec = nn.Linear(
            latent_dim,
            128 * self._latent_spatial
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.ConvTranspose1d(
                64, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.ConvTranspose1d(
                32, 1,
                kernel_size=4,
                stride=2,
                padding=1
            )
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.bottleneck_enc(h.flatten(1))

    def decode(self, z):
        h = self.bottleneck_dec(z)
        h = h.view(-1, 128, self._latent_spatial)
        return self.decoder(h)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


def reconstruction_loss(recon, target):
    mse = nn.functional.mse_loss(recon, target)

    diff_recon = recon[:, :, 1:] - recon[:, :, :-1]
    diff_target = target[:, :, 1:] - target[:, :, :-1]

    grad_loss = nn.functional.mse_loss(diff_recon, diff_target)

    return mse + 0.5 * grad_loss


def train_model(
    X_train,
    X_val,
    epochs=100,
    batch_size=128,
    lr=1e-3,
    max_samples=50000
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    if len(X_train) > max_samples:
        idx = np.random.choice(len(X_train), max_samples, replace=False)
        X_train = X_train[idx]

    print(f"Train windows: {len(X_train)}")
    print(f"Validation windows: {len(X_val)}")

    X_train_t = torch.tensor(
        X_train[:, np.newaxis, :],
        dtype=torch.float32
    )

    X_val_t = torch.tensor(
        X_val[:, np.newaxis, :],
        dtype=torch.float32
    )

    train_loader = DataLoader(
        TensorDataset(X_train_t),
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(X_val_t),
        batch_size=batch_size
    )

    model = CNNAutoencoder(latent_dim=8).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader)
    )

    best_val = float("inf")
    history = []
    val_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0

        for (batch,) in train_loader:
            batch = batch.to(device)

            recon = model(batch)
            loss = reconstruction_loss(recon, batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total += loss.item()

        avg_train = total / len(train_loader)
        history.append(avg_train)

        model.eval()
        val_total = 0

        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                loss = reconstruction_loss(recon, batch)
                val_total += loss.item()

        avg_val = val_total / len(val_loader)
        val_history.append(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), MODELS_PATH / "cnn_ae_best.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch}/{epochs} "
                f"train={avg_train:.5f} "
                f"val={avg_val:.5f}"
            )

    model.load_state_dict(
        torch.load(
            MODELS_PATH / "cnn_ae_best.pt",
            map_location=device
        )
    )

    return model, history, val_history


class AnomalyScorer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.scaler = RobustScaler()
        self.threshold = None
        self._train_scores = None
        self._latent_mean = None
        self._latent_std = None

    def _forward(self, X):
        self.model.eval()

        X_t = torch.tensor(
            X[:, np.newaxis, :],
            dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            z = self.model.encode(X_t)
            recon = self.model.decode(z)

        return (
            X_t.cpu().numpy(),
            recon.cpu().numpy(),
            z.cpu().numpy()
        )

    def _raw_scores(self, Xo, Xr, Z):
        mse = np.mean((Xo - Xr) ** 2, axis=(1, 2))
        max_err = np.max(np.abs(Xo - Xr), axis=(1, 2))

        grad = np.mean(
            (
                np.diff(Xo, axis=2) -
                np.diff(Xr, axis=2)
            ) ** 2,
            axis=(1, 2)
        )

        zn = (Z - self._latent_mean) / (self._latent_std + 1e-8)
        latent = np.linalg.norm(zn, axis=1)

        return np.stack([mse, max_err, grad, latent], axis=1)

    def fit(self, X_train):
        Xo, Xr, Z = self._forward(X_train)

        self._latent_mean = Z.mean(axis=0)
        self._latent_std = Z.std(axis=0)

        raw = self._raw_scores(Xo, Xr, Z)
        self.scaler.fit(raw)

        scores = self.scaler.transform(raw)

        final_scores = (
            0.4 * scores[:, 0] +
            0.3 * scores[:, 1] +
            0.2 * scores[:, 2] +
            0.1 * scores[:, 3]
        )

        self._train_scores = final_scores

    def score(self, X):
        Xo, Xr, Z = self._forward(X)

        raw = self._raw_scores(Xo, Xr, Z)
        scores = self.scaler.transform(raw)

        return (
            0.4 * scores[:, 0] +
            0.3 * scores[:, 1] +
            0.2 * scores[:, 2] +
            0.1 * scores[:, 3]
        )

    def optimize_threshold(self, X_val, y_val):
        scores = self.score(X_val)

        thresholds = np.linspace(scores.min(), scores.max(), 200)

        best_f1 = 0
        best_t = thresholds[0]

        for t in thresholds:
            pred = (scores > t).astype(int)
            f1 = f1_score(y_val, pred, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        self.threshold = best_t
        print(f"Best threshold: {best_t:.4f}")
        print(f"Validation F1: {best_f1:.4f}")


def evaluate(scorer, X_test, y_test):
    scores = scorer.score(X_test)
    pred = (scores > scorer.threshold).astype(int)

    roc = roc_auc_score(y_test, scores)
    f1 = f1_score(y_test, pred)

    prec, rec, _ = precision_recall_curve(y_test, scores)
    pr_auc = auc(rec, prec)

    print("\nEvaluation")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")
    print(f"F1     : {f1:.4f}")

    print(classification_report(
        y_test,
        pred,
        target_names=["Normal", "Anomaly"]
    ))

    return roc, f1


if __name__ == "__main__":
    import sys
    sys.path.append("src")

    from data.ecg_loader import load_processed_labeled

    X_train, X_val, y_val, X_test, y_test = load_processed_labeled()

    model, history, val_history = train_model(
        X_train,
        X_val,
        epochs=100
    )
    torch.save(model.state_dict(), MODELS_PATH / "cnn_ae.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scorer = AnomalyScorer(model, device)
    scorer.fit(X_train)
    scorer.optimize_threshold(X_val, y_val)

    evaluate(scorer, X_test, y_test)