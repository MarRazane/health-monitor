import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    auc,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset

MODELS_PATH = Path("models")
MODELS_PATH.mkdir(exist_ok=True)

class PositionalEncoding(nn.Module):
    

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])



class TransformerAutoencoder(nn.Module):
    
    def __init__(
        self,
        seq_len=256,
        d_model=32,
        nhead=4,
        num_layers=1,
        latent_dim=8,
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # encoder side
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc    = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)

        enc_layer    = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
            norm_first=False,  
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.to_latent = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, latent_dim)
        )
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, d_model * seq_len),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 512),
            nn.ReLU(),

            nn.Linear(512, seq_len)
        )
        self.output_proj = nn.Linear(d_model, 1)

    def encode(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)

        pooled = x.mean(dim=1)

        return self.to_latent(pooled)

    def decode(self, z):
        x = self.decoder(z)
        return x.unsqueeze(-1)          

    def forward(self, x):
        return self.decode(self.encode(x))



def reconstruction_loss(recon, target):

    mse  = nn.functional.mse_loss(recon, target)
    grad_recon  = recon[:, 1:, :]  - recon[:, :-1, :]
    grad_target = target[:, 1:, :] - target[:, :-1, :]
    grad_loss = nn.functional.mse_loss(grad_recon, grad_target)
    return mse + 0.5 * grad_loss



def train_model(X_train, X_val, epochs=40, batch_size=128, lr=5e-4, max_samples=20000):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    if len(X_train) > max_samples:
        idx = np.random.choice(len(X_train), max_samples, replace=False)
        X_train = X_train[idx]
    print(f"  Train : {len(X_train)}   Val : {len(X_val)}")

    def to_tensor(X):
        return torch.tensor(X[:, :, np.newaxis], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(to_tensor(X_train)), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(to_tensor(X_val)),   batch_size=batch_size)

    model = TransformerAutoencoder(seq_len=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val = float("inf")
    best_epoch = 0
    patience = 8
    history = []
    val_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            loss  = reconstruction_loss(model(batch), batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()

        avg_train = running / len(train_loader)
        history.append(avg_train)

        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                running_val += reconstruction_loss(model(batch), batch).item()
        avg_val = running_val / len(val_loader)
        val_history.append(avg_val)

        scheduler.step()

        if avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch

            torch.save(
            model.state_dict(),
            MODELS_PATH / "transformer_ae_best.pt"
        )

        elif epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0 or epoch == 1:
            marker = " <- best" if avg_val == best_val else ""
            print(
                f"Epoch {epoch:>3}/{epochs}  "
                f"train={avg_train:.5f}  "
                f"val={avg_val:.5f}{marker}"
            )

    model.load_state_dict(
        torch.load(MODELS_PATH / "transformer_ae_best.pt", map_location=device)
    )
    print(f"\nBest val loss: {best_val:.6f}")
    return model, history, val_history


class AnomalyScorer:
   
    def __init__(self, model, device):
        self.model  = model
        self.device = device
        self.scaler = RobustScaler()
        self.threshold     = None
        self._train_scores = None
        self._latent_mean  = None
        self._latent_std   = None

    def _forward(self, X, batch_size=256):
    
        self.model.eval()
        all_xo, all_xr, all_z = [], [], []

        for i in range(0, len(X), batch_size):
            chunk = X[i : i + batch_size]
            X_t   = torch.tensor(chunk[:, :, np.newaxis], dtype=torch.float32).to(self.device)

            with torch.no_grad():
                z  = self.model.encode(X_t)
                recon = self.model.decode(z)

            all_xo.append(X_t.cpu().numpy())
            all_xr.append(recon.cpu().numpy())
            all_z.append(z.cpu().numpy())

        return (
            np.concatenate(all_xo),
            np.concatenate(all_xr),
            np.concatenate(all_z),
        )

    def _raw_scores(self, Xo, Xr, Z):
        mse     = np.mean((Xo - Xr) ** 2, axis=(1, 2))
        max_err = np.max(np.abs(Xo - Xr), axis=(1, 2))
        grad    = np.mean((np.diff(Xo, axis=1) - np.diff(Xr, axis=1)) ** 2, axis=(1, 2))

        Z_norm = (Z - self._latent_mean) / (self._latent_std + 1e-8)
        latent = np.linalg.norm(Z_norm, axis=1)

        return np.stack([mse, max_err, grad, latent], axis=1)

    def _combine(self, scaled):
        return (
            0.4 * scaled[:, 0]
            + 0.3 * scaled[:, 1]
            + 0.2 * scaled[:, 2]
            + 0.1 * scaled[:, 3]
        )

    def fit(self, X_train):
        Xo, Xr, Z = self._forward(X_train)
        self._latent_mean = Z.mean(axis=0)
        self._latent_std  = Z.std(axis=0)
        raw = self._raw_scores(Xo, Xr, Z)
        self.scaler.fit(raw)
        self._train_scores = self._combine(self.scaler.transform(raw))

    def score(self, X):
        Xo, Xr, Z = self._forward(X)
        return self._combine(self.scaler.transform(self._raw_scores(Xo, Xr, Z)))

    def optimize_threshold(self, X_train):
        train_scores = self.score(X_train)

        self.threshold = np.percentile(
            train_scores,
            99.5
        )

        print(f"Threshold: {self.threshold:.4f}")

        return self.threshold


def evaluate(scorer, X_test, y_test, title="Transformer AE"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay

    scores = scorer.score(X_test)
    pred   = (scores > scorer.threshold).astype(int)

    roc = roc_auc_score(y_test, scores)
    f1 = f1_score(y_test, pred, zero_division=0)
    pr, rc, _ = precision_recall_curve(y_test, scores)
    pr_auc = auc(rc, pr)

    print(f"  {title}")
    print(f"  ROC-AUC : {roc:.4f}")
    print(f"  PR-AUC : {pr_auc:.4f}")
    print(f"  F1 : {f1:.4f}")
    print(classification_report(y_test, pred, target_names=["Normal", "Anomaly"], zero_division=0))

    return roc, f1, scores

def compare_models(cnn_scorer, transformer_scorer, X_test, y_test):
    
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("CNN Autoencoder vs Transformer", fontweight="bold")

    colors = {"CNN": "steelblue", "Transformer": "darkorange"}
    results = {}

    for name, scorer in [("CNN", cnn_scorer), ("Transformer", transformer_scorer)]:
        scores = scorer.score(X_test)
        pred = (scores > scorer.threshold).astype(int)
        roc = roc_auc_score(y_test, scores)
        f1  = f1_score(y_test, pred, zero_division=0)
        results[name] = {"roc": roc, "f1": f1}

        axes[0].hist(scores[y_test == 0], bins=50, alpha=0.4, color=colors[name], density=True, label=f"{name} normal")
        axes[0].hist(scores[y_test == 1], bins=50, alpha=0.4, color=colors[name], density=True, histtype="step", linestyle="--", linewidth=2, label=f"{name} anomaly")

        RocCurveDisplay.from_predictions(
            y_test, scores, name=f"{name} (AUC={roc:.3f})", ax=axes[1], color=colors[name]
        )

    axes[0].set_title("Score distributions")
    axes[0].set_xlabel("Anomaly score")
    axes[0].legend(fontsize=8)

    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[1].set_title("ROC curves")

    # grouped bar chart
    names = list(results.keys())
    x     = np.arange(len(names))
    w     = 0.35
    rocs  = [results[n]["roc"] for n in names]
    f1s   = [results[n]["f1"]  for n in names]
    clrs  = list(colors.values())

    axes[2].bar(x - w / 2, rocs, w, label="ROC-AUC", color=clrs)
    axes[2].bar(x + w / 2, f1s,  w, label="F1",      color=clrs, alpha=0.6)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Metric comparison")
    axes[2].legend()

    for i, (r, f) in enumerate(zip(rocs, f1s)):
        axes[2].text(i - w / 2, r + 0.01, f"{r:.3f}", ha="center", fontsize=9)
        axes[2].text(i + w / 2, f + 0.01, f"{f:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("models/cnn_vs_transformer.png", dpi=130)
    plt.show()
    return results


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    sys.path.append("src")

    from data.ecg_loader import load_processed_labeled
    from models.cnn_autoencoder import CNNAutoencoder, AnomalyScorer as CNNScorer

    print("Loading dataset")
    X_train, X_val, y_val, X_test, y_test = load_processed_labeled()
    print(f" X_train : {X_train.shape}")
    print(f" X_val : {X_val.shape}   anomaly={y_val.mean():.2%}")
    print(f" X_test : {X_test.shape}  anomaly={y_test.mean():.2%}")

    print("\nTraining Transformer Autoencoder")
    model, history, val_history = train_model(
        X_train, X_val, epochs=40, batch_size=256, max_samples=20000,
    )
    torch.save(model.state_dict(), "models/transformer_ae.pt")

    plt.figure(figsize=(10, 3))
    plt.plot(history, label="Train", color="steelblue",  lw=1.5)
    plt.plot(val_history, label="Val",   color="darkorange", lw=1.5)
    plt.title("Transformer Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("models/transformer_loss.png", dpi=130)
    plt.show()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nFitting Transformer Scorer")
    t_scorer = AnomalyScorer(model, device)
    t_scorer.fit(X_train)
    t_scorer.optimize_threshold(X_train)

    print("\nEvaluating Transformer")
    roc_t, f1_t, _ = evaluate(t_scorer, X_test, y_test)

    print("\nLooking for CNN checkpoint for comparison")
    cnn_path = next(
        (p for p in ["models/cnn_ae_best.pt", "models/cnn_ae.pt"] if Path(p).exists()),
        None,
    )

    if cnn_path is None:
        print("No CNN model found — skipping comparison")
        print(f"  Transformer  ROC-AUC : {roc_t:.4f}   F1 : {f1_t:.4f}")
    else:
        cnn_model = CNNAutoencoder(seq_len=256, latent_dim=8)
        cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
        cnn_model.eval()

        cnn_scorer = CNNScorer(cnn_model, device)
        cnn_scorer.fit(X_train)
        cnn_scorer.optimize_threshold(X_val, y_val)

        print("\nCNN vs Transformer")
        results = compare_models(cnn_scorer, t_scorer, X_test, y_test)
        print(f"  CNN ROC-AUC : {results['CNN']['roc']:.4f}   F1 : {results['CNN']['f1']:.4f}")
        print(f"  Transformer  ROC-AUC : {results['Transformer']['roc']:.4f}   F1 : {results['Transformer']['f1']:.4f}")
        