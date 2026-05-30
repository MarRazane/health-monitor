import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (roc_auc_score, f1_score, classification_report,
                              precision_recall_curve, auc)

MODELS_PATH = Path("models")
MODELS_PATH.mkdir(exist_ok=True)


#  Conv1D VAE 

class Conv1DVAE(nn.Module):
    def __init__(self, seq_len=256, latent_dim=8):
        super().__init__()
        self._latent_spatial = seq_len // 8

        self.enc_conv = nn.Sequential(
            nn.Conv1d(1, 32,  kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),  nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),  nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
        )
        flat_dim = 128 * self._latent_spatial

        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_log_var = nn.Linear(flat_dim, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, flat_dim)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),  nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),  nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(32,  1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x):
        h = self.enc_conv(x).flatten(1)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 128, self._latent_spatial)
        return self.dec_conv(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z  = self.reparameterize(mu, log_var)
        recon  = self.decode(z)
        return recon, mu, log_var


#  ELBO Loss 

def elbo_loss(recon, target, mu, log_var, beta=1.0):
    mse  = F.mse_loss(recon, target, reduction='mean')
    d_r  = recon[:, :, 1:]  - recon[:, :, :-1]
    d_t   = target[:, :, 1:] - target[:, :, :-1]
    grad_loss = F.mse_loss(d_r, d_t, reduction='mean')
    recon_loss = mse + 0.5 * grad_loss
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


#  Training 

def train_model(X_train, X_val, epochs=80, batch_size=128,
                lr=1e-3, max_samples=20000, beta=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}  beta={beta}")

    if len(X_train) > max_samples:
        idx     = np.random.choice(len(X_train), max_samples, replace=False)
        X_train = X_train[idx]
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}")

    to_t = lambda X: torch.tensor(X[:, np.newaxis, :], dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(to_t(X_train)),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(to_t(X_val)),
                              batch_size=batch_size)

    model= Conv1DVAE(seq_len=256, latent_dim=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    best_val = float('inf')
    history  = {'train': [], 'val': [], 'recon': [], 'kl': []}

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_recon, t_kl = 0, 0, 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            recon, mu, log_var = model(batch)
            loss, rl, kl       = elbo_loss(recon, batch, mu, log_var, beta)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss  += loss.item()
            t_recon += rl.item()
            t_kl    += kl.item()

        n = len(train_loader)
        history['train'].append(t_loss  / n)
        history['recon'].append(t_recon / n)
        history['kl'].append(t_kl    / n)

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for (vb,) in val_loader:
                vb = vb.to(device)
                recon, mu, log_var = model(vb)
                loss, _, _ = elbo_loss(recon, vb, mu, log_var, beta)
                v_loss += loss.item()
        avg_val = v_loss / len(val_loader)
        history['val'].append(avg_val)
        scheduler.step()

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), MODELS_PATH / "vae_best.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{epochs}  "
                  f"train={history['train'][-1]:.5f}  "
                  f"val={avg_val:.5f}  "
                  f"recon={history['recon'][-1]:.5f}  "
                  f"kl={history['kl'][-1]:.5f}"
                  f"{'  ← best' if avg_val == best_val else ''}")

    model.load_state_dict(
        torch.load(MODELS_PATH / "vae_best.pt", map_location=device)
    )
    print(f"\nBest val loss: {best_val:.6f}")
    return model, history


#  Anomaly Scorer 

class VAEAnomalyScorer:
    def __init__(self, model, device='cpu', n_samples=10):
        self.model = model
        self.device  = device
        self.n_samples  = n_samples
        self.scaler = RobustScaler()
        self.threshold = None
        self._train_scores = None

    def _score_batch(self, X: np.ndarray, batch_size=256):
        all_recon_mean, all_recon_std, all_kl = [], [], []

        for i in range(0, len(X), batch_size):
            batch = X[i: i + batch_size]
            X_t   = torch.tensor(
                batch[:, np.newaxis, :], dtype=torch.float32
            ).to(self.device)

            recon_errors, kl_scores = [], []
            self.model.train()
            with torch.no_grad():
                for _ in range(self.n_samples):
                    recon, mu, log_var = self.model(X_t)
                    recon_errors.append(
                        F.mse_loss(recon, X_t, reduction='none')
                        .mean(dim=(1, 2)).cpu().numpy()
                    )
                    kl = -0.5 * (
                        1 + log_var - mu.pow(2) - log_var.exp()
                    ).mean(dim=1).cpu().numpy()
                    kl_scores.append(kl)
            self.model.eval()

            all_recon_mean.append(np.mean(recon_errors, axis=0))
            all_recon_std.append(np.std(recon_errors,   axis=0))
            all_kl.append(np.mean(kl_scores,            axis=0))

        return (np.concatenate(all_recon_mean),
                np.concatenate(all_recon_std),
                np.concatenate(all_kl))

    def _combined(self, rm, rs, km):
        raw = np.stack([rm, rs, km], axis=1)
        s   = self.scaler.transform(raw)
        return 0.5 * s[:, 0] + 0.3 * s[:, 2] + 0.2 * s[:, 1]

    def fit(self, X_train: np.ndarray):
        print("  Fitting VAE scorer (Monte Carlo)...")
        rm, rs, km = self._score_batch(X_train)
        raw        = np.stack([rm, rs, km], axis=1)
        self.scaler.fit(raw)
        self._train_scores = self._combined(rm, rs, km)
        print(f"  ✓ mean={self._train_scores.mean():.4f}"
              f"  std={self._train_scores.std():.4f}")

    def score(self, X: np.ndarray) -> np.ndarray:
        rm, rs, km = self._score_batch(X)
        return self._combined(rm, rs, km)

    def optimize_threshold(self, X_val, y_val):
        scores = self.score(X_val)
        candidates = np.linspace(scores.min(), scores.max(), 300)
        best_f1, best_t = 0.0, candidates[0]
        for t in candidates:
            f = f1_score(y_val, (scores > t).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, float(t)
        self.threshold = best_t
        print(f"  ✓ Threshold: {best_t:.4f}  Val-F1: {best_f1:.4f}")


#  Evaluation 

def evaluate(scorer, X_test, y_test, title="Conv1D VAE"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay

    scores = scorer.score(X_test)
    pred   = (scores > scorer.threshold).astype(int)
    roc = roc_auc_score(y_test, scores)
    f1 = f1_score(y_test, pred, zero_division=0)
    pr, rc, _ = precision_recall_curve(y_test, scores)
    pr_auc = auc(rc, pr)

    print(f"\n{'='*48}")
    print(f"  {title}")
    print(f"{'='*48}")
    print(f"  Threshold : {scorer.threshold:.4f}")
    print(f"  ROC-AUC   : {roc:.4f}")
    print(f"  PR-AUC    : {pr_auc:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(classification_report(y_test, pred,
                                 target_names=['Normal', 'Anomaly'],
                                 zero_division=0))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{title} — ROC-AUC={roc:.4f}  F1={f1:.4f}",
                 fontweight='bold')

    axes[0].hist(scores[y_test==0], bins=50, color='green',
                 alpha=0.6, label='Normal',  density=True)
    axes[0].hist(scores[y_test==1], bins=50, color='red',
                 alpha=0.6, label='Anomaly', density=True)
    axes[0].axvline(scorer.threshold, color='black',
                    linestyle='--', label='Threshold')
    axes[0].set_title('Score Distributions')
    axes[0].set_xlabel('ELBO Anomaly Score')
    axes[0].legend()

    RocCurveDisplay.from_predictions(
        y_test, scores,
        name=f"VAE (AUC={roc:.3f})",
        ax=axes[1], color='purple'
    )
    axes[1].plot([0,1],[0,1],'k--', alpha=0.3)
    axes[1].set_title('ROC Curve')

    axes[2].plot(sorted(scorer._train_scores), color='steelblue',
                 alpha=0.7, linewidth=0.8)
    axes[2].axhline(scorer.threshold, color='red',
                    linestyle='--', label='Threshold')
    axes[2].set_title('Train Scores (sorted)')
    axes[2].set_xlabel('Sample index')
    axes[2].set_ylabel('Anomaly score')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('models/vae_evaluation.png', dpi=130)
    plt.show()
    print(" models/vae_evaluation.png")
    return roc, f1


#  Latent Space Visualisation 

def plot_latent_space(model, X_test, y_test, device='cpu'):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    print("  Running t-SNE on latent space...")
    model.eval()
    X_t = torch.tensor(
        X_test[:, np.newaxis, :], dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        mu, _ = model.encode(X_t)
    Z = mu.cpu().numpy()

    n    = min(3000, len(Z))
    idx  = np.random.choice(len(Z), n, replace=False)
    Z_2d = TSNE(n_components=2, random_state=42,
                perplexity=30).fit_transform(Z[idx])
    labels = y_test[idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("VAE Latent Space (t-SNE)", fontweight='bold')

    for cls, name, color in [(0, 'Normal', 'steelblue'),
                               (1, 'Anomaly', 'red')]:
        mask = labels == cls
        axes[0].scatter(Z_2d[mask, 0], Z_2d[mask, 1],
                        c=color, label=name, alpha=0.4, s=8)
    axes[0].set_title('Normal vs Anomaly in Latent Space')
    axes[0].legend()

    norms = np.linalg.norm(Z[idx], axis=1)
    axes[1].hist(norms[labels==0], bins=40, color='steelblue',
                 alpha=0.6, label='Normal',  density=True)
    axes[1].hist(norms[labels==1], bins=40, color='red',
                 alpha=0.6, label='Anomaly', density=True)
    axes[1].set_title('Latent Norm Distribution')
    axes[1].set_xlabel('||z||₂')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('models/vae_latent_space.png', dpi=130)
    plt.show()
    print(" models/vae_latent_space.png")


# Full Model Comparison 

def compare_all_models(scorers_dict, X_test, y_test):
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("All Models Comparison — Day 5", fontweight='bold')

    colors  = {'CNN AE': 'steelblue',
                'Transformer': 'darkorange',
                'VAE': 'purple'}
    summary = {}

    for name, scorer in scorers_dict.items():
        scores = scorer.score(X_test)
        pred   = (scores > scorer.threshold).astype(int)
        roc = roc_auc_score(y_test, scores)
        f1 = f1_score(y_test, pred, zero_division=0)
        summary[name] = {'roc': roc, 'f1': f1}

        RocCurveDisplay.from_predictions(
            y_test, scores,
            name=f"{name} (AUC={roc:.3f})",
            ax=axes[0], color=colors[name]
        )

    axes[0].plot([0,1],[0,1],'k--', alpha=0.3)
    axes[0].set_title('ROC Curves — All Models')

    names = list(summary.keys())
    x = np.arange(len(names))
    w = 0.35
    rocs  = [summary[n]['roc'] for n in names]
    f1s = [summary[n]['f1']  for n in names]
    clrs  = [colors[n] for n in names]

    axes[1].bar(x - w/2, rocs, w, label='ROC-AUC', color=clrs)
    axes[1].bar(x + w/2, f1s,  w, label='F1',      color=clrs, alpha=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, fontsize=9)
    axes[1].set_ylim(0, 1)
    axes[1].set_title('Metric Comparison')
    axes[1].legend()
    for i, (r, f) in enumerate(zip(rocs, f1s)):
        axes[1].text(i-w/2, r+0.01, f'{r:.3f}', ha='center', fontsize=8)
        axes[1].text(i+w/2, f+0.01, f'{f:.3f}', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('models/all_models_comparison.png', dpi=130)
    plt.show()
    print("models/all_models_comparison.png")
    return summary


# Main 

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    sys.path.append('src')

    from data.ecg_loader import load_processed_labeled

    import importlib.util

    def load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    cnn_mod = load_module("cnn_ae",   "src/models/cnn_autoencoder.py")
    trans_mod = load_module("trans_ae", "src/models/transformer_ae.py")

    CNNAutoencoder = cnn_mod.CNNAutoencoder
    CNNScorer = cnn_mod.AnomalyScorer
    TransformerAutoencoder = trans_mod.TransformerAutoencoder
    TransScorer = trans_mod.AnomalyScorer

    #Load dataset 
    print("Loading dataset ")
    X_train, X_val, y_val, X_test, y_test = load_processed_labeled()
    print(f"  X_train : {X_train.shape}")
    print(f"  X_val   : {X_val.shape}   anomaly={y_val.mean():.2%}")
    print(f"  X_test  : {X_test.shape}  anomaly={y_test.mean():.2%}")

    #  Train VAE 
    print("\nTraining Conv1D VAE ")
    vae, history = train_model(
        X_train, X_val,
        epochs=80, batch_size=128,
        max_samples=20000, beta=0.5
    )
    torch.save(vae.state_dict(), "models/vae.pt")
    print("✓ Saved → models/vae.pt")

    #  Loss curves 
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].plot(history['train'], label='Train', color='purple')
    axes[0].plot(history['val'],   label='Val',   color='darkorange')
    axes[0].set_title('ELBO Loss'); axes[0].legend()
    axes[1].plot(history['recon'], color='steelblue')
    axes[1].set_title('Reconstruction Loss')
    axes[2].plot(history['kl'],    color='red')
    axes[2].set_title('KL Divergence')
    for ax in axes:
        ax.set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig('models/vae_loss.png', dpi=130)
    plt.show()

    #  Fit VAE scorer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\nFitting VAE Scorer ")
    vae_scorer = VAEAnomalyScorer(vae, str(device), n_samples=10)
    vae_scorer.fit(X_train[:10000])

    val_size  = int(len(X_test) * 0.2)
    X_thr     = X_test[:val_size]
    y_thr     = y_test[:val_size]
    X_eval    = X_test[val_size:]
    y_eval    = y_test[val_size:]

    vae_scorer.optimize_threshold(X_thr, y_thr)

    # Evaluate VAE 
    print("\n Evaluating VAE ")
    roc_vae, f1_vae = evaluate(vae_scorer, X_eval, y_eval)

    #  Latent space
    print("\nLatent Space t-SNE")
    plot_latent_space(vae, X_test, y_test, device=str(device))

    #  Load CNN + Transformer 
    print("\nLoading all models for comparison ")
    scorers_dict = {'VAE': vae_scorer}

    # CNN — load if exists
    cnn_paths = ["models/cnn_ae_best.pt", "models/cnn_ae.pt"]
    cnn_path  = next((p for p in cnn_paths if Path(p).exists()), None)

    if cnn_path:
        cnn = CNNAutoencoder(seq_len=256, latent_dim=8)
        cnn.load_state_dict(torch.load(cnn_path, map_location=device))
        cnn_scorer = CNNScorer(cnn, device)
        cnn_scorer.fit(X_train[:10000])
        cnn_scorer.optimize_threshold(X_val, y_val)
        scorers_dict['CNN AE'] = cnn_scorer
        print(" CNN loaded")
    else:
        print("CNN model not found — skipping")

    # Transformer — load if exists
    trans_path = "models/transformer_ae_best.pt"
    if Path(trans_path).exists():
        trans = TransformerAutoencoder(seq_len=256)
        trans.load_state_dict(torch.load(trans_path, map_location=device))
        trans_scorer = TransScorer(trans, device)
        trans_scorer.fit(X_train[:10000])
        trans_scorer.optimize_threshold(X_val, y_val)
        scorers_dict['Transformer'] = trans_scorer
        print(" Transformer loaded")
    else:
        print("Transformer model not found")

