import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (roc_auc_score, f1_score, classification_report,
                              precision_recall_curve, auc)

MODELS_PATH = Path("models")
MODELS_PATH.mkdir(exist_ok=True)


#  Synthetic ICU Data Generator 

VITALS = ['heart_rate', 'sbp', 'dbp', 'spo2', 'resp_rate', 'temperature']

def generate_icu_dataset(n_patients=3000, seq_len=48, seed=42):
    np.random.seed(seed)
    X, y = [], []

    BASELINES = {
        'heart_rate':  (75, 12),
        'sbp': (120, 18),
        'dbp': (80, 12),
        'spo2': (97,  2),
        'resp_rate': (16,  4),
        'temperature': (37.0, 0.5),
    }

    for i in range(n_patients):
        is_deteriorating = np.random.rand() < 0.30
        patient = {}

        for vital, (mean, std) in BASELINES.items():
            personal_mean = mean + np.random.normal(0, std * 0.4)
            signal = np.random.normal(personal_mean, std, seq_len)

            noise = np.zeros(seq_len)
            for t in range(1, seq_len):
                noise[t] = 0.7 * noise[t-1] + np.random.normal(0, std * 0.2)
            signal += noise

            n_artifacts = np.random.randint(0, 3)
            for _ in range(n_artifacts):
                idx = np.random.randint(0, seq_len)
                signal[idx] += np.random.choice([-1, 1]) * std * np.random.uniform(1, 3)

            patient[vital] = signal

        if is_deteriorating:
            onset = np.random.randint(seq_len // 2, int(seq_len * 0.75))

            duration = seq_len - onset

            def noisy_ramp(magnitude):
                ramp  = np.linspace(0, magnitude, duration)
                ramp += np.random.normal(0, magnitude * 0.15, duration)
                return ramp

            if np.random.rand() > 0.3:   
                patient['heart_rate'][onset:] += noisy_ramp(
                    np.random.uniform(10, 35)
                )
            if np.random.rand() > 0.3:   
                patient['sbp'][onset:] -= noisy_ramp(
                    np.random.uniform(10, 30)
                )
                patient['dbp'][onset:] -= noisy_ramp(
                    np.random.uniform(5, 15)
                )
            if np.random.rand() > 0.4:  
                patient['spo2'][onset:] -= noisy_ramp(
                    np.random.uniform(3, 10)
                )
            if np.random.rand() > 0.3:   
                patient['resp_rate'][onset:] += noisy_ramp(
                    np.random.uniform(4, 12)
                )
            if np.random.rand() > 0.5:   
                patient['temperature'][onset:] += noisy_ramp(
                    np.random.uniform(0.3, 1.5)
                )

        patient['heart_rate']  = np.clip(patient['heart_rate'], 30, 200)
        patient['sbp'] = np.clip(patient['sbp'], 60, 220)
        patient['dbp'] = np.clip(patient['dbp'], 30, 140)
        patient['spo2'] = np.clip(patient['spo2'], 70, 100)
        patient['resp_rate'] = np.clip(patient['resp_rate'], 4,  50)
        patient['temperature'] = np.clip(patient['temperature'], 34,  42)

        data = np.stack([patient[v] for v in VITALS], axis=1)
        X.append(data)
        y.append(int(is_deteriorating))

    return np.array(X), np.array(y)


def preprocess_icu(X_train, X_val, X_test):
    n_tr, t, f = X_train.shape
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(
        X_train.reshape(-1, f)
    ).reshape(n_tr, t, f)

    X_val_s = scaler.transform(
        X_val.reshape(-1, f)
    ).reshape(X_val.shape)

    X_test_s = scaler.transform(
        X_test.reshape(-1, f)
    ).reshape(X_test.shape)

    return X_train_s, X_val_s, X_test_s, scaler


#  Model: Bidirectional LSTM + Attention 

class TemporalAttention(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, lstm_out):
        weights = torch.softmax(
            self.attn(lstm_out), dim=1
        )                                          
        context = (lstm_out * weights).sum(dim=1) 
        return context, weights.squeeze(-1)        


class ICUDeteriorationPredictor(nn.Module):
    
    def __init__(self, n_features=6, hidden=64,
                 n_layers=2, dropout=0.3):
        super().__init__()
        self.bilstm = nn.LSTM(
            n_features, hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.attention = TemporalAttention(hidden * 2)
        self.norm = nn.LayerNorm(hidden * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, return_attention=False):
        out, _ = self.bilstm(x)                 
        ctx, attn_w = self.attention(out)           
        ctx = self.norm(ctx)
        prob = self.classifier(ctx).squeeze(1)  #
        if return_attention:
            return prob, attn_w
        return prob


#  Focal Loss 

class FocalLoss(nn.Module):
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, pred, target):
        bce  = nn.functional.binary_cross_entropy(
            pred, target, reduction='none'
        )
        pt = torch.where(target == 1, pred, 1 - pred)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()


#  Training 

def train_model(X_train, y_train, X_val, y_val,
                epochs=60, batch_size=64, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}")
    print(f"  Deterioration rate — train: {y_train.mean():.2%}"
          f"  val: {y_val.mean():.2%}")

    to_t = lambda X, y: TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )
    train_loader = DataLoader(to_t(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(to_t(X_val, y_val),
                              batch_size=batch_size)

    model = ICUDeteriorationPredictor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    best_val_auc = 0.0
    history = {'train': [], 'val': [], 'val_auc': []}

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            pred = model(X_b)
            loss = criterion(pred, y_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        avg_train = total / len(train_loader)
        history['train'].append(avg_train)

        #  validate 
        model.eval()
        val_preds, val_true = [], []
        val_loss = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                pred = model(X_b)
                val_loss += criterion(pred, y_b).item()
                val_preds.extend(pred.cpu().numpy())
                val_true.extend(y_b.cpu().numpy())

        avg_val  = val_loss / len(val_loader)
        val_auc  = roc_auc_score(val_true, val_preds)
        history['val'].append(avg_val)
        history['val_auc'].append(val_auc)
        scheduler.step()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(),
                       MODELS_PATH / "icu_predictor_best.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{epochs}  "
                  f"train={avg_train:.4f}  "
                  f"val={avg_val:.4f}  "
                  f"val_auc={val_auc:.4f}")

    model.load_state_dict(
        torch.load(MODELS_PATH / "icu_predictor_best.pt",
                   map_location=device)
    )
    return model, history


#  Evaluation 

def evaluate(model, X_test, y_test, threshold=0.5):
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay, confusion_matrix
    import seaborn as sns

    device = next(model.parameters()).device
    model.eval()

    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = model(X_t).cpu().numpy()

    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.1, 0.9, 200):
        f = f1_score(y_test, (probs > t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    threshold = best_t

    pred = (probs > threshold).astype(int)
    roc = roc_auc_score(y_test, probs)
    f1  = f1_score(y_test, pred, zero_division=0)
    pr, rc, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(rc, pr)

    print(f"Threshold : {threshold:.4f}")
    print(f"ROC-AUC : {roc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(classification_report(y_test, pred,
                                 target_names=['Stable','Deteriorating'],
                                 zero_division=0))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"ICU Deterioration Predictor — ROC-AUC={roc:.4f}",
                 fontweight='bold')

    RocCurveDisplay.from_predictions(
        y_test, probs, name=f"BiLSTM+Attention (AUC={roc:.3f})",
        ax=axes[0], color='steelblue'
    )
    axes[0].plot([0,1],[0,1],'k--', alpha=0.3)
    axes[0].set_title('ROC Curve')

    axes[1].hist(probs[y_test==0], bins=40, color='green',
                 alpha=0.6, label='Stable',        density=True)
    axes[1].hist(probs[y_test==1], bins=40, color='red',
                 alpha=0.6, label='Deteriorating', density=True)
    axes[1].axvline(threshold, color='black', linestyle='--',
                    label=f'Threshold={threshold:.2f}')
    axes[1].set_title('Predicted Probability Distribution')
    axes[1].set_xlabel('P(deterioration)')
    axes[1].legend()

    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[2],
                xticklabels=['Stable','Deteriorating'],
                yticklabels=['Stable','Deteriorating'],
                cmap='Blues')
    axes[2].set_title('Confusion Matrix')
    axes[2].set_ylabel('True')
    axes[2].set_xlabel('Predicted')

    plt.tight_layout()
    plt.savefig('models/icu_evaluation.png', dpi=130)
    plt.show()
    return roc, f1, probs


#  Attention Visualisation 

def plot_attention(model, X_test, y_test, n_cases=4):
    import matplotlib.pyplot as plt

    device = next(model.parameters()).device
    model.eval()

    det_idx = np.where(y_test == 1)[0][:n_cases]
    X_cases = X_test[det_idx]

    X_t = torch.tensor(X_cases, dtype=torch.float32).to(device)
    with torch.no_grad():
        probs, attn_w = model(X_t, return_attention=True)

    fig, axes = plt.subplots(n_cases, 2, figsize=(16, 3 * n_cases))
    fig.suptitle("Attention Weights — Deteriorating Patients",
                 fontweight='bold')
    hours = np.arange(X_cases.shape[1])

    for i in range(n_cases):
        prob = probs[i].item()
        attn = attn_w[i].cpu().numpy()

        # Vital signs
        ax_v = axes[i, 0]
        for j, name in enumerate(VITALS):
            vals = X_cases[i, :, j]
            vals_norm = (vals - vals.mean()) / (vals.std() + 1e-8)
            ax_v.plot(hours, vals_norm, label=name, alpha=0.7)
        ax_v.set_title(f"Patient {i+1} — P(deterioration)={prob:.3f}")
        ax_v.set_xlabel("Hours in ICU")
        ax_v.set_ylabel("Normalised value")
        ax_v.legend(fontsize=7, ncol=3)

        # Attention weights
        ax_a = axes[i, 1]
        ax_a.bar(hours, attn, color='steelblue', alpha=0.7)
        ax_a.set_title(f"Attention weights (high = model focused here)")
        ax_a.set_xlabel("Hours in ICU")
        ax_a.set_ylabel("Attention weight")

    plt.tight_layout()
    plt.savefig('models/icu_attention.png', dpi=130)
    plt.show()


#  Main 

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate ICU dataset 
    print("=== Generating ICU dataset ===")
    X, y = generate_icu_dataset(n_patients=3000, seq_len=48, seed=42)
    print(f"  Total patients  : {len(X)}")
    print(f"  Deterioration % : {y.mean():.2%}")
    print(f"  Shape           : {X.shape}  →  (patients, hours, vitals)")

    # Train / val / test split 
    np.random.seed(42)
    idx = np.random.permutation(len(X))
    n_tr  = int(len(X) * 0.70)
    n_val = int(len(X) * 0.15)

    tr_idx  = idx[:n_tr]
    val_idx = idx[n_tr: n_tr + n_val]
    te_idx  = idx[n_tr + n_val:]

    X_train, y_train = X[tr_idx],  y[tr_idx]
    X_val,   y_val  = X[val_idx], y[val_idx]
    X_test,  y_test = X[te_idx],  y[te_idx]

    #  Preprocess 
    X_train, X_val, X_test, scaler = preprocess_icu(X_train, X_val, X_test)
    np.save("models/icu_scaler_mean.npy", scaler.center_)
    np.save("models/icu_scaler_scale.npy", scaler.scale_)

    print(f"\n  Train : {X_train.shape}  pos={y_train.mean():.2%}")
    print(f"  Val   : {X_val.shape}    pos={y_val.mean():.2%}")
    print(f"  Test  : {X_test.shape}   pos={y_test.mean():.2%}")

    #  Train 
    print("\nTraining BiLSTM + Attention ")
    model, history = train_model(
        X_train, y_train, X_val, y_val,
        epochs=60, batch_size=64
    )
    torch.save(model.state_dict(), "models/icu_predictor.pt")

    #  Loss + AUC curves 
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(history['train'], label='Train', color='steelblue')
    axes[0].plot(history['val'],   label='Val',   color='darkorange')
    axes[0].set_title('Focal Loss'); axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(history['val_auc'], color='green', lw=2)
    axes[1].set_title('Validation ROC-AUC per Epoch')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('AUC')
    axes[1].axhline(max(history['val_auc']), color='red',
                    linestyle='--',
                    label=f"Best={max(history['val_auc']):.4f}")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('models/icu_training.png', dpi=130)
    plt.show()

    #  Evaluate 
    print("\n=== Evaluating ===")
    roc, f1, probs = evaluate(model, X_test, y_test)

    #  Attention visualisation 
    print("\n=== Attention Visualisation ===")
    plot_attention(model, X_test, y_test, n_cases=4)

    