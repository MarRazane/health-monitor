import sys
import time
import importlib.util
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.append("src")


# Model loading helpers

def _load_module(name, path):
    spec   = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODELS  = {}
SCORERS = {}
DEVICE  = torch.device("cpu")


def load_all_models():

    global MODELS, SCORERS, DEVICE

    cnn_mod   = _load_module("cnn_ae",   "src/models/cnn_autoencoder.py")
    trans_mod = _load_module("trans_ae", "src/models/transformer_ae.py")
    vae_mod   = _load_module("vae",      "src/models/vae.py")

    # --- CNN ---
    cnn_path = next(
        (p for p in ["models/cnn_ae_best.pt", "models/cnn_ae.pt"] if Path(p).exists()),
        None,
    )
    if cnn_path:
        cnn = cnn_mod.CNNAutoencoder(seq_len=256, latent_dim=8)
        cnn.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
        cnn.eval()
        MODELS["cnn"] = cnn
        print(f"  CNN loaded from {cnn_path}")

    # --- Transformer ---
    if Path("models/transformer_ae_best.pt").exists():
        trans = trans_mod.TransformerAutoencoder(seq_len=256)
        trans.load_state_dict(
            torch.load("models/transformer_ae_best.pt", map_location=DEVICE)
        )
        trans.eval()
        MODELS["transformer"] = trans
        print("  Transformer loaded")

    # --- VAE ---
    if Path("models/vae_best.pt").exists():
        vae = vae_mod.Conv1DVAE(seq_len=256, latent_dim=8)
        vae.load_state_dict(
            torch.load("models/vae_best.pt", map_location=DEVICE)
        )
        vae.eval()
        MODELS["vae"] = vae
        print("  VAE loaded")

    if not MODELS:
        print("  Warning: no model checkpoints found — API will start but predictions won't work")
        return

    from data.ecg_loader import load_processed_labeled

    X_train, X_val, y_val, X_test, y_test = load_processed_labeled()
    X_fit = X_train[:8000]  

    if "cnn" in MODELS:
        scorer = cnn_mod.AnomalyScorer(MODELS["cnn"], DEVICE)
        scorer.fit(X_fit)
        scorer.optimize_threshold(X_val, y_val)   
        SCORERS["cnn"] = scorer

    if "transformer" in MODELS:
        scorer = trans_mod.AnomalyScorer(MODELS["transformer"], DEVICE)
        scorer.fit(X_fit)
        scorer.optimize_threshold(X_fit)           
        SCORERS["transformer"] = scorer

    if "vae" in MODELS:
        scorer = vae_mod.VAEAnomalyScorer(MODELS["vae"], str(DEVICE), n_samples=5)
        scorer.fit(X_fit)
        scorer.optimize_threshold(X_fit)           
        SCORERS["vae"] = scorer

    print(f"\nReady — loaded models: {list(MODELS.keys())}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=== Loading models ===")
    load_all_models()
    yield
    print("=== Shutting down ===")


# App

app = FastAPI(
    title="Health Monitor API",
    description="Real-time ECG anomaly detection and ICU deterioration prediction",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request
class ECGRequest(BaseModel):
    signal: list[float] = Field(
        ...,
        min_length=256,
        max_length=256,
        description="256 ECG samples recorded at 360 Hz",
    )
    model: str = Field(
        "transformer",
        description="Which scorer to use: cnn | transformer | vae",
    )


class ECGResponse(BaseModel):
    model:         str
    anomaly_score: float
    is_anomaly:    bool
    threshold:     float
    confidence:    float
    latency_ms:    float


class ICURequest(BaseModel):
    vitals: list[list[float]] = Field(
        ...,
        description="48 hours x 6 vitals ordered as [HR, SBP, DBP, SpO2, RR, Temp]",
    )


class ICUResponse(BaseModel):
    deterioration_probability: float
    is_deteriorating:          bool
    risk_level:                str
    latency_ms:                float


class HealthResponse(BaseModel):
    status:        str
    models_loaded: list[str]
    device:        str


# Endpoints

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        models_loaded=list(MODELS.keys()),
        device=str(DEVICE),
    )


@app.get("/models")
async def list_models():
    return {
        "available": list(MODELS.keys()),
        "scorers":   list(SCORERS.keys()),
    }


@app.post("/predict/ecg", response_model=ECGResponse)
async def predict_ecg(req: ECGRequest):
   
    if req.model not in SCORERS:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{req.model}' is not loaded. Available: {list(SCORERS.keys())}",
        )

    signal = np.array(req.signal, dtype=np.float32)
    signal = (signal - signal.mean()) / (signal.std() + 1e-8)

    t0      = time.perf_counter()
    scorer  = SCORERS[req.model]
    score   = float(scorer.score(signal[np.newaxis, :])[0])
    latency = (time.perf_counter() - t0) * 1000

    is_anomaly = score > scorer.threshold

    # normalise distance from threshold to [0, 1] as a rough confidence
    margin     = abs(score - scorer.threshold)
    confidence = float(min(margin / (abs(scorer.threshold) + 1e-8), 1.0))

    return ECGResponse(
        model=req.model,
        anomaly_score=round(score, 6),
        is_anomaly=is_anomaly,
        threshold=round(float(scorer.threshold), 6),
        confidence=round(confidence, 4),
        latency_ms=round(latency, 2),
    )


@app.post("/predict/batch")
async def predict_batch(signals: list[list[float]], model: str = "transformer"):
    
    if model not in SCORERS:
        raise HTTPException(status_code=404, detail=f"Model '{model}' is not available")
    if len(signals) > 500:
        raise HTTPException(status_code=400, detail="Maximum 500 signals per batch request")

    X = np.array(signals, dtype=np.float32)
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

    t0      = time.perf_counter()
    scores  = SCORERS[model].score(X)
    latency = (time.perf_counter() - t0) * 1000

    threshold = SCORERS[model].threshold
    preds     = (scores > threshold).tolist()

    return {
        "model":          model,
        "n_signals":      len(signals),
        "anomaly_scores": [round(float(s), 6) for s in scores],
        "predictions":    preds,
        "anomaly_count":  sum(preds),
        "threshold":      round(float(threshold), 6),
        "latency_ms":     round(latency, 2),
    }


@app.post("/predict/icu", response_model=ICUResponse)
async def predict_icu(req: ICURequest):
   
    if len(req.vitals) != 48 or any(len(row) != 6 for row in req.vitals):
        raise HTTPException(
            status_code=400,
            detail="ICU input must be exactly 48 hours x 6 vitals",
        )

    icu_path = "models/icu_predictor_best.pt"
    if not Path(icu_path).exists():
        raise HTTPException(
            status_code=503,
            detail="ICU model checkpoint not found — train the model first",
        )

    icu_mod = _load_module("icu", "src/models/icu_predictor.py")
    model   = icu_mod.ICUDeteriorationPredictor()
    model.load_state_dict(torch.load(icu_path, map_location=DEVICE))
    model.eval()

    X   = np.array(req.vitals, dtype=np.float32)[np.newaxis]  # (1, 48, 6)
    X_t = torch.tensor(X)

    t0 = time.perf_counter()
    with torch.no_grad():
        prob = float(model(X_t).item())
    latency = (time.perf_counter() - t0) * 1000

    if prob >= 0.7:
        risk = "HIGH"
    elif prob >= 0.4:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return ICUResponse(
        deterioration_probability=round(prob, 4),
        is_deteriorating=prob >= 0.5,
        risk_level=risk,
        latency_ms=round(latency, 2),
    )


@app.get("/demo/ecg")
async def demo_ecg(model: str = "transformer", anomalous: bool = False):
    
    t   = np.linspace(0, 2, 256)
    ecg = (
        np.sin(2 * np.pi * 1.2 * t)
        + 0.3 * np.sin(2 * np.pi * 3.6 * t)
        + np.random.normal(0, 0.05, 256)
    )

    if anomalous:
        start = np.random.randint(80, 180)
        ecg[start : start + 20] += np.random.normal(0, 2.0, 20)

    return await predict_ecg(ECGRequest(signal=ecg.tolist(), model=model))