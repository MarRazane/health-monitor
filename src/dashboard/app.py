import sys
import time

import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.append("src")


# App config

st.set_page_config(
    page_title="Health Monitor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"


# Sidebar

with st.sidebar:
    st.title("🫀 Health Monitor")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["ECG Monitor", "ICU Predictor", "Model Comparison", "API Status"],
    )
    st.markdown("---")
    model_choice = st.selectbox(
        "ECG Model",
        ["transformer", "cnn", "vae"],
        index=0,
    )
    st.markdown("---")
    st.caption("MIT-BIH Arrhythmia Database")
    st.caption("BiLSTM + CNN + Transformer + VAE")


# Helpers

def generate_ecg(anomalous=False, heart_rate=75):
 
    fs = 360
    t  = np.linspace(0, 256 / fs, 256)
    hr = heart_rate / 60

    ecg = (
        np.sin(2 * np.pi * hr * t)
        + 0.3 * np.sin(2 * np.pi * hr * 3 * t)
        + 0.1 * np.sin(2 * np.pi * hr * 5 * t)
    )

    beat_interval = int(fs / hr)
    for i in range(0, 256, beat_interval):
        if i + 5 < 256:
            ecg[i : i + 5] += np.array([0.2, 0.8, 2.0, 0.6, 0.1])

    ecg += np.random.normal(0, 0.05, 256)

    if anomalous:
        idx = np.random.randint(60, 180)
        ecg[idx : idx + 15] += (
            np.random.choice([-1, 1]) * np.random.uniform(1.5, 3.0)
        )
        ecg[idx : idx + 15] += np.random.normal(0, 0.5, 15)

    return ecg.tolist()


def call_api(endpoint, payload=None, method="GET"):
 
    try:
        if method == "POST":
            r = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=10)
        else:
            r = requests.get(f"{API_URL}{endpoint}", timeout=10)
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "API not running — start with: uvicorn src.api.main:app --port 8000"
    except Exception as e:
        return None, str(e)


#  ECG Monitor

if page == "ECG Monitor":
    st.title("📈 Real-Time ECG Anomaly Detection")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        heart_rate = st.slider("Simulated heart rate (BPM)", 40, 150, 75)
    with col2:
        inject_anomaly = st.checkbox("Inject anomaly", value=False)
    with col3:
        n_windows = st.slider("Windows to stream", 5, 30, 10)

    st.markdown("---")

    if st.button("Start ECG Stream", type="primary", use_container_width=True):
        ecg_placeholder     = st.empty()
        alert_placeholder   = st.empty()
        metrics_placeholder = st.empty()

        score_history   = []
        anomaly_history = []

        progress = st.progress(0)

        for i in range(n_windows):
            # inject an anomaly every fifth window when the toggle is on
            is_anomalous = inject_anomaly and (i % 5 == 4)
            signal = generate_ecg(anomalous=is_anomalous, heart_rate=heart_rate)

            result, err = call_api(
                "/predict/ecg",
                {"signal": signal, "model": model_choice},
                method="POST",
            )

            if err:
                st.error(f"API Error: {err}")
                break

            score      = result["anomaly_score"]
            is_anomaly = result["is_anomaly"]
            threshold  = result["threshold"]
            latency    = result["latency_ms"]

            score_history.append(score)
            anomaly_history.append(is_anomaly)

            # combined plot: raw ECG on top, rolling score below
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("ECG Signal", "Anomaly Score History"),
                row_heights=[0.6, 0.4],
            )

            sig_color = "red" if is_anomaly else "steelblue"
            fig.add_trace(
                go.Scatter(
                    y=signal, mode="lines",
                    line=dict(color=sig_color, width=1.2),
                    name="ECG",
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    y=score_history, mode="lines+markers",
                    line=dict(color="orange", width=2),
                    marker=dict(
                        color=["red" if a else "green" for a in anomaly_history],
                        size=8,
                    ),
                    name="Score",
                ),
                row=2, col=1,
            )
            fig.add_hline(
                y=threshold, line_dash="dash", line_color="red",
                row=2, col=1,
                annotation_text=f"Threshold = {threshold:.3f}",
            )
            fig.update_layout(
                height=500, showlegend=False,
                margin=dict(l=0, r=0, t=30, b=0),
            )
            ecg_placeholder.plotly_chart(fig, use_container_width=True)

            # per-window metrics row
            m1, m2, m3, m4 = metrics_placeholder.columns(4)
            m1.metric("Window",    f"{i + 1}/{n_windows}")
            m2.metric("Score",     f"{score:.4f}")
            m3.metric("Latency",   f"{latency:.1f} ms")
            m4.metric("Anomalies", f"{sum(anomaly_history)}/{len(anomaly_history)}")

            if is_anomaly:
                alert_placeholder.error(
                    f"ANOMALY DETECTED — Score: {score:.4f} > Threshold: {threshold:.4f} "
                    f"| Model: {model_choice.upper()}"
                )
            else:
                alert_placeholder.success(
                    f"Normal rhythm — Score: {score:.4f} | Latency: {latency:.1f} ms"
                )

            progress.progress((i + 1) / n_windows)
            time.sleep(0.3)

        st.success(
            f"Streamed {n_windows} windows — {sum(anomaly_history)} anomalies detected"
        )


#  ICU Predictor

elif page == "ICU Predictor":
    st.title("🏥 ICU Patient Deterioration Predictor")
    st.markdown("Simulate 48-hour vital signs and predict patient deterioration risk.")

    st.markdown("### Vital Signs Configuration")

    col1, col2, col3 = st.columns(3)
    with col1:
        hr_mean  = st.slider("Heart Rate (BPM)",    40, 160, 75)
        sbp_mean = st.slider("Systolic BP (mmHg)",  70, 200, 120)
    with col2:
        dbp_mean = st.slider("Diastolic BP (mmHg)", 40, 130, 80)
        spo2     = st.slider("SpO2 (%)",             80, 100, 97)
    with col3:
        rr   = st.slider("Resp Rate (br/min)",    8,  40, 16)
        temp = st.slider("Temperature (°C)",     35.0, 41.0, 37.0, 0.1)

    deteriorating = st.checkbox("Simulate deterioration in last 24 hours", value=False)

    if st.button("Predict Deterioration", type="primary", use_container_width=True):
        # build 48-hour vital sign time series with optional deterioration ramp
        hr_sig   = np.random.normal(hr_mean,   5,   48)
        sbp_sig  = np.random.normal(sbp_mean,  10,  48)
        dbp_sig  = np.random.normal(dbp_mean,  7,   48)
        spo2_sig = np.random.normal(spo2,       1,   48)
        rr_sig   = np.random.normal(rr,         2,   48)
        temp_sig = np.random.normal(temp,       0.2, 48)

        if deteriorating:
            onset = 24
            hr_sig[onset:]   += np.linspace(0, 25,  24)
            sbp_sig[onset:]  -= np.linspace(0, 25,  24)
            spo2_sig[onset:] -= np.linspace(0, 7,   24)
            rr_sig[onset:]   += np.linspace(0, 10,  24)
            temp_sig[onset:] += np.linspace(0, 1.2, 24)

        vitals = [
            [
                float(hr_sig[i]),   float(sbp_sig[i]),
                float(dbp_sig[i]),  float(spo2_sig[i]),
                float(rr_sig[i]),   float(temp_sig[i]),
            ]
            for i in range(48)
        ]

        result, err = call_api("/predict/icu", {"vitals": vitals}, method="POST")

        if err:
            st.error(f"API Error: {err}")
        else:
            prob = result["deterioration_probability"]
            risk = result["risk_level"]
            det  = result["is_deteriorating"]

            col_a, col_b, col_c = st.columns(3)
            col_a.metric(
                "Deterioration Probability",
                f"{prob * 100:.1f}%",
                delta="HIGH RISK" if det else "LOW RISK",
            )
            col_b.metric("Risk Level", risk)
            col_c.metric("Latency", f"{result['latency_ms']:.1f} ms")

            if risk == "HIGH":
                st.error("HIGH RISK — Immediate intervention recommended")
            elif risk == "MEDIUM":
                st.warning("MEDIUM RISK — Increased monitoring advised")
            else:
                st.success("LOW RISK — Patient appears stable")

            # 2x3 subplot grid, one panel per vital sign
            vital_names = ["Heart Rate", "SBP", "DBP", "SpO2", "Resp Rate", "Temperature"]
            vital_colors = ["red", "blue", "cyan", "green", "orange", "purple"]

            fig = make_subplots(rows=2, cols=3, subplot_titles=vital_names)
            for idx, (name, color) in enumerate(zip(vital_names, vital_colors)):
                r, c   = divmod(idx, 3)
                values = [v[idx] for v in vitals]
                fig.add_trace(
                    go.Scatter(
                        x=list(range(48)), y=values,
                        mode="lines",
                        line=dict(color=color, width=1.5),
                        name=name,
                    ),
                    row=r + 1, col=c + 1,
                )
                if deteriorating:
                    fig.add_vline(
                        x=24, line_dash="dash", line_color="red",
                        row=r + 1, col=c + 1,
                    )

            fig.update_layout(
                height=450, showlegend=False,
                title="48-Hour Vital Signs",
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)


# — Model Comparison

elif page == "Model Comparison":
    st.title("📊 Model Performance Comparison")

    # these numbers come from the evaluation scripts; update after retraining
    results = {
        "CNN AE":      {"roc_auc": 0.735, "f1": 0.580, "pr_auc": 0.520, "latency_ms": 8,  "params": "~180K"},
        "Transformer": {"roc_auc": 0.852, "f1": 0.631, "pr_auc": 0.730, "latency_ms": 45, "params": "~420K"},
        "VAE":         {"roc_auc": 0.819, "f1": 0.609, "pr_auc": 0.558, "latency_ms": 22, "params": "~200K"},
        "ICU BiLSTM":  {"roc_auc": 0.979, "f1": 0.942, "pr_auc": 0.978, "latency_ms": 12, "params": "~95K"},
    }

    # headline metric cards
    cols = st.columns(4)
    for i, (name, m) in enumerate(results.items()):
        with cols[i]:
            st.metric(f"**{name}**", f"AUC {m['roc_auc']:.3f}", delta=f"F1 {m['f1']:.3f}")

    st.markdown("---")

    names  = list(results.keys())
    colors = ["steelblue", "darkorange", "purple", "green"]

    col_l, col_r = st.columns(2)

    with col_l:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=names,
            y=[results[n]["roc_auc"] for n in names],
            marker_color=colors,
            text=[f"{results[n]['roc_auc']:.3f}" for n in names],
            textposition="outside",
            name="ROC-AUC",
        ))
        fig.add_trace(go.Bar(
            x=names,
            y=[results[n]["f1"] for n in names],
            marker_color=colors,
            opacity=0.6,
            text=[f"{results[n]['f1']:.3f}" for n in names],
            textposition="outside",
            name="F1",
        ))
        fig.update_layout(
            title="ROC-AUC & F1 by Model",
            barmode="group",
            yaxis=dict(range=[0, 1.1]),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # latency vs AUC — models in the top-left corner are the sweet spot
        fig2 = go.Figure()
        for name, color in zip(names, colors):
            fig2.add_trace(go.Scatter(
                x=[results[name]["latency_ms"]],
                y=[results[name]["roc_auc"]],
                mode="markers+text",
                marker=dict(size=20, color=color),
                text=[name],
                textposition="top center",
                name=name,
            ))
        fig2.update_layout(
            title="Latency vs ROC-AUC",
            xaxis_title="Inference Latency (ms)",
            yaxis_title="ROC-AUC",
            height=380,
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Detailed Metrics")
    st.dataframe(
        {
            "Model":      names,
            "ROC-AUC":    [results[n]["roc_auc"]   for n in names],
            "F1 Score":   [results[n]["f1"]         for n in names],
            "PR-AUC":     [results[n]["pr_auc"]     for n in names],
            "Latency ms": [results[n]["latency_ms"] for n in names],
            "Parameters": [results[n]["params"]     for n in names],
        },
        use_container_width=True,
    )


#  API Status

elif page == "API Status":
    st.title("API Status")

    result, err = call_api("/health")

    if err:
        st.error(f"API Offline — {err}")
        st.code("uvicorn src.api.main:app --reload --port 8000", language="bash")
    else:
        st.success("API Online")

        col1, col2 = st.columns(2)
        col1.metric("Status", result["status"].upper())
        col2.metric("Device", result["device"])

        st.markdown("### Loaded Models")
        for m in result["models_loaded"]:
            st.markdown(f"- `{m}`")

        st.markdown("### Endpoints")
        endpoints = [
            ("GET",  "/health",        "Health check"),
            ("GET",  "/models",        "List loaded models"),
            ("POST", "/predict/ecg",   "ECG anomaly detection"),
            ("POST", "/predict/batch", "Batch ECG prediction"),
            ("POST", "/predict/icu",   "ICU deterioration prediction"),
            ("GET",  "/demo/ecg",      "Demo ECG prediction"),
            ("GET",  "/docs",          "Swagger UI"),
        ]
        for method, path, desc in endpoints:
            marker = "GET" if method == "GET" else "POST"
            st.markdown(f"`{marker}` **{path}** — {desc}")

        if st.button("Run Demo Prediction"):
            demo, err = call_api(f"/demo/ecg?model={model_choice}&anomalous=false")
            if demo:
                st.json(demo)