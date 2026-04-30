import wfdb
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

DATASET_PATH  = Path("data/raw/mitbih")
PROCESSED_PATH = Path("data/processed")

# All 48 records in MIT-BIH
ALL_RECORDS = [
    '100','101','102','103','104','105','106','107','108','109',
    '111','112','113','114','115','116','117','118','119','121',
    '122','123','124','200','201','202','203','205','207','208',
    '209','210','212','213','214','215','217','219','220','221',
    '222','223','228','230','231','232','233','234'
]

# Normal-rhythm-only records (best for training the autoencoder)
NORMAL_RECORDS = ['100', '101', '103', '105', '112', '113', '115', '117', '121', '122']


def load_record(record_id: str, channel: int = 0):
    """Load one MIT-BIH record from local files."""
    path = str(DATASET_PATH / record_id)
    record     = wfdb.rdrecord(path)
    annotation = wfdb.rdann(path, 'atr')
    signal     = record.p_signal[:, channel]   # shape (N,)
    return signal, annotation.symbol, annotation.sample


def create_windows(signal: np.ndarray, window_size: int = 256, stride: int = 64):
    """Sliding window segmentation."""
    windows = []
    for start in range(0, len(signal) - window_size, stride):
        windows.append(signal[start : start + window_size])
    return np.array(windows)


def build_dataset(
    train_records=None,
    test_records=None,
    window_size: int = 256,
    stride: int = 64,
):
    """
    Build train / test arrays from local MIT-BIH files.

    Train: normal-only records  → autoencoder learns normal morphology
    Test:  mixed records        → contains real arrhythmias
    """
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    if train_records is None:
        train_records = NORMAL_RECORDS[:6]          # 6 normal records
    if test_records is None:
        test_records  = ['200', '201', '202', '203', '205']  # arrhythmia records

    scaler = StandardScaler()

    # ── training set (normal only) ──────────────────────────────────
    print("Loading training records (normal)...")
    train_windows = []
    for rid in train_records:
        try:
            signal, _, _ = load_record(rid)
            signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
            windows = create_windows(signal, window_size, stride)
            train_windows.append(windows)
            print(f"  ✓ record {rid}  →  {len(windows)} windows")
        except Exception as e:
            print(f"  ✗ record {rid} skipped: {e}")

    X_train = np.concatenate(train_windows, axis=0)

    # ── test set (mixed — real arrhythmias) ────────────────────────
    print("\nLoading test records (arrhythmia)...")
    test_windows = []
    for rid in test_records:
        try:
            signal, _, _ = load_record(rid)
            signal = scaler.transform(signal.reshape(-1, 1)).flatten()
            windows = create_windows(signal, window_size, stride)
            test_windows.append(windows)
            print(f"  ✓ record {rid}  →  {len(windows)} windows")
        except Exception as e:
            print(f"  ✗ record {rid} skipped: {e}")

    X_test = np.concatenate(test_windows, axis=0)

    # ── save ────────────────────────────────────────────────────────
    np.save(PROCESSED_PATH / "X_train.npy", X_train)
    np.save(PROCESSED_PATH / "X_test.npy",  X_test)

    print(f"\n✓ Saved to {PROCESSED_PATH}/")
    print(f"  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    return X_train, X_test, scaler


def load_processed():
    """Load pre-built arrays from disk."""
    X_train = np.load(PROCESSED_PATH / "X_train.npy")
    X_test  = np.load(PROCESSED_PATH / "X_test.npy")
    return X_train, X_test


def plot_sample(record_id='100', n_seconds=10, fs=360):
    """Quick sanity-check plot of one record."""
    import matplotlib.pyplot as plt

    signal, symbols, samples = load_record(record_id)
    end = n_seconds * fs

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(signal[:end], linewidth=0.7, color='steelblue')

    # Mark beat annotations
    for sym, samp in zip(symbols, samples):
        if samp < end:
            color = 'red' if sym != 'N' else 'green'
            ax.axvline(samp, color=color, alpha=0.4, linewidth=0.8)
            ax.text(samp, signal[samp] + 0.1, sym, fontsize=6, color=color)

    ax.set_title(f"MIT-BIH Record {record_id} — green=Normal, red=Arrhythmia")
    ax.set_xlabel("Sample (360 Hz)")
    ax.set_ylabel("mV")
    plt.tight_layout()
    plt.savefig(PROCESSED_PATH / f"record_{record_id}_plot.png", dpi=130)
    plt.show()
    print(f"✓ Plot saved → data/processed/record_{record_id}_plot.png")


if __name__ == "__main__":
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    # 1. Sanity check — plot record 100
    print("=== Plotting record 100 ===")
    plot_sample('100')

    # 2. Build dataset
    print("\n=== Building dataset ===")
    X_train, X_test, scaler = build_dataset()

    print("\n=== Ready for Day 2 — LSTM training ===")