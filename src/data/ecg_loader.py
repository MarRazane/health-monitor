import wfdb
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

DATASET_PATH = Path("data/raw/mitbih")
PROCESSED_PATH = Path("data/processed")

NORMAL_BEATS = {'N', 'L', 'R', 'e', 'j'}
ANOMALY_BEATS = {'V', 'F', 'f', 'E', '/', 'A', 'a', 'J', 'S', '!', 'Q'}

# Proper record-level split
TRAIN_RECORDS = ['100', '101', '103', '105', '112', '113', '115', '117']
VAL_RECORDS   = ['121', '122']
TEST_RECORDS  = ['200', '201', '202', '203', '205', '208']


def load_record(record_id: str, channel: int = 0):
    path = str(DATASET_PATH / record_id)
    record = wfdb.rdrecord(path)
    annotation = wfdb.rdann(path, 'atr')
    signal = record.p_signal[:, channel]
    return signal, annotation.symbol, annotation.sample


def label_window(anomaly_slice, threshold=0.10):
    return int(np.mean(anomaly_slice) >= threshold)


def process_record(record_id, scaler, window_size=256, stride=128):
    signal, symbols, beat_samples = load_record(record_id)
    signal = scaler.transform(signal.reshape(-1, 1)).flatten()

    n = len(signal)
    anomaly_mask = np.zeros(n, dtype=int)

    for sym, samp in zip(symbols, beat_samples):
        if sym in ANOMALY_BEATS:
            lo = max(0, samp - window_size // 4)
            hi = min(n, samp + window_size // 4)
            anomaly_mask[lo:hi] = 1

    windows = []
    labels = []

    for start in range(0, n - window_size, stride):
        end = start + window_size
        windows.append(signal[start:end])
        labels.append(label_window(anomaly_mask[start:end]))

    return np.array(windows), np.array(labels)


def build_labeled_dataset(window_size=256, stride=128):
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    print("Fitting scaler on TRAIN records only...")
    train_signals = []

    for rid in TRAIN_RECORDS:
        signal, _, _ = load_record(rid)
        train_signals.append(signal)

    scaler = StandardScaler()
    scaler.fit(np.concatenate(train_signals).reshape(-1, 1))

    def build_split(records, name):
        print(f"Building {name} set...")
        X_all = []
        y_all = []

        for rid in records:
            X, y = process_record(
                rid,
                scaler,
                window_size=window_size,
                stride=stride
            )
            X_all.append(X)
            y_all.append(y)
            print(f"  {rid}: {len(X)} windows")

        return np.vstack(X_all), np.concatenate(y_all)

    X_train_full, y_train_full = build_split(TRAIN_RECORDS, "train")
    X_val, y_val = build_split(VAL_RECORDS, "validation")
    X_test, y_test = build_split(TEST_RECORDS, "test")

    # train autoencoder only on normal windows
    X_train = X_train_full[y_train_full == 0]

    np.save(PROCESSED_PATH / "X_train.npy", X_train)
    np.save(PROCESSED_PATH / "X_val.npy", X_val)
    np.save(PROCESSED_PATH / "y_val.npy", y_val)
    np.save(PROCESSED_PATH / "X_test.npy", X_test)
    np.save(PROCESSED_PATH / "y_test.npy", y_test)

    print("\nDataset saved")
    print(f"X_train: {X_train.shape} (normal only)")
    print(f"X_val  : {X_val.shape}")
    print(f"Val anomaly rate : {y_val.mean():.2%}")
    print(f"X_test : {X_test.shape}")
    print(f"Test anomaly rate: {y_test.mean():.2%}")

    return X_train, X_val, y_val, X_test, y_test


def load_processed_labeled():
    X_train = np.load(PROCESSED_PATH / "X_train.npy")
    X_val = np.load(PROCESSED_PATH / "X_val.npy")
    y_val = np.load(PROCESSED_PATH / "y_val.npy")
    X_test = np.load(PROCESSED_PATH / "X_test.npy")
    y_test = np.load(PROCESSED_PATH / "y_test.npy")

    return X_train, X_val, y_val, X_test, y_test


def plot_sample(record_id='100', n_seconds=10, fs=360):
    import matplotlib.pyplot as plt

    signal, symbols, samples = load_record(record_id)
    end = n_seconds * fs

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(signal[:end], linewidth=0.7)

    for sym, samp in zip(symbols, samples):
        if samp < end:
            color = 'red' if sym in ANOMALY_BEATS else 'green'
            ax.axvline(samp, color=color, alpha=0.4, linewidth=0.8)

    ax.set_title(f"MIT-BIH Record {record_id}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    build_labeled_dataset()