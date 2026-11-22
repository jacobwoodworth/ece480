"""
dataset.py  -  loads the CSV files written by motor_logger.py

Reads every CSV file and returns:
    X       feature table (one row per 1-second epoch)
    y       fault label for each row (stable / overload / airflow_blocked / misalignment)
    groups  run ID for each row
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Features the random forest uses
FEATURES = ["irms_A", "vrms_V", "power_W", "pf", "crest", "thd", "sideband", "ripple", "freq_Hz"]


def load_dataset(data_dir=DATA_DIR, verbose=True):
    # Find every CSV inside the label folders
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*/*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in the label folders under {data_dir}")

    # Read each run, skipping empty or unrelated files
    frames = []
    for f in files:
        df = pd.read_csv(f)
        if df.empty or "label" not in df.columns:
            if verbose:
                print(f"  skipping {f.relative_to(data_dir)} (empty or not a logger file)")
            continue
        frames.append(df)

    # Stack all runs into one table
    if not frames:
        raise ValueError(f"All CSV files under {data_dir} are empty. Record some runs first.")
    df = pd.concat(frames, ignore_index=True)

    # Remove rows with missing values or a clipped sensor signal
    before = len(df)
    df = df.dropna(subset=FEATURES)
    if "clip_I" in df.columns:
        df = df[(df["clip_I"] == 0) & (df["clip_V"] == 0)]
    dropped = before - len(df)

    # Print how much data there is for each label
    if verbose:
        print(f"Loaded {len(df)} rows from {len(frames)} runs in {data_dir}"
              + (f"  ({dropped} bad rows dropped)" if dropped else ""))
        summary = df.groupby("label").agg(rows=("run_id", "size"), runs=("run_id", "nunique"))
        print(summary.to_string(), "\n")

    # Features, labels and run IDs for training
    return df[FEATURES], df["label"], df["run_id"]


# Running this file directly shows a quick summary of the data
if __name__ == "__main__":
    X, y, groups = load_dataset()
    print(X.describe().round(3).to_string())