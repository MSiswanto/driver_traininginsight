# utils/anomaly.py
import os
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from typing import Tuple, List

DEFAULT_TELEMETRY_PATH = "data/telemetry_filtered_v2.csv"

def _chunked_aggregate_means(path: str, metrics_keep: List[str], chunksize: int = 200_000) -> pd.DataFrame:
    """
    Read long-format telemetry CSV in chunks and compute aggregated mean per
    (vehicle_id, vehicle_number, lap, telemetry_name).
    Returns aggregated long-format DataFrame with columns:
      vehicle_id, vehicle_number, lap, telemetry_name, telemetry_value_mean
    This prevents loading the entire file into memory as a single DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Telemetry file not found at: {path}")

    agg_list = []
    cols = None
    reader = pd.read_csv(path, chunksize=chunksize, engine="python")
    for i, chunk in enumerate(reader):
        # normalize columns
        chunk.columns = [c.strip() for c in chunk.columns]
        # ensure minimal columns
        if not set(["vehicle_id", "vehicle_number", "lap", "telemetry_name", "telemetry_value"]).issubset(set(chunk.columns)):
            # try relaxed parsing (single header cell with commas)
            raise ValueError("Input CSV must contain: vehicle_id, vehicle_number, lap, telemetry_name, telemetry_value")
        # restrict to metrics we want
        if metrics_keep:
            chunk = chunk[chunk["telemetry_name"].isin(metrics_keep)]
        # convert numeric where possible
        chunk["telemetry_value"] = pd.to_numeric(chunk["telemetry_value"], errors="coerce")
        # drop rows with null telemetry_value to reduce work
        chunk = chunk.dropna(subset=["telemetry_value"])
        if chunk.empty:
            continue
        grp = chunk.groupby(["vehicle_id", "vehicle_number", "lap", "telemetry_name"], as_index=False)["telemetry_value"].mean()
        agg_list.append(grp)
    if not agg_list:
        return pd.DataFrame(columns=["vehicle_id","vehicle_number","lap","telemetry_name","telemetry_value"])
    combined = pd.concat(agg_list, ignore_index=True)
    # combine again in case same keys appear across chunks
    combined = combined.groupby(["vehicle_id","vehicle_number","lap","telemetry_name"], as_index=False)["telemetry_value"].mean()
    return combined.rename(columns={"telemetry_value": "telemetry_value_mean"})

def load_and_pivot(path: str = DEFAULT_TELEMETRY_PATH,
                   metrics_keep=None,
                   chunksize: int = 200_000) -> pd.DataFrame:
    """
    Build a pivoted (wide) per-(vehicle_id,vehicle_number,lap) table using chunked aggregation.
    Returns DataFrame with columns: vehicle_id, vehicle_number, lap, <metric columns...>
    """
    if metrics_keep is None:
        metrics_keep = ["speed", "aps", "pbrake_f", "pbrake_r", "Steering_Angle", "accx_can", "accy_can", "ath"]

    long_agg = _chunked_aggregate_means(path, metrics_keep, chunksize=chunksize)
    if long_agg.empty:
        return pd.DataFrame()

    # pivot to wide
    pivot = long_agg.pivot_table(
        index=["vehicle_id", "vehicle_number", "lap"],
        columns="telemetry_name",
        values="telemetry_value_mean",
        aggfunc="mean"
    ).reset_index()

    # flatten columns
    pivot.columns.name = None
    pivot.columns = [str(c) for c in pivot.columns]

    # cast numeric telemetry columns
    for c in pivot.columns:
        if c not in ["vehicle_id", "vehicle_number", "lap"]:
            pivot[c] = pd.to_numeric(pivot[c], errors="coerce")

    return pivot

def feature_engineer(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Convert pivoted wide table into a compact feature DataFrame suitable for anomaly detection.
    We build only a few robust features to keep memory small.
    """
    df = df_wide.copy()
    # choose columns present
    telemetry_cols = [c for c in df.columns if c not in ["vehicle_id", "vehicle_number", "lap"]]

    # normalize some names
    rename_map = {
        "Steering_Angle": "Steering_Angle",  # keep original if present
        "aps": "aps",
        "ath": "ath",
        "accx_can": "accx_can",
        "accy_can": "accy_can",
        "pbrake_f": "pbrake_f",
        "pbrake_r": "pbrake_r"
    }
    # pick simple features
    feats = pd.DataFrame(index=df.index)

    # speed
    if "speed" in df.columns:
        feats["speed"] = pd.to_numeric(df["speed"], errors="coerce")
    else:
        feats["speed"] = np.nan

    # throttle prefer aps then ath
    if "aps" in df.columns:
        feats["throttle"] = pd.to_numeric(df["aps"], errors="coerce")
    elif "ath" in df.columns:
        feats["throttle"] = pd.to_numeric(df["ath"], errors="coerce")
    else:
        feats["throttle"] = np.nan

    # brakes
    if "pbrake_f" in df.columns:
        feats["brake"] = pd.to_numeric(df["pbrake_f"], errors="coerce")
    elif "pbrake_r" in df.columns:
        feats["brake"] = pd.to_numeric(df["pbrake_r"], errors="coerce")
    else:
        feats["brake"] = np.nan

    # steering absolute
    steer_col = next((c for c in df.columns if "steer" in c.lower()), None)
    if steer_col:
        feats["steer_abs"] = df[steer_col].abs()
    else:
        feats["steer_abs"] = np.nan

    # accel magnitude if both present
    if "accx_can" in df.columns and "accy_can" in df.columns:
        feats["accel_mag"] = np.sqrt(df["accx_can"].fillna(0)**2 + df["accy_can"].fillna(0)**2)
    else:
        feats["accel_mag"] = np.nan

    # derived
    feats["throttle_brake_diff"] = feats["throttle"].fillna(0) - feats["brake"].fillna(0)

    # fill minimal missing values with median (safe)
    feats = feats.fillna(feats.median())

    return feats

def robust_z_score(series: pd.Series) -> pd.Series:
    med = series.median()
    mad = np.median(np.abs(series - med))
    if mad == 0 or np.isnan(mad):
        std = series.std() if series.std() != 0 else 1.0
        return (series - med) / std
    return 0.6745 * (series - med) / mad

def detect_anomalies_combined(feats: pd.DataFrame,
                              isolation_contamination: float = 0.02,
                              zscore_threshold: float = 3.0,
                              iso_weight: float = 0.6,
                              z_weight: float = 0.4,
                              random_state: int = 42) -> pd.DataFrame:
    """
    Run scaled IsolationForest on small feature matrix (rows ~ vehicle-lap).
    Returns features + iso_score + max_z + final_score + is_anomaly
    """
    X = feats.copy()
    # keep limited dtype float32 to cut memory usage
    X = X.astype(np.float32)

    # scale with RobustScaler
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X.values)

    iso = IsolationForest(contamination=isolation_contamination, random_state=random_state, n_jobs=1)  # n_jobs=1 safer
    iso.fit(Xs)
    iso_raw = -iso.decision_function(Xs)  # bigger -> more anomalous
    # normalize iso to 0..1
    iso_min, iso_max = iso_raw.min(), iso_raw.max()
    iso_norm = (iso_raw - iso_min) / (iso_max - iso_min + 1e-12)

    # robust z per column, then max per row
    # Use float32 to reduce memory
    z_abs = np.abs(X.apply(robust_z_score, axis=0).astype(np.float32))
    max_z = z_abs.max(axis=1).values
    z_norm = np.clip(max_z / (zscore_threshold * 2.0), 0.0, 1.0)

    final_score = iso_weight * iso_norm + z_weight * z_norm

    out = feats.copy().reset_index(drop=True)
    out["iso_score"] = iso_norm
    out["max_z"] = max_z
    out["z_norm"] = z_norm
    out["final_score"] = final_score
    threshold = max(np.percentile(final_score, 100*(1 - isolation_contamination)), 0.6)
    out["is_anomaly"] = out["final_score"] >= threshold
    return out

def run_full_anomaly_pipeline(path: str = DEFAULT_TELEMETRY_PATH,
                              metrics_keep=None,
                              contamination: float = 0.02,
                              chunksize: int = 200_000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete pipeline: chunked aggregate -> pivot -> features -> anomalies.
    Returns (pivot_df, anomalies_df)
    pivot_df: per vehicle_id,vehicle_number,lap wide table
    anomalies_df: same index rows with anomaly outputs
    """
    pivot = load_and_pivot(path=path, metrics_keep=metrics_keep, chunksize=chunksize)
    if pivot is None or pivot.empty:
        return pd.DataFrame(), pd.DataFrame()

    feats = feature_engineer(pivot)
    anomalies = detect_anomalies_combined(feats, isolation_contamination=contamination, random_state=42)
    return pivot.reset_index(drop=True), anomalies.reset_index(drop=True)

def format_anomaly_summary(pivot: pd.DataFrame,
                           anomalies: pd.DataFrame,
                           top_n: int = 10) -> str:
    """
    Create a readable text summary explaining the top anomalies.
    """
    if anomalies is None or anomalies.empty:
        return "No anomalies detected."

    # Sort by highest anomaly score
    top = anomalies.sort_values("final_score", ascending=False).head(top_n)

    lines = ["Top Detected Anomalies:"]
    for _, row in top.iterrows():
        vid = row.get("vehicle_number", row.get("vehicle_id", "N/A"))
        lap = row.get("lap", "N/A")
        score = row.get("final_score", 0.0)
        iso = row.get("iso_score", 0.0)
        mz = row.get("max_z", 0.0)

        lines.append(
            f"- Vehicle {vid}, Lap {lap} → final={score:.3f}, iso={iso:.3f}, z={mz:.2f}"
        )

    return "\n".join(lines)

