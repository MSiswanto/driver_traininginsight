# utils/anomaly.py
import os
import math
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Callable, Union
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

DEFAULT_TELEMETRY_PATH = "https://github.com/MSiswanto/driver_traininginsight/releases/download/csv/telemetry_filtered_v2.csv"

# Type for progress callback: accepts float 0..1 or int percent
ProgressCallback = Optional[Callable[[float], None]]


def load_if_path(x: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """Return DataFrame if x is a DataFrame, otherwise read CSV from path/URL."""
    if isinstance(x, pd.DataFrame):
        return x.copy()
    if isinstance(x, str):
        return pd.read_csv(x)
    raise TypeError(f"Unsupported telemetry input: {type(x)}")


def _chunked_aggregate_means(path_or_df: Union[str, pd.DataFrame],
                             metrics_keep: Optional[List[str]],
                             chunksize: int = 200_000,
                             progress: ProgressCallback = None) -> pd.DataFrame:
    """
    Read long-format telemetry CSV in chunks and compute aggregated mean per
    (vehicle_id, vehicle_number, lap, telemetry_name).
    Accepts either:
      - path_or_df: URL or local path (string) -> read in chunks
      - OR a Pandas DataFrame (long format) -> process in-memory
    Returns aggregated long-format DataFrame with columns:
      vehicle_id, vehicle_number, lap, telemetry_name, telemetry_value_mean
    """
    if metrics_keep is None:
        metrics_keep = ["speed", "aps", "pbrake_f", "pbrake_r",
                        "Steering_Angle", "accx_can", "accy_can", "ath"]

    # Prepare reader: if DataFrame, use single-chunk list
    if isinstance(path_or_df, pd.DataFrame):
        reader = [path_or_df]
        total_est = len(path_or_df)
    else:
        # assume string path/URL
        if not (isinstance(path_or_df, str) and (path_or_df.startswith("http://") or path_or_df.startswith("https://") or os.path.exists(path_or_df))):
            raise FileNotFoundError(f"Telemetry file not found or invalid path: {path_or_df}")
        reader = pd.read_csv(path_or_df, chunksize=chunksize, engine="python")
        total_est = None  # unknown

    agg_list = []
    processed = 0
    for i, chunk in enumerate(reader):
        # if it's a chunk generator, chunk is DataFrame already
        chunk.columns = [c.strip() for c in chunk.columns]

        # ensure minimal columns
        must_cols = {"vehicle_id", "vehicle_number", "lap", "telemetry_name", "telemetry_value"}
        if not must_cols.issubset(set(chunk.columns)):
            raise ValueError("Input CSV must contain: vehicle_id, vehicle_number, lap, telemetry_name, telemetry_value")

        # restrict to metrics we want
        if metrics_keep:
            chunk = chunk[chunk["telemetry_name"].isin(metrics_keep)]

        # convert numeric where possible
        chunk["telemetry_value"] = pd.to_numeric(chunk["telemetry_value"], errors="coerce")
        chunk = chunk.dropna(subset=["telemetry_value"])
        if chunk.empty:
            processed += len(chunk)
            if progress and total_est:
                progress(min(1.0, processed / total_est))
            continue

        grp = chunk.groupby(["vehicle_id", "vehicle_number", "lap", "telemetry_name"], as_index=False)["telemetry_value"].mean()
        agg_list.append(grp)

        processed += len(chunk)
        if progress and total_est:
            progress(min(1.0, processed / total_est))

    if not agg_list:
        return pd.DataFrame(columns=["vehicle_id", "vehicle_number", "lap", "telemetry_name", "telemetry_value_mean"])

    combined = pd.concat(agg_list, ignore_index=True)
    combined = combined.groupby(["vehicle_id", "vehicle_number", "lap", "telemetry_name"], as_index=False)["telemetry_value"].mean()
    combined = combined.rename(columns={"telemetry_value": "telemetry_value_mean"})
    return combined


def load_and_pivot(path_or_df: Union[str, pd.DataFrame] = DEFAULT_TELEMETRY_PATH,
                   metrics_keep: Optional[List[str]] = None,
                   chunksize: int = 200_000,
                   progress: ProgressCallback = None) -> pd.DataFrame:
    """
    Build a pivoted (wide) per-(vehicle_id,vehicle_number,lap) table using chunked aggregation.
    Accepts DataFrame or path/URL. Returns wide DataFrame.
    """
    if metrics_keep is None:
        metrics_keep = ["speed", "aps", "pbrake_f", "pbrake_r", "Steering_Angle", "accx_can", "accy_can", "ath"]

    # If path_or_df is already wide-format (has telemetry columns), attempt fast detect:
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
        # detection heuristics: if telemetry_name present -> long-format; else likely already wide
        long_format = {"telemetry_name", "telemetry_value"}.issubset(set(df.columns))
        if not long_format and {"vehicle_id", "vehicle_number", "lap"}.issubset(set(df.columns)):
            # assume already wide
            pivot = df
            # ensure columns are strings
            pivot.columns = [str(c) for c in pivot.columns]
            # cast numeric telemetry columns
            for c in pivot.columns:
                if c not in ["vehicle_id", "vehicle_number", "lap"]:
                    pivot[c] = pd.to_numeric(pivot[c], errors="coerce")
            return pivot
        # else fallthrough to aggregated long-format processing

    long_agg = _chunked_aggregate_means(path_or_df, metrics_keep=metrics_keep, chunksize=chunksize, progress=progress)
    if long_agg.empty:
        return pd.DataFrame()

    pivot = long_agg.pivot_table(
        index=["vehicle_id", "vehicle_number", "lap"],
        columns="telemetry_name",
        values="telemetry_value_mean",
        aggfunc="mean"
    ).reset_index()

    pivot.columns.name = None
    pivot.columns = [str(c) for c in pivot.columns]

    for c in pivot.columns:
        if c not in ["vehicle_id", "vehicle_number", "lap"]:
            pivot[c] = pd.to_numeric(pivot[c], errors="coerce")

    return pivot


def feature_engineer(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Convert pivoted wide table into a compact feature DataFrame suitable for anomaly detection.
    """
    df = df_wide.copy()
    feats = pd.DataFrame(index=df.index)

    # common features (robust)
    def _to_num(col):
        return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)

    feats["speed"] = _to_num("speed")
    feats["throttle"] = _to_num("aps").fillna(_to_num("ath"))
    feats["brake"] = _to_num("pbrake_f").fillna(_to_num("pbrake_r"))
    steer_col = next((c for c in df.columns if "steer" in c.lower()), None)
    feats["steer_abs"] = df[steer_col].abs() if steer_col else pd.Series(np.nan, index=df.index)
    if "accx_can" in df.columns and "accy_can" in df.columns:
        feats["accel_mag"] = np.sqrt(_to_num("accx_can").fillna(0)**2 + _to_num("accy_can").fillna(0)**2)
    else:
        feats["accel_mag"] = np.nan

    feats["throttle_brake_diff"] = feats["throttle"].fillna(0) - feats["brake"].fillna(0)

    # minimal fill
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
                              random_state: int = 42,
                              n_jobs: int = 1) -> pd.DataFrame:
    """
    Run IsolationForest + robust z scoring. Returns features + iso_score + max_z + final_score + is_anomaly.
    """
    if feats is None or feats.empty:
        return pd.DataFrame()

    X = feats.copy().astype(np.float32)
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X.values)

    iso = IsolationForest(contamination=isolation_contamination, random_state=random_state, n_jobs=n_jobs)
    iso.fit(Xs)
    iso_raw = -iso.decision_function(Xs)
    iso_min, iso_max = iso_raw.min(), iso_raw.max()
    iso_norm = (iso_raw - iso_min) / (iso_max - iso_min + 1e-12)

    # z-scores
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


def run_full_anomaly_pipeline(path_or_df: Union[str, pd.DataFrame] = DEFAULT_TELEMETRY_PATH,
                              metrics_keep: Optional[List[str]] = None,
                              contamination: float = 0.02,
                              chunksize: int = 200_000,
                              progress: ProgressCallback = None,
                              sample_max_rows: Optional[int] = 250_000,
                              n_jobs: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete pipeline that accepts:
      - path_or_df: URL/local path (string) OR a DataFrame (long or wide)
    Returns (pivot_df, anomalies_df).
    Optional:
      - progress: callback(progress_float) for UI updates
      - sample_max_rows: if long input is huge, subsample to this many rows before pivoting
    """
    # If input is DataFrame and already wide, load_and_pivot will detect it
    pivot = load_and_pivot(path_or_df, metrics_keep=metrics_keep, chunksize=chunksize, progress=progress)

    if pivot is None or pivot.empty:
        return pd.DataFrame(), pd.DataFrame()

    # optionally downsample very large pivot tables (keep stratified by vehicle_number+lap)
    if sample_max_rows and len(pivot) > sample_max_rows:
        # sample proportionally by vehicle_number to keep representation
        pivot = pivot.groupby("vehicle_number", group_keys=False).apply(
            lambda g: g.sample(min(len(g), max(1, int(sample_max_rows / pivot["vehicle_number"].nunique()))), random_state=42)
        ).reset_index(drop=True)

    feats = feature_engineer(pivot)
    anomalies = detect_anomalies_combined(feats, isolation_contamination=contamination, random_state=42, n_jobs=n_jobs)
    return pivot.reset_index(drop=True), anomalies.reset_index(drop=True)


def format_anomaly_summary(pivot: pd.DataFrame,
                           anomalies: pd.DataFrame,
                           top_n: int = 10) -> str:
    if anomalies is None or anomalies.empty:
        return "No anomalies detected."
    top = anomalies.sort_values("final_score", ascending=False).head(top_n)
    lines = ["Top Detected Anomalies:"]
    for _, row in top.iterrows():
        vid = row.get("vehicle_number", row.get("vehicle_id", "N/A"))
        lap = row.get("lap", "N/A")
        score = row.get("final_score", 0.0)
        lines.append(f"- Vehicle {vid}, Lap {lap} → final={score:.3f}")
    return "\n".join(lines)
