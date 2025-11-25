import os
import pandas as pd

DATA_FOLDER = "data"

# ================================================================
# LOAD RAW FILES
# ================================================================
def load_all_data(folder_path=DATA_FOLDER):

    important_files = [
        "23_AnalysisEnduranceWithSections_Race 1_Anonymized.csv", # telemetry
        "26_Weather_Race 1_Anonymized.csv",                       # weather
        "99_Best 10 Laps By Driver_Race 1_Anonymized.csv",        # lap times
    ]

    data = {}

    for filename in important_files:
        path = os.path.join(folder_path, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Loaded {filename} → {df.shape}")
            data[filename] = df
        else:
            print(f"⚠️ Missing file: {filename}")

    return data


# ================================================================
# PROVIDE CLEAN DATA FOR ANALYSIS
# ================================================================
def load_lap_data():
    """Return cleaned lap time dataset from file 99."""
    path = os.path.join(DATA_FOLDER, "99_Best 10 Laps By Driver_Race 1_Anonymized.csv")
    df = pd.read_csv(path)

    # Standardize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Expected important columns:
    # driver_number / driver / lap / lap_time
    rename_map = {
        "driver_number": "driver",
        "driver_name": "driver",
        "lapnumber": "lap",
        "lap_number": "lap",
        "bestlap": "lap_time",
        "best_lap": "lap_time",
        "lap_time_s": "lap_time"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Convert times to float
    if "lap_time" in df.columns:
        df["lap_time"] = pd.to_numeric(df["lap_time"], errors="coerce")

    return df


def load_telemetry_data():
    """Return telemetry dataset from file 23."""
    path = os.path.join(DATA_FOLDER, "23_AnalysisEnduranceWithSections_Race 1_Anonymized.csv")
    df = pd.read_csv(path)

    # Standardize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Expected telemetry fields:
    # driver, distance, speed, brake, throttle, steering_angle

    rename_map = {
        "driver_number": "driver",
        "driver_name": "driver",
        "distance_m": "distance",
        "speed_kmh": "speed",
        "brake_pressure": "brake",
        "throttle_position": "throttle",
        "aps": "throttle",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Fill missing numeric
    numeric_cols = ["distance", "speed", "brake", "throttle"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
