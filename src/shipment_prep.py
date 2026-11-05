"""Cleaning and feature construction shared by training and scoring."""
from __future__ import annotations

import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

NOISE_COLUMNS = [
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"
]


def clean_shipments(df: pd.DataFrame, medians: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Clean a raw shipment export and fill its three numeric gaps."""
    clean = df.drop_duplicates().copy()
    clean["service_type"] = clean["service_type"].str.strip().str.title()
    clean["payment_mode"] = clean["payment_mode"].fillna("Unknown")
    clean["declared_value_inr"] = pd.to_numeric(
        clean["declared_value"].astype("string").str.replace("₹", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    ).astype(float)
    clean = clean.drop(columns="declared_value")
    clean["dispatch_time"] = pd.to_datetime(clean["dispatch_time"], errors="raise")
    clean.loc[clean["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    fill_columns = ["distance_km", "parcel_weight_kg", "declared_value_inr"]
    if medians is None:
        medians = {column: float(clean[column].median()) for column in fill_columns}
    else:
        medians = {column: float(medians[column]) for column in fill_columns}
    clean[fill_columns] = clean[fill_columns].fillna(medians)
    return clean, medians


def build_features(clean: pd.DataFrame, hubs: pd.DataFrame) -> pd.DataFrame:
    """Attach hub fields and add the model features that are known at dispatch."""
    if hubs["hub_id"].duplicated().any():
        raise ValueError("hubs must contain exactly one row per hub_id")

    featured = clean.merge(hubs, on="hub_id", how="left", validate="many_to_one")
    if featured[["hub_tier", "hub_daily_capacity"]].isna().any().any():
        missing = featured.loc[featured["hub_tier"].isna(), "hub_id"].unique()
        raise ValueError(f"Unknown hub_id values: {missing.tolist()}")

    featured["promised_hours"] = featured["service_type"].map(PROMISED_HOURS).astype(int)
    featured["is_cod"] = featured["payment_mode"].eq("COD").astype(int)
    featured["is_weekend"] = featured["dispatch_time"].dt.dayofweek.ge(5).astype(int)
    featured["dispatch_hour"] = featured["dispatch_time"].dt.hour.astype(int)
    featured["log_declared_value"] = np.log1p(featured["declared_value_inr"])
    return featured
