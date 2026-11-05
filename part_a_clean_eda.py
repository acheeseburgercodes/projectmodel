"""Build the modelling table and the first set of delay evidence."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.shipment_prep import FEATURES, build_features, clean_shipments

DATA = Path("data")
OUTPUTS = Path("outputs")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.4f")


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    raw = pd.read_csv(DATA / "raw_shipments.csv")
    hubs = pd.read_csv(DATA / "hubs.csv")
    deduped = raw.drop_duplicates()

    before_spellings = deduped["service_type"].nunique()
    clean, medians = clean_shipments(raw)
    model = build_features(clean, hubs)
    model["is_delayed"] = (model["delivery_hours"] > model["promised_hours"]).astype(int)
    write_csv(model, DATA / "modelling_table.csv")

    q1, q3 = model["delivery_hours"].quantile([0.25, 0.75])
    iqr = q3 - q1
    checks = [
        ("raw_rows", len(raw)),
        ("duplicate_rows_removed", len(raw) - len(deduped)),
        ("rows_after_dedup", len(deduped)),
        ("missing_distance_km_before", deduped["distance_km"].isna().sum()),
        ("missing_parcel_weight_before", deduped["parcel_weight_kg"].isna().sum()),
        ("sentinel_weight_rows", deduped["parcel_weight_kg"].gt(100).sum()),
        ("missing_declared_value_before", deduped["declared_value"].isna().sum()),
        ("missing_payment_mode_before", deduped["payment_mode"].isna().sum()),
        ("service_type_spellings_before", before_spellings),
        ("service_type_spellings_after", clean["service_type"].nunique()),
        ("declared_value_mean_inr", model["declared_value_inr"].mean()),
        ("declared_value_median_inr", model["declared_value_inr"].median()),
        ("log_declared_value_mean", model["log_declared_value"].mean()),
        ("log_declared_value_median", model["log_declared_value"].median()),
        ("median_distance_km", medians["distance_km"]),
        ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
        ("median_declared_value_inr", medians["declared_value_inr"]),
        ("rows_after_hub_merge", len(model)),
        ("nulls_in_features_and_target", model[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()),
        ("delayed_count", model["is_delayed"].sum()),
        ("delay_rate", model["is_delayed"].mean()),
        ("hours_q1", q1),
        ("hours_q3", q3),
        ("hours_lower_bound", q1 - 1.5 * iqr),
        ("hours_upper_bound", q3 + 1.5 * iqr),
        ("hours_outlier_count", ((model["delivery_hours"] < q1 - 1.5 * iqr) | (model["delivery_hours"] > q3 + 1.5 * iqr)).sum()),
    ]
    write_csv(pd.DataFrame(checks, columns=["metric", "value"]), OUTPUTS / "part_a_checks.csv")

    model["distance_band"] = pd.cut(
        model["distance_km"], bins=[0, 200, 400, 600, np.inf], labels=["0-200", "201-400", "401-600", "601+"],
        include_lowest=True, right=True,
    )
    evidence = []
    for dimension in ["service_type", "hub_tier", "distance_band"]:
        grouped = model.groupby(dimension, observed=True, sort=True)
        for group, part in grouped:
            evidence.append({
                "dimension": dimension, "group": group, "shipments": len(part),
                "delayed": int(part["is_delayed"].sum()), "delay_rate": part["is_delayed"].mean(),
                "mean_hours": part["delivery_hours"].mean(),
            })
    write_csv(pd.DataFrame(evidence), OUTPUTS / "part_a_delay_by_group.csv")

    numeric = model.select_dtypes(include="number")
    corr = numeric.corr()
    corr.to_csv(OUTPUTS / "part_a_correlations.csv", float_format="%.4f")


if __name__ == "__main__":
    main()
