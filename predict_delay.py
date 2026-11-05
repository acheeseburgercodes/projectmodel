"""Score the pending shipment batch and turn risk into an operations action list."""
from pathlib import Path

import joblib
import pandas as pd

from src.shipment_prep import build_features, clean_shipments


def action_for(flagged: int, segment: str) -> str:
    if not flagged:
        return "No call"
    if segment == "Escalate":
        return "Call customer + hub manager"
    if segment == "Watch":
        return "Call customer + watch hub"
    return "Call customer"


def main() -> None:
    models = Path("models")
    model = joblib.load(models / "delay_model.pkl")
    threshold = float(pd.read_csv(models / "delay_threshold.csv").loc[0, "threshold"])
    feature_order = pd.read_csv(models / "feature_order.csv")["feature"].tolist()
    median_frame = pd.read_csv(models / "impute_values.csv")
    medians = dict(zip(median_frame["column"], median_frame["median"]))

    new = pd.read_csv("data/new_shipments.csv")
    clean, _ = clean_shipments(new, medians=medians)
    featured = build_features(clean, pd.read_csv("data/hubs.csv"))
    probability = model.predict_proba(featured[feature_order])[:, 1]
    segments = pd.read_csv("outputs/hub_segments.csv")[["hub_id", "segment"]]

    scored = featured[["shipment_id", "hub_id"]].merge(segments, on="hub_id", validate="many_to_one")
    scored["delay_probability"] = probability.round(4)
    scored["flagged"] = (probability >= threshold).astype(int)
    scored["action"] = [action_for(flag, segment) for flag, segment in zip(scored["flagged"], scored["segment"])]
    scored.to_csv("outputs/new_shipment_actions.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
