"""Save the chosen classifier together with its training-time support files."""
import ast
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES, clean_shipments

MODELS = Path("models")
OUTPUTS = Path("outputs")


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    tuning_path = OUTPUTS / "part_c_tuning.csv"
    threshold_path = OUTPUTS / "part_c_thresholds.csv"
    if not tuning_path.exists() or not threshold_path.exists():
        from part_c_classification import main as train_classifier
        train_classifier()

    table = pd.read_csv("data/modelling_table.csv")
    x, y = table[FEATURES], table["is_delayed"]
    x_train, _, y_train, _ = train_test_split(
        x, y, test_size=0.30, random_state=42, stratify=y
    )
    tuning = pd.read_csv(tuning_path)
    priority = {"default_forest_same_7_features": 0, "grid_search": 1, "randomized_search": 2}
    tuning["priority"] = tuning["search"].map(priority)
    chosen = tuning.sort_values(["val_f1", "priority"], ascending=[False, True]).iloc[0]
    parameters = ast.literal_eval(chosen["best_params"])
    model = Pipeline([
        ("selector", SelectKBest(score_func=f_classif, k=7)),
        ("model", RandomForestClassifier(random_state=42, **parameters)),
    ]).fit(x_train, y_train)
    threshold_table = pd.read_csv(threshold_path)
    eligible = threshold_table[threshold_table["recall"] >= 0.80]
    threshold = float(eligible["threshold"].max()) if not eligible.empty else 0.1

    raw = pd.read_csv("data/raw_shipments.csv")
    _, medians = clean_shipments(raw)

    model_path = MODELS / "delay_model.pkl"
    joblib.dump(model, model_path)
    pd.DataFrame({"threshold": [threshold]}).to_csv(
        MODELS / "delay_threshold.csv", index=False, float_format="%.4f"
    )
    pd.DataFrame({"feature": FEATURES}).to_csv(MODELS / "feature_order.csv", index=False)
    pd.DataFrame({"column": list(medians), "median": list(medians.values())}).to_csv(
        MODELS / "impute_values.csv", index=False, float_format="%.4f"
    )

    # The serialization check catches a surprisingly common deployment failure early.
    sample = table[FEATURES].head(25)
    before = model.predict_proba(sample)[:, 1]
    after = joblib.load(model_path).predict_proba(sample)[:, 1]
    pd.DataFrame({"metric": ["reload_max_abs_probability_diff"], "value": [abs(before - after).max()]}).to_csv(
        OUTPUTS / "part_d_reload_check.csv", index=False, float_format="%.10f"
    )


if __name__ == "__main__":
    main()
