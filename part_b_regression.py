"""Compare delivery-time regressors using a leakage-safe train/validation/test split."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.shipment_prep import FEATURES

OUTPUTS = Path("outputs")


def score(name, split, y_true, prediction):
    return {
        "model": name,
        "split": split,
        "MAE": mean_absolute_error(y_true, prediction),
        "RMSE": mean_squared_error(y_true, prediction) ** 0.5,
        "R2": r2_score(y_true, prediction),
    }


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    table = pd.read_csv("data/modelling_table.csv")
    x, y = table[FEATURES], table["delivery_hours"]
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.30, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.50, random_state=42)

    scaler = StandardScaler().fit(x_train)
    train_scaled = scaler.transform(x_train)
    val_scaled = scaler.transform(x_val)
    test_scaled = scaler.transform(x_test)
    scale_checks = pd.DataFrame({
        "metric": ["train_rows", "validation_rows", "test_rows", "train_scaled_max_abs_mean", "test_scaled_max_abs_mean"],
        "value": [len(x_train), len(x_val), len(x_test), np.abs(train_scaled.mean(axis=0)).max(), np.abs(test_scaled.mean(axis=0)).max()],
    })
    scale_checks.to_csv(OUTPUTS / "part_b_checks.csv", index=False, float_format="%.4f")

    rows = []
    baseline = np.repeat(y_train.mean(), len(y_val))
    rows.append(score("mean_baseline", "validation", y_val, baseline))

    linear = LinearRegression().fit(train_scaled, y_train)
    rows.append(score("linear_regression", "validation", y_val, linear.predict(val_scaled)))

    depth_rows = []
    for depth in [1, 2, 3, 4, 6, 8, 12, None]:
        tree = DecisionTreeRegressor(max_depth=depth, random_state=42).fit(x_train, y_train)
        depth_rows.append({
            "max_depth": "unlimited" if depth is None else depth,
            "train_rmse": mean_squared_error(y_train, tree.predict(x_train)) ** 0.5,
            "val_rmse": mean_squared_error(y_val, tree.predict(x_val)) ** 0.5,
        })
    depth_frame = pd.DataFrame(depth_rows)
    depth_frame.to_csv(OUTPUTS / "part_b_tree_depth.csv", index=False, float_format="%.4f")
    best_depth = int(depth_frame.loc[depth_frame["val_rmse"].idxmin(), "max_depth"])
    best_tree = DecisionTreeRegressor(max_depth=best_depth, random_state=42).fit(x_train, y_train)
    rows.append(score(f"decision_tree_depth_{best_depth}", "validation", y_val, best_tree.predict(x_val)))

    forest = RandomForestRegressor(n_estimators=100, random_state=42).fit(x_train, y_train)
    rows.append(score("random_forest", "validation", y_val, forest.predict(x_val)))

    rate_rows = []
    boosters = {}
    for rate in [0.01, 0.1, 1.0]:
        booster = GradientBoostingRegressor(n_estimators=100, learning_rate=rate, random_state=42).fit(x_train, y_train)
        prediction = booster.predict(x_val)
        rate_rows.append({
            "learning_rate": rate,
            "val_rmse": mean_squared_error(y_val, prediction) ** 0.5,
            "val_r2": r2_score(y_val, prediction),
        })
        boosters[rate] = booster
    rate_frame = pd.DataFrame(rate_rows)
    rate_frame.to_csv(OUTPUTS / "part_b_learning_rate.csv", index=False, float_format="%.4f")
    best_rate = float(rate_frame.loc[rate_frame["val_rmse"].idxmin(), "learning_rate"])
    booster = boosters[best_rate]
    rows.append(score("gradient_boosting", "validation", y_val, booster.predict(x_val)))

    candidates = {
        "linear_regression": (rows[1]["RMSE"], linear, True),
        "gradient_boosting": (rows[-1]["RMSE"], booster, False),
    }
    winner_name, (_, winner, uses_scaled) = min(candidates.items(), key=lambda item: item[1][0])
    final_prediction = winner.predict(test_scaled if uses_scaled else x_test)
    rows.append(score(winner_name, "test", y_test, final_prediction))
    pd.DataFrame(rows).to_csv(OUTPUTS / "part_b_metrics.csv", index=False, float_format="%.4f")

    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rows = []
    for name, estimator in [
        ("linear_regression", Pipeline([
            ("scale", StandardScaler()), ("model", LinearRegression())
        ])),
        ("gradient_boosting", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)),
    ]:
        scores = cross_val_score(estimator, x_train, y_train, cv=folds, scoring="r2")
        cv_rows.append({"model": name, "mean_r2": scores.mean(), "std_r2": scores.std()})
    pd.DataFrame(cv_rows).to_csv(OUTPUTS / "part_b_cv.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
