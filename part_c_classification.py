"""Train the delay classifier, tune the forest, and set the alert threshold."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.shipment_prep import FEATURES, NOISE_COLUMNS

OUTPUTS = Path("outputs")


def result_row(name, truth, prediction):
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, prediction, average="binary", zero_division=0
    )
    return {
        "model": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": accuracy_score(truth, prediction), "precision": precision,
        "recall": recall, "f1": f1,
    }


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    table = pd.read_csv("data/modelling_table.csv")
    x, y = table[FEATURES], table["is_delayed"]
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.30, random_state=42, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    rows = [result_row("blind_all_on_time", y_val, np.zeros(len(y_val), dtype=int))]
    logistic = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=42)),
    ]).fit(x_train, y_train)
    rows.append(result_row("logistic_regression", y_val, logistic.predict(x_val)))

    tree_scores = []
    for depth in [1, 2, 3, 4, 5, 6, 8, 10]:
        tree = DecisionTreeClassifier(criterion="gini", max_depth=depth, random_state=42).fit(x_train, y_train)
        candidate = result_row(f"decision_tree_depth_{depth}", y_val, tree.predict(x_val))
        tree_scores.append(candidate)
    best_tree = max(tree_scores, key=lambda row: (row["f1"], -int(row["model"].rsplit("_", 1)[1])))
    rows.append(best_tree)

    default_forest = RandomForestClassifier(n_estimators=100, random_state=42).fit(x_train, y_train)
    rows.append(result_row("random_forest_default", y_val, default_forest.predict(x_val)))
    pd.DataFrame(rows).to_csv(OUTPUTS / "part_c_models.csv", index=False, float_format="%.4f")

    selector = SelectKBest(score_func=f_classif, k="all").fit(x_train, y_train)
    signal = pd.DataFrame({
        "feature": FEATURES,
        "f_score": selector.scores_,
        "rf_importance": default_forest.feature_importances_,
    })
    signal["f_rank"] = signal["f_score"].rank(method="min", ascending=False).astype(int)
    signal["is_noise"] = signal["feature"].isin(NOISE_COLUMNS)
    signal = signal[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]].sort_values("f_rank")
    signal.to_csv(OUTPUTS / "part_c_feature_scores.csv", index=False, float_format="%.4f")

    final_selector = SelectKBest(score_func=f_classif, k=7).fit(x_train, y_train)
    selected_train = final_selector.transform(x_train)
    selected_val = final_selector.transform(x_val)
    parameters = {
        "n_estimators": [50, 100, 150],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [1, 5, 10],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42), parameters, cv=5, n_jobs=-1
    ).fit(selected_train, y_train)
    randomized = RandomizedSearchCV(
        RandomForestClassifier(random_state=42), parameters, n_iter=9, cv=5,
        random_state=42, n_jobs=-1,
    ).fit(selected_train, y_train)

    default_selected = RandomForestClassifier(n_estimators=100, random_state=42).fit(selected_train, y_train)
    default_f1 = result_row("default_forest_same_7_features", y_val, default_selected.predict(selected_val))["f1"]
    default_row = {
        "search": "default_forest_same_7_features", "candidates": 1, "total_fits": 1,
        "best_params": "{}", "best_cv_accuracy": np.nan, "val_f1": default_f1,
    }
    tuning_rows = []
    search_candidates = [("default_forest_same_7_features", default_f1, {})]
    for method, search, candidates_count in [
        ("grid_search", grid, 27), ("randomized_search", randomized, 9)
    ]:
        val_prediction = search.predict(selected_val)
        f1 = result_row(method, y_val, val_prediction)["f1"]
        tuning_rows.append({
            "search": method, "candidates": candidates_count, "total_fits": candidates_count * 5,
            "best_params": str(search.best_params_), "best_cv_accuracy": search.best_score_, "val_f1": f1,
        })
        search_candidates.append((method, f1, search.best_params_))
    tuning_rows.append(default_row)
    pd.DataFrame(tuning_rows).to_csv(OUTPUTS / "part_c_tuning.csv", index=False, float_format="%.4f")

    tie_priority = {"default_forest_same_7_features": 0, "grid_search": 1, "randomized_search": 2}
    winner_name, _, chosen_params = sorted(
        search_candidates, key=lambda item: (-item[1], tie_priority[item[0]])
    )[0]
    winner = Pipeline([
        ("selector", SelectKBest(score_func=f_classif, k=7)),
        ("model", RandomForestClassifier(random_state=42, **chosen_params)),
    ]).fit(x_train, y_train)

    val_probability = winner.predict_proba(x_val)[:, 1]
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    threshold_rows = []
    for threshold in thresholds:
        metrics = result_row("threshold", y_val, (val_probability >= threshold).astype(int))
        threshold_rows.append({
            "threshold": threshold, "TP": metrics["TP"], "FP": metrics["FP"],
            "accuracy": metrics["accuracy"], "precision": metrics["precision"],
            "recall": metrics["recall"], "f1": metrics["f1"],
        })
    threshold_frame = pd.DataFrame(threshold_rows)
    threshold_frame.to_csv(OUTPUTS / "part_c_thresholds.csv", index=False, float_format="%.4f")

    # Missing a late parcel is costlier than an unnecessary call, so retain at least
    # 0.80 validation recall and use the highest threshold that still does so.
    eligible = threshold_frame[threshold_frame["recall"] >= 0.80]
    threshold = float(eligible["threshold"].max()) if not eligible.empty else 0.1
    test_row = result_row("final_model_test", y_test, (winner.predict_proba(x_test)[:, 1] >= threshold).astype(int))
    pd.DataFrame([test_row]).to_csv(OUTPUTS / "part_c_final_test.csv", index=False, float_format="%.4f")

if __name__ == "__main__":
    main()
