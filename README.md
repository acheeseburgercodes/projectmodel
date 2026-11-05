# Delivery Last-Mile Delay Intelligence

This project turns a messy shipment export into two operational answers: how long a parcel is likely to take, and whether the team should intervene before it misses its promise. It also groups the 24 hubs by how they actually operate, rather than assuming the administrative tier is always right.

## How to run

Use Python 3.10 or newer from the repository root. The commands below reproduce the full project in order.

```bash
pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

`package_model.py` creates `models/delay_model.pkl` using the scikit-learn version pinned in `requirements.txt`. A pickle is version-sensitive, so rerun that script after installing the pinned packages instead of reusing the committed file with a different scikit-learn release.

## Data findings

The raw export contains 1,224 rows. Cleaning removes 24 duplicate rows, treats five 999 kg scanner-fault readings as missing, standardises 12 service spellings down to three, parses rupee values, and median-fills the three numeric inputs. The resulting modelling table has 1,200 shipments and no missing model values.

The checked-in output tables are the source for these findings:

- Economy has the lowest delay rate at **8.57%**, while Express is delayed most often at **48.97%** because its 36-hour promise is much tighter.
- Delay rate rises from **4.00% in Tier 1** to **21.50% in Tier 2** and **44.00% in Tier 3**.
- Long-distance lanes are the clearest warning group: delay rate climbs from **2.19% at 0–200 km** to **85.23% at 601+ km**.
- The three strongest non-target correlations with delivery hours are promised hours (**+0.8372**), distance (**+0.4013**) and number of stops (**+0.3247**).

## Model results (regression)

The baseline predicts the training mean for every validation shipment. Linear regression, a tuned-depth decision tree, a random forest, and gradient boosting are compared on the same untouched validation rows. Only the selected regression model is evaluated on the test set. The tree-depth and learning-rate tables make the bias/variance trade-off visible: shallow trees underfit, an unlimited tree has a much wider train-to-validation gap, and learning rate 0.1 balances progress and overshoot better than 0.01 or 1.0 on this split.

Linear regression wins this validation split with **RMSE 4.8912 hours** and **R² 0.9313**, narrowly ahead of gradient boosting at **RMSE 4.9165** and **R² 0.9306**. Linear regression is therefore tested once and reaches **R² 0.9493**. Across the shared five folds, gradient boosting is stronger and steadier (**mean R² 0.9345, std 0.0072**) than linear regression (**0.9247, std 0.0092**). For an offline batch with regular retraining I would favour boosting; for this pinned holdout and the simplest explainable deployment, the linear result is defensible.

## Model results (classification)

The all-on-time baseline reaches **76.67% accuracy** only because delayed parcels are the minority; its recall and F1 are both zero. Logistic regression raises validation F1 to **0.8046**, the selected depth-6 tree reaches **0.7470**, and the default forest reaches **0.7848**. The five bookkeeping columns are deliberately retained as a negative control. Together they account for **22.45%** of the forest importance, while each ranks eighth or lower by F-score and every noise F-score is below 5.

The final forest uses the top seven features selected on training rows only. Grid search evaluates **27 candidates / 135 fits** and randomized search evaluates **9 candidates / 45 fits**. The alert threshold is selected on validation probabilities, never on the test set: the highest threshold whose validation recall remains at least 0.80 is **0.55**. This reflects the business rule that missing a genuinely delayed parcel costs more than making an extra call. On test data it gives **precision 0.7143, recall 0.7317 and F1 0.7229**.

## Operations recommendation

Three segments are retained even though separation rises from **5.393 at K=3** to **6.305 at K=4** and **6.887 at K=5**. Three still produces usable groups of **9, 8 and 7 hubs**, while K=5 creates groups as small as two hubs. Segment names follow actual delay rate: **Standard 2.57%**, **Watch 15.50%**, and **Escalate 46.00%**. `outputs/tier_vs_segment_mismatch.csv` identifies the four hubs whose observed pattern differs from their official tier: Kolkata, Nagpur, Kochi and Dehradun.

The action list combines parcel risk with the hub segment:

- Unflagged parcels need no call.
- Flagged parcels in Standard hubs get a customer call.
- Flagged parcels in Watch hubs add hub monitoring.
- Flagged parcels in Escalate hubs also go to the hub manager.

For the supplied pending batch, **6 of 48 parcels are flagged** and **5 of those need hub-manager escalation**.

## Production plan

Before production, the same preprocessing should run in one versioned service for both training and scoring. The rollout plan covers the seven practical readiness characteristics:

- **Accuracy:** keep the held-out test report and require recall of at least 0.70 before promotion.
- **Reliability:** validate the schema and unknown hubs, retry transient batch failures, and quarantine invalid rows.
- **Reproducibility:** retain the generator seed, pinned dependencies, feature order, medians and threshold beside the model.
- **Performance (speed):** batch vectorised scoring and track p95 runtime; investigate if 10,000 rows exceed 30 seconds.
- **Maintainability:** keep cleaning in `src/shipment_prep.py`, add regression tests around its public functions, and version model bundles.
- **Consistency:** use the same shared cleaning and feature functions during training and prediction, with a reload equality check.
- **Monitorability:** log model version, schema result, probability, threshold and final action for every scored parcel.

Monitor at least these quantities weekly, split by service and hub segment where useful:

- delayed share among parcels whose outcomes are known; trigger investigation above the training rate by 5 percentage points;
- share of parcels flagged; trigger if it moves by more than 10 percentage points week over week;
- recall on matured parcels; trigger below 0.70;
- precision of calls; trigger below 0.40;
- missing or sentinel rate for each model input; trigger above 2%;
- distance and stops distribution drift; trigger when PSI exceeds 0.20;
- scoring failures or unknown hubs; trigger on any non-zero batch;
- probability calibration; trigger if observed delay differs from predicted risk by more than 0.10 in two adjacent bands.

## Limitations

- The data is synthetic, so performance is evidence that the pipeline works, not proof of real-world lift.
- Only 41 delayed parcels reach the test set, which makes recall and precision estimates noisy.
- One validation split sets the alert threshold; it should be confirmed on later time windows.
- Outcomes ignore cancellations, attempted deliveries and customer availability.
- Hub behaviour and seasonal promotion effects can drift after deployment.
- The current action costs are qualitative; a measured cost matrix would support a better threshold decision.
