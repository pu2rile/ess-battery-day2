"""B1-only selection, untouched B1 hold-out, then B2/B3 external evaluation."""
from pathlib import Path
import argparse
import json
import platform
import time
import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.special import exp10
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.features import CORE, EXTENDED

SEED = 42

def score(y, pred):
    return {"MAPE_pct": float(mean_absolute_percentage_error(y, pred) * 100),
            "MAE_cycles": float(mean_absolute_error(y, pred)),
            "RMSE_cycles": float(np.sqrt(mean_squared_error(y, pred))),
            "R2": float(r2_score(y, pred))}

def candidates():
    def model(estimator):
        pipeline = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                             ("scale", StandardScaler()), ("model", estimator)])
        return TransformedTargetRegressor(regressor=pipeline, func=np.log10, inverse_func=exp10)
    specs = [
        ("DummyMedian", CORE, DummyRegressor(strategy="median"), {}),
        ("LogLinear_DeltaQ", ["log_dq_var"], model(LinearRegression()), {}),
        ("Ridge_Core", CORE, model(Ridge()), {"regressor__model__alpha": [.1, 1., 10., 100.]}),
        ("ElasticNet_Core", CORE, model(ElasticNet(max_iter=100000, random_state=SEED)),
         {"regressor__model__alpha": [.001, .01, .1], "regressor__model__l1_ratio": [.2, .8]}),
        ("ElasticNet_Extended", EXTENDED, model(ElasticNet(max_iter=100000, random_state=SEED)),
         {"regressor__model__alpha": [.001, .01, .1], "regressor__model__l1_ratio": [.2, .8]}),
        ("GradientBoosting_Core", CORE,
         model(GradientBoostingRegressor(loss="huber", learning_rate=.05, n_estimators=100,
                                         min_samples_leaf=4, random_state=SEED)),
         {"regressor__model__max_depth": [1, 2], "regressor__model__n_estimators": [100, 200]}),
    ]
    return specs

def fit_search(estimator, grid, X, y, groups, folds=4):
    if not grid:
        return clone(estimator).fit(X, y), {}
    cv = GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    search = GridSearchCV(estimator, grid, scoring="neg_mean_absolute_percentage_error",
                          cv=cv, n_jobs=1, error_score="raise")
    search.fit(X, y, groups=groups)
    return search.best_estimator_, search.best_params_

def run_training(feature_path, results_dir):
    start = time.time()
    df = pd.read_csv(feature_path)
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    b1 = df.query("batch == 'B1'").reset_index(drop=True)
    ti, vi = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED)
                  .split(b1, groups=b1.policy_group))
    train, valid = b1.iloc[ti].copy(), b1.iloc[vi].copy()
    assert set(train.policy_group).isdisjoint(valid.policy_group)
    assert set(train.cell_uid).isdisjoint(valid.cell_uid)
    if train[EXTENDED].isna().all().any():
        raise ValueError("All-missing training feature; revise declared feature set.")
    assignment = df[["cell_uid", "batch", "policy_group"]].copy()
    assignment["split"] = assignment.batch.map({"B1": "B1_train", "B2": "B2_test", "B3": "B3_external"})
    assignment.loc[assignment.cell_uid.isin(valid.cell_uid), "split"] = "B1_valid"
    assignment.to_csv(out / "split_assignments.csv", index=False)
    groups = train.policy_group.to_numpy()
    y = train.cycle_life.to_numpy()
    outer = list(GroupKFold(n_splits=5).split(train, y, groups))
    fold_rows, oof_rows, comparisons = [], [], []
    for name, cols, estimator, grid in candidates():
        fold_scores = []
        for fold, (tr, va) in enumerate(outer, 1):
            assert set(groups[tr]).isdisjoint(groups[va])
            fitted, params = fit_search(estimator, grid, train.iloc[tr][cols], y[tr], groups[tr])
            pred = fitted.predict(train.iloc[va][cols])
            values = score(y[va], pred)
            fold_scores.append(values["MAPE_pct"])
            fold_rows.append({"model": name, "fold": fold, "n_train": len(tr), "n_valid": len(va),
                              **values, "params": json.dumps(params, sort_keys=True)})
            for idx, yp in zip(va, pred):
                oof_rows.append({"model": name, "cell_uid": train.iloc[idx].cell_uid, "fold": fold,
                                 "actual": y[idx], "prediction": float(yp)})
        comparisons.append({"model": name, "n_features": len(cols),
                            "CV_MAPE_mean_pct": float(np.mean(fold_scores)),
                            "CV_MAPE_sd_pp": float(np.std(fold_scores, ddof=1))})
        print(f"{name}: nested grouped CV MAPE {np.mean(fold_scores):.2f}%", flush=True)
    compare = pd.DataFrame(comparisons).sort_values("CV_MAPE_mean_pct").reset_index(drop=True)
    compare.to_csv(out / "model_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "cv_folds.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(out / "oof_predictions.csv", index=False)
    best_name = compare.iloc[0].model
    _, cols, estimator, grid = next(x for x in candidates() if x[0] == best_name)
    final, params = fit_search(estimator, grid, train[cols], y, groups)
    # Freeze selection before evaluating any hold-out or external labels.
    frozen = {"model": best_name, "features": cols, "params": params,
              "selection": "minimum B1-training nested grouped 5-fold mean MAPE",
              "random_state": SEED, "trained_on": train.cell_uid.tolist(),
              "valid_cells": valid.cell_uid.tolist(), "refit_with_holdout": False}
    (out / "frozen_model_spec.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    joblib.dump({"model": final, "features": cols}, out / "final_model.joblib")
    metrics, predictions = [], []
    for name, sub in [("Valid (Batch 1 Hold-out)", valid),
                      ("Test (Batch 2)", df.query("batch == 'B2'")),
                      ("External (Batch 3)", df.query("batch == 'B3'"))]:
        pred = final.predict(sub[cols])
        if not np.isfinite(pred).all() or (pred <= 0).any():
            raise ValueError("Nonpositive/nonfinite predictions")
        metrics.append({"split": name, "n": len(sub), **score(sub.cycle_life, pred)})
        p = sub[["cell_uid", "batch", "policy_group", "cycle_life"]].copy()
        p["split"] = name
        p["prediction"] = pred
        p["residual"] = pred - p.cycle_life
        p["APE_pct"] = abs(p.residual) / p.cycle_life * 100
        predictions.append(p)
    metric_df = pd.DataFrame(metrics)
    metric_df.to_csv(out / "evaluation_metrics.csv", index=False)
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(out / "predictions.csv", index=False)
    predictions.query("batch == 'B2'").nlargest(5, "APE_pct").to_csv(out / "largest_errors_batch2.csv", index=False)
    cv_mape = float(compare.iloc[0].CV_MAPE_mean_pct)
    valid_mape, test_mape, ext_mape = metric_df.MAPE_pct.tolist()
    table = pd.DataFrame([
        ["Train (Batch 1 CV)", cv_mape, "%", "B1 train only; mean of 5 nested GroupKFold outer folds"],
        ["Valid (Batch 1 Hold-out)", valid_mape, "%", f"{len(valid)} cells; unseen charging-protocol groups"],
        ["Test (Batch 2)", test_mape, "%", "Frozen B1-training model; no test tuning"],
        ["Gap (Train-Valid)", valid_mape-cv_mape, "pp", "Valid minus CV; positive means error increased"],
        ["Gap (Valid-Test)", test_mape-valid_mape, "pp", "Test minus Valid; positive means error increased"],
        ["Gap (Target-Test)", test_mape-9.1, "pp", "Test minus assignment target 9.1%"],
        ["External (Batch 3)", ext_mape, "%", "Same frozen model; optional external evaluation"],
        ["Gap (Batch2-Batch3)", ext_mape-test_mape, "pp", "Batch3 minus Batch2"],
    ], columns=["Index", "MAPE_or_Gap", "Unit", "Note"])
    table.to_csv(out / "model_performance.csv", index=False)
    summary = {**frozen, "train_n": len(train), "valid_n": len(valid),
               "train_protocols": train.policy_group.nunique(), "valid_protocols": valid.policy_group.nunique(),
               "cv_mape": cv_mape, "cv_sd": float(compare.iloc[0].CV_MAPE_sd_pp),
               "valid_mape": valid_mape, "test_mape": test_mape, "external_mape": ext_mape,
               "target": 9.1, "gap_target_test": test_mape-9.1,
               "python": platform.python_version(), "sklearn": sklearn.__version__,
               "elapsed_seconds": round(time.time()-start, 2)}
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(table.to_string(index=False), flush=True)
    return summary, compare, table

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features.csv")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    run_training(args.features, args.results)
