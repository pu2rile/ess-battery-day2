"""Post-evaluation diagnostics. Never replace or tune the frozen selected model."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from src.train import candidates, score

def diagnose(project):
    project = Path(project)
    out = project / "results"
    df = pd.read_csv(project / "data/processed/features.csv")
    spec = json.loads((out / "frozen_model_spec.json").read_text())
    train = df[df.cell_uid.isin(spec["trained_on"])]
    valid = df[df.cell_uid.isin(spec["valid_cells"])]
    # These two fixed, untuned references existed before seeing test scores.
    # Their post-hoc comparison is descriptive, not a new model-selection stage.
    rows, predictions = [], []
    for name, cols, estimator, grid in candidates():
        if name not in ["DummyMedian", "LogLinear_DeltaQ"]:
            continue
        assert not grid
        fitted = clone(estimator).fit(train[cols], train.cycle_life)
        for split, sub in [("B1_valid", valid), ("B2_test", df.query("batch == 'B2'")),
                           ("B3_external", df.query("batch == 'B3'"))]:
            pred = fitted.predict(sub[cols])
            rows.append({"model": name, "split": split, "n": len(sub), **score(sub.cycle_life, pred)})
            for uid, actual, yp in zip(sub.cell_uid, sub.cycle_life, pred):
                predictions.append({"model": name, "split": split, "cell_uid": uid,
                                    "actual": actual, "prediction": float(yp)})
    pd.DataFrame(rows).to_csv(out / "reference_baselines.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "reference_predictions.csv", index=False)
    ranges = []
    for feature in spec["features"]:
        low, high = train[feature].min(), train[feature].max()
        for batch in ["B2", "B3"]:
            sub = df.query("batch == @batch")
            ranges.append({"feature": feature, "batch": batch, "train_min": low, "train_max": high,
                           "external_min": sub[feature].min(), "external_max": sub[feature].max(),
                           "outside_train_range_n": int(((sub[feature] < low) | (sub[feature] > high)).sum()),
                           "n": len(sub)})
    pd.DataFrame(ranges).to_csv(out / "feature_shift.csv", index=False)
    artifact = joblib.load(out / "final_model.joblib")
    fitted = artifact["model"]
    if hasattr(fitted, "regressor_") and hasattr(fitted.regressor_.named_steps["model"], "coef_"):
        pipe = fitted.regressor_
        transformed = pipe[:-1].transform(df[spec["features"]])
        coef = pipe.named_steps["model"].coef_
        names = pipe[:-1].get_feature_names_out(spec["features"])
        pd.DataFrame({"feature": names, "coefficient_standardized_log10_target": coef}).to_csv(out / "model_coefficients.csv", index=False)
        contributions = pd.DataFrame(transformed * coef, columns=names)
        contributions.insert(0, "batch", df.batch)
        contributions.insert(0, "cell_uid", df.cell_uid)
        contributions.to_csv(out / "log10_prediction_contributions.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sets = [train, valid, df.query("batch == 'B2'"), df.query("batch == 'B3'")]
    axes[0].boxplot([s.charge_mean for s in sets], labels=["B1 train", "B1 valid", "B2", "B3"])
    axes[0].set(ylabel="Mean charge time (dataset units)", title="A shifted input: charge time")
    low, high = train.charge_mean.min(), train.charge_mean.max()
    axes[0].axhspan(low, high, color="#2878B5", alpha=.1, label="B1 training range")
    axes[0].legend(frameon=False, fontsize=8)
    baselines = pd.DataFrame(rows).query("split == 'B2_test'")
    selected = json.loads((out / "run_summary.json").read_text())
    names = ["Selected ElasticNet", "Median reference", "Delta-Q reference"]
    scores = [selected["test_mape"],
              float(baselines.query("model == 'DummyMedian'").MAPE_pct.iloc[0]),
              float(baselines.query("model == 'LogLinear_DeltaQ'").MAPE_pct.iloc[0])]
    axes[1].barh(names, scores, color=["#D9534F", "#9BB9D0", "#2A9D8F"])
    axes[1].axvline(9.1, color="#777777", linestyle="--", label="Assignment target: 9.1%")
    axes[1].set(xlabel="Batch 2 MAPE (%)", title="Fixed reference models (diagnostic only)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures/06_shift_and_references.png", dpi=160)
    plt.close(fig)
    return pd.DataFrame(rows), pd.DataFrame(ranges)

if __name__ == "__main__":
    refs, shift = diagnose(Path(__file__).resolve().parents[1])
    print(refs.to_string(index=False))
