"""Create figures and descriptive tables from the frozen run."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"B1": "#2878B5", "B2": "#E1812C", "B3": "#2A9D8F"}

def make_figures(project):
    project = Path(project)
    figures = project / "results/figures"
    figures.mkdir(exist_ok=True, parents=True)
    data = pd.read_csv(project / "data/processed/features.csv")
    pred = pd.read_csv(project / "results/predictions.csv")
    comparison = pd.read_csv(project / "results/model_comparison.csv")
    summary = json.loads((project / "results/run_summary.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
    stats = data.groupby("batch").cycle_life.agg(["count", "mean", "median", "min", "max"])
    stats.to_csv(project / "results/batch_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for b, sub in data.groupby("batch"):
        axes[0].hist(sub.cycle_life, bins=np.arange(100, 2601, 200),
                     alpha=.5, color=COLORS[b], label=f"{b} (n={len(sub)})")
        axes[1].scatter(sub.log_dq_var, sub.cycle_life, s=26, alpha=.75,
                        color=COLORS[b], label=b)
    axes[0].set(xlabel="Observed cycle life", ylabel="Number of cells", title="Cohort distributions")
    axes[1].set(xlabel="log10 Var[Q100(V) - Q10(V)]", ylabel="Observed cycle life",
                title="Early discharge-curve change")
    for ax in axes:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "01_data_overview.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ordered = comparison.sort_values("CV_MAPE_mean_pct", ascending=False)
    ax.barh(ordered.model, ordered.CV_MAPE_mean_pct,
            xerr=ordered.CV_MAPE_sd_pp, capsize=3,
            color=["#2A9D8F" if x == summary["model"] else "#9BB9D0" for x in ordered.model])
    ax.set(xlabel="MAPE (%) | error bars = SD across 5 outer folds",
           title="Model selection: B1 training only, nested protocol-group CV")
    fig.tight_layout()
    fig.savefig(figures / "02_model_comparison.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = {"B1": "B1 hold-out", "B2": "B2 final test", "B3": "B3 external"}
    for ax, (b, sub) in zip(axes, pred.groupby("batch")):
        lim = max(sub.cycle_life.max(), sub.prediction.max()) * 1.05
        ax.scatter(sub.cycle_life, sub.prediction, color=COLORS[b], alpha=.8, s=28)
        ax.plot([0, lim], [0, lim], "--", color="#777777", linewidth=1)
        ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Actual cycle life",
                ylabel="Predicted cycle life", title=f"{labels[b]}\nMAPE {sub.APE_pct.mean():.2f}% | n={len(sub)}")
    fig.tight_layout()
    fig.savefig(figures / "03_actual_vs_predicted.png")
    plt.close(fig)
    b2 = pred.query("batch == 'B2'")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(b2.cycle_life, b2.residual, s=32, color=COLORS["B2"])
    axes[0].axhline(0, linestyle="--", color="#777777")
    for offset, (_, r) in zip([(0, 42), (38, 8), (-10, -25)], b2.nlargest(3, "APE_pct").iterrows()):
        axes[0].annotate(r.cell_uid, (r.cycle_life, r.residual), xytext=offset,
                         textcoords="offset points", fontsize=8,
                         arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": .6})
    axes[0].set(xlabel="Actual cycle life", ylabel="Prediction - actual (cycles)", title="Batch 2 residuals")
    worst = b2.nlargest(5, "APE_pct").sort_values("APE_pct")
    axes[1].barh(worst.cell_uid, worst.APE_pct, color=COLORS["B2"])
    axes[1].set(xlabel="Absolute percentage error (%)", title="Batch 2: five largest errors")
    fig.tight_layout()
    fig.savefig(figures / "04_error_analysis.png")
    plt.close(fig)
    b1 = data.query("batch == 'B1'")
    policy = b1.groupby("policy_group").cycle_life.agg(["mean", "count"]).sort_values("mean")
    policy.to_csv(project / "results/b1_policy_summary.csv")
    fig, ax = plt.subplots(figsize=(10, 5.3))
    ax.barh(policy.index, policy["mean"], color=COLORS["B1"], alpha=.8)
    for pos, (_, row) in enumerate(policy.iterrows()):
        ax.text(row["mean"] + 8, pos, f"n={int(row['count'])}", va="center", fontsize=8)
    ax.set(xlabel="Mean cycle life", title="B1 charging protocols (observational, not causal)")
    fig.tight_layout()
    fig.savefig(figures / "05_charging_policy.png")
    plt.close(fig)
    # These descriptions follow evaluation; they do not change the model.
    diagnostics = data.query("batch == 'B2'").merge(b2[["cell_uid", "APE_pct"]], on="cell_uid")
    diagnostics["outside_B1_train_logdq_range"] = (
        (diagnostics.log_dq_var < data[data.cell_uid.isin(summary["trained_on"])].log_dq_var.min()) |
        (diagnostics.log_dq_var > data[data.cell_uid.isin(summary["trained_on"])].log_dq_var.max()))
    diagnostics.to_csv(project / "results/batch2_error_features.csv", index=False)
    return stats

if __name__ == "__main__":
    make_figures(Path(__file__).resolve().parents[1])
