"""Extract one row per cell using measurements from cycles 2--100 only."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import h5py
import numpy as np
import pandas as pd

FILES = {
    "B1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "B2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "B3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
CORE = ["log_dq_var", "log_abs_dq_min", "qd_slope", "ir_mean", "charge_mean"]
EXTENDED = CORE + ["qd_initial", "qd_delta", "ir_delta", "temp_mean", "c1", "q1", "c2"]

def numeric(dataset):
    return np.asarray(dataset, dtype=float).reshape(-1)

def finite_mean(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan

def slope(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() >= 2 else np.nan

def build_features(raw_dir, out_dir):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, audits, manifest = [], [], []
    for batch, filename in FILES.items():
        path = raw_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. See README.md.")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest.append({"batch": batch, "filename": filename,
                         "bytes": path.stat().st_size, "sha256": digest})
        with h5py.File(path, "r") as f:
            b = f["batch"]
            for i in range(b["cycles"].shape[0]):
                uid = f"{batch}C{i}"
                life = float(numeric(f[b["cycle_life"][i, 0]])[0])
                policy = "".join(chr(int(x)) for x in numeric(f[b["policy_readable"][i, 0]]))
                match = re.search(r"([\d.]+)C\(([\d.]+)%\)-([\d.]+)C", policy)
                c1, q1, c2 = map(float, match.groups()) if match else (np.nan,) * 3
                # Group equal numeric charging protocols, even if a suffix differs.
                group = f"{c1:g}C({q1:g}%)-{c2:g}C" if match else policy.strip()
                s = f[b["summary"][i, 0]]
                cycle = numeric(s["cycle"])
                if not np.array_equal(cycle, np.arange(1, len(cycle) + 1)):
                    raise ValueError(f"Cycle-index mapping needs review: {uid}")
                early = (cycle >= 2) & (cycle <= 100)
                xs = cycle[early]
                values = {k: numeric(s[k])[early].copy() for k in
                          ["QDischarge", "IR", "Tavg", "chargetime"]}
                # Fixed domain checks; never estimated using B2/B3 labels.
                qd = values["QDischarge"]
                qd[(qd < .5) | (qd > 1.3)] = np.nan
                ir = values["IR"]
                ir[ir <= 0] = np.nan
                ct = values["chargetime"]
                ct[ct <= 0] = np.nan
                cyc = f[b["cycles"][i, 0]]
                dq = np.array([])
                if cyc["Qdlin"].shape[0] >= 100:
                    q10 = numeric(f[cyc["Qdlin"][9, 0]])
                    q100 = numeric(f[cyc["Qdlin"][99, 0]])
                    voltage = numeric(f[b["Vdlin"][i, 0]])
                    if not (len(q10) == len(q100) == len(voltage)):
                        raise ValueError(f"Voltage-grid length mismatch: {uid}")
                    ok = np.isfinite(q10) & np.isfinite(q100) & np.isfinite(voltage)
                    dq = q100[ok] - q10[ok]
                reasons = []
                if not np.isfinite(life) or life <= 100:
                    reasons.append("missing_or_nonpositive_life_after_cycle100")
                if early.sum() < 99:
                    reasons.append("incomplete_first100")
                if len(dq) < 100 or np.var(dq) <= 0:
                    reasons.append("invalid_deltaq")
                audit = {"cell_uid": uid, "batch": batch, "cycle_life": life,
                         "policy": policy, "policy_group": group,
                         "early_rows": int(early.sum()), "dq_points": len(dq),
                         "included": not reasons, "reason": ";".join(reasons) or "included"}
                audits.append(audit)
                if reasons:
                    continue
                row = {"cell_uid": uid, "batch": batch, "policy": policy,
                       "policy_group": group, "cycle_life": life,
                       "log_dq_var": float(np.log10(np.var(dq) + 1e-12)),
                       "log_abs_dq_min": float(np.log10(abs(dq.min()) + 1e-12)),
                       "qd_initial": finite_mean(qd[xs <= 6]),
                       "qd_delta": finite_mean(qd[xs >= 96]) - finite_mean(qd[xs <= 6]),
                       "qd_slope": slope(xs, qd), "ir_mean": finite_mean(ir),
                       "ir_delta": finite_mean(ir[xs >= 96]) - finite_mean(ir[xs <= 6]),
                       "charge_mean": finite_mean(ct), "temp_mean": finite_mean(values["Tavg"]),
                       "c1": c1, "q1": q1, "c2": c2}
                rows.append(row)
    features, audit = pd.DataFrame(rows), pd.DataFrame(audits)
    assert features.cell_uid.is_unique
    # Exact feature duplication is not proof of physical identity, but catches accidental copies.
    duplicate = features.duplicated(EXTENDED, keep=False)
    if duplicate.any():
        raise ValueError(f"Review duplicate early measurements: {features.loc[duplicate, 'cell_uid'].tolist()}")
    features.to_csv(out_dir / "features.csv", index=False)
    audit.to_csv(out_dir / "cell_audit.csv", index=False)
    (out_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return features, audit

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()
    features, audit = build_features(args.raw_dir, args.out_dir)
    print(audit.groupby("batch").agg(total=("cell_uid", "count"), included=("included", "sum")))
