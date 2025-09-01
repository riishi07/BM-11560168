#!/usr/bin/env python3
# Scripts/finalise_results.py

import json, re, math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RUNS_ROOT = Path("Experiments/runs")
OUT_DIR = Path("Experiments")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def find_runs():
    rows = []
    for summ in RUNS_ROOT.glob("**/report/summary.json"):
        run_dir = summ.parent.parent
        name = run_dir.name
        d = json.loads(summ.read_text())
        # parse method/weight/seed from folder name
        m = "unknown"
        if "baseline" in name:
            m = "baseline"
        elif "ilp_only" in name:
            m = "ilp_only"
        elif "ilp_lns" in name:
            m = "ilp_lns"

        w = re.search(r"w(\d+)", name)
        s = re.search(r"seed(\d+)", name)
        rows.append(
            {
                "run_dir": str(run_dir),
                "name": name,
                "method": m,
                "weight_stability": int(w.group(1)) if w else None,
                "seed": int(s.group(1)) if s else None,
                "L1": d.get("L1_fairness_hours"),
                "NMD": d.get("NMD"),
                "unchanged_pct": d.get("unchanged_pct"),
                "hours_changed": d.get("total_hours_changed"),
                "pct_green": d.get("pct_green"),
                "pct_amber_or_green": d.get("pct_amber_or_green"),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["method", "weight_stability", "seed", "name"]
    )


def normal_ci(series, alpha=0.05):
    x = np.asarray(series, dtype=float)
    n = x.size
    if n == 0:
        return (np.nan, np.nan, np.nan)
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    z = 1.96  # normal approx; fine for n=4
    half = z * (sd / math.sqrt(n)) if n > 1 else 0.0
    return mean, mean - half, mean + half


def save_frontier_seed42(df):
    f = df[(df["method"] == "ilp_lns") & (df["seed"] == 42)]
    f = f.dropna(subset=["NMD", "unchanged_pct", "weight_stability"])
    f = f.sort_values("weight_stability")
    f[["weight_stability", "NMD", "unchanged_pct"]].to_csv(
        OUT_DIR / "frontier_seed42.csv", index=False
    )
    if f.empty:
        return
    plt.figure()
    plt.scatter(f["NMD"], f["unchanged_pct"])
    for _, r in f.iterrows():
        plt.annotate(
            f"w={int(r['weight_stability'])}",
            (r["NMD"], r["unchanged_pct"]),
            xytext=(3, 3),
            textcoords="offset points",
        )
    plt.xlabel("NMD (lower is better)")
    plt.ylabel("% unchanged assignments (higher is better)")
    plt.title("Equity–Stability Frontier (seed=42)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_frontier_seed42.png", dpi=300)
    plt.close()


def save_convergence_plot():
    # pick w=3, seed=42
    hist = RUNS_ROOT / "ilp_lns_w3_seed42" / "lns_history.csv"
    if not hist.exists():
        # try to find any lns_history
        cands = list(RUNS_ROOT.glob("**/lns_history.csv"))
        if not cands:
            return
        hist = cands[0]
    df = pd.read_csv(hist)
    if df.empty:
        return
    plt.figure()
    plt.plot(
        df["t"] if "t" in df.columns else df["elapsed_sec"],
        df["best"] if "best" in df.columns else df["best_cost"],
        label="best",
    )
    x = df["t"] if "t" in df.columns else df["elapsed_sec"]
    y = df["cur"] if "cur" in df.columns else df["current_cost"]
    if y is not None:
        plt.plot(x, y, label="current")
    plt.xlabel("Time (s)")
    plt.ylabel("Objective")
    plt.title("LNS Convergence (seed=42, w=3)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_convergence_w3_seed42.png", dpi=300)
    plt.close()


def bar_three_methods(df, metric, title, outname):
    # baseline, ilp_only_seed42, ilp_lns_w3_seed42
    pick = []
    # baseline
    b = df[df["method"] == "baseline"].copy()
    if not b.empty:
        pick.append(("Baseline", float(b.iloc[0][metric])))
    # ilp-only seed 42
    i = df[(df["method"] == "ilp_only") & ((df["seed"].isna()) | (df["seed"] == 42))]
    if not i.empty:
        pick.append(("ILP-only", float(i.iloc[0][metric])))
    # lns w=3 seed 42
    l = df[
        (df["method"] == "ilp_lns") & (df["weight_stability"] == 3) & (df["seed"] == 42)
    ]
    if not l.empty:
        pick.append(("ILP+LNS (w=3)", float(l.iloc[0][metric])))
    if not pick:
        return
    labels, values = zip(*pick)
    plt.figure()
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=0)
    plt.ylabel(metric.replace("_", " "))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIG_DIR / outname, dpi=300)
    plt.close()


def rmse_bars():
    triples = [
        ("baseline_eval", "Baseline"),
        ("ilp_only_seed42", "ILP-only"),
        ("ilp_lns_w3_seed42", "ILP+LNS (w=3)"),
    ]
    # team (use as TR/TS if 'team' encodes that; else still informative)
    dfs = []
    for folder, label in triples:
        p = RUNS_ROOT / folder / "report" / "rmse_by_team.csv"
        if not p.exists():
            continue
        t = pd.read_csv(p)
        t["label"] = label
        dfs.append(t)
    if dfs:
        D = pd.concat(dfs, ignore_index=True)
        # pivot to wide for top few cohorts to avoid clutter
        cohorts = D["team"].dropna().unique().tolist()
        for coh in cohorts:
            sub = D[D["team"] == coh].copy()
            plt.figure()
            plt.bar(sub["label"], sub["rmse_delta"])
            plt.xlabel("")
            plt.ylabel("RMSE of (L−T) [hours]")
            plt.title(f"Cohort fairness — team '{coh}'")
            plt.tight_layout()
            plt.savefig(FIG_DIR / f"fig_rmse_team_{coh.replace(' ', '_')}.png", dpi=300)
            plt.close()
        # summary (all cohorts together) — simple stacked-like bars by saving separate figs is overkill;
        # instead, one combined chart:
        plt.figure()
        wide = D.pivot(index="team", columns="label", values="rmse_delta")
        wide.plot(kind="bar")
        plt.ylabel("RMSE of (L−T) [hours]")
        plt.title("Cohort fairness by team")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig_rmse_team.png", dpi=300)
        plt.close()
    # grade
    dfs = []
    for folder, label in triples:
        p = RUNS_ROOT / folder / "report" / "rmse_by_grade.csv"
        if not p.exists():
            continue
        t = pd.read_csv(p)
        t["label"] = label
        dfs.append(t)
    if dfs:
        D = pd.concat(dfs, ignore_index=True)
        plt.figure()
        wide = D.pivot(index="grade", columns="label", values="rmse_delta")
        wide.plot(kind="bar")
        plt.ylabel("RMSE of (L−T) [hours]")
        plt.title("Cohort fairness by grade")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig_rmse_grade.png", dpi=300)
        plt.close()


def main():
    df = find_runs()
    if df.empty:
        print("No runs found under Experiments/runs/**/report/summary.json")
        return
    df.to_csv(OUT_DIR / "kpis_summary.csv", index=False)
    # CI table for w=3 across seeds
    w3 = df[(df["method"] == "ilp_lns") & (df["weight_stability"] == 3)].dropna(
        subset=["seed"]
    )
    ci_rows = []
    for metric in ["L1", "NMD", "unchanged_pct", "hours_changed"]:
        m, lo, hi = normal_ci(w3[metric].dropna())
        ci_rows.append(
            {
                "metric": metric,
                "mean": round(m, 4),
                "ci95_lo": round(lo, 4),
                "ci95_hi": round(hi, 4),
                "n": int(w3[metric].dropna().shape[0]),
            }
        )
    pd.DataFrame(ci_rows).to_csv(OUT_DIR / "ci_w3.csv", index=False)

    # plots
    save_frontier_seed42(df)
    save_convergence_plot()
    bar_three_methods(
        df, "L1", "L1 fairness (hours) — Baseline vs ILP vs ILP+LNS", "fig_kpi_L1.png"
    )
    bar_three_methods(df, "NMD", "NMD — Baseline vs ILP vs ILP+LNS", "fig_kpi_NMD.png")
    bar_three_methods(
        df,
        "unchanged_pct",
        "% unchanged assignments — Baseline vs ILP vs ILP+LNS",
        "fig_kpi_unchanged.png",
    )
    bar_three_methods(
        df,
        "hours_changed",
        "Total hours changed — Baseline vs ILP vs ILP+LNS",
        "fig_kpi_hours_changed.png",
    )
    rmse_bars()

    print("Wrote:")
    print(" -", OUT_DIR / "kpis_summary.csv")
    print(" -", OUT_DIR / "ci_w3.csv")
    print("Figures in:", FIG_DIR)


if __name__ == "__main__":
    main()
