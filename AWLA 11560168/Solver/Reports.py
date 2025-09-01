"""
Solver/Reports.py
=================
Reporting utilities for AWLA:
- WAM band table (per-staff, per-team, per-grade)
- Fairness summaries (by team/grade)
- Stability vs baseline (if provided)
- Splits per module
- NEW KPIs: NMD, RMSE by cohort, % unchanged assignments, total hours changed
- Multi-format export: CSV + Excel + JSON

Usage (CLI):
    python -m Solver.Reports \
      --instance Dataset/instances/awla_synth_instance_v1 \
      --config Experiments/runs/demo/resolved_config.yaml \
      --assignment Experiments/runs/demo/assignment.csv \
      --out_dir Experiments/runs/demo/report

Author: AWLA Team
License: MIT
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

from Solver.Problem import Problem


# --------------------------
# helpers
# --------------------------
def _safe_ratio(n: float, d: float) -> float:
    if d == 0:
        return 0.0 if n == 0 else float("inf")
    return float(n) / float(d)


def _band_of(
    ratio: float, green: tuple[float, float], amber: tuple[float, float]
) -> str:
    lo_g, hi_g = green
    lo_a, hi_a = amber
    if np.isinf(ratio):
        return "RED"
    if lo_g <= ratio <= hi_g:
        return "GREEN"
    if lo_a <= ratio <= hi_a:
        return "AMBER"
    return "RED"


def _ensure_out(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


# --------------------------
# main
# --------------------------
def generate_reports(
    instance_dir: Path | str,
    resolved_config_path: Path | str,
    assignment_csv: Path | str,
    out_dir: Path | str,
) -> Dict[str, Any]:
    """Generate all reports and export to CSV/Excel/JSON. Returns a small dict summary."""
    out_dir = Path(out_dir)
    _ensure_out(out_dir)

    cfg = yaml.safe_load(Path(resolved_config_path).read_text())
    P = Problem(Path(instance_dir), cfg).load()

    # Load assignment
    A = pd.read_csv(assignment_csv)
    required_cols = {"module_id", "duty_id", "staff_id", "hours_assigned"}
    if not required_cols.issubset(set(A.columns)):
        raise ValueError(f"assignment.csv must have columns: {required_cols}")

    # Dataframes
    staff = P.get_dataframe("staff")
    modules = P.get_dataframe("modules")
    duties = P.get_dataframe("duties")
    baseline = P.get_dataframe("baseline_plan")  # may be empty

    # Band thresholds (config key: reports.band_green / reports.band_amber)
    rpt = cfg.get("reports", {})
    green = tuple(rpt.get("band_green", [0.95, 1.05]))
    amber = tuple(rpt.get("band_amber", [0.90, 1.10]))

    # --------------------------
    # per-staff loads, deltas, bands
    # --------------------------
    targets_map = P.targets  # dict: staff_id -> target_hours
    loads_series = A.groupby("staff_id")["hours_assigned"].sum()

    staff_cols = ["staff_id", "name", "contract_type", "fte", "team", "grade", "campus"]
    for c in staff_cols:
        if c not in staff.columns:
            staff[c] = ""  # ensure column exists

    staff_df = staff[staff_cols].copy()
    staff_df["target_hours"] = (
        staff_df["staff_id"].map(targets_map).astype(float).fillna(0.0)
    )
    staff_df["load_hours"] = (
        loads_series.reindex(staff_df["staff_id"]).astype(float).fillna(0.0).values
    )
    staff_df["delta_hours"] = staff_df["load_hours"] - staff_df["target_hours"]
    staff_df["ratio_L_over_T"] = staff_df.apply(
        lambda r: np.nan
        if r["target_hours"] == 0
        else r["load_hours"] / r["target_hours"],
        axis=1,
    )
    staff_df["band"] = staff_df["ratio_L_over_T"].apply(
        lambda r: "N/A" if pd.isna(r) else _band_of(float(r), green, amber)
    )

    staff_report = staff_df[
        [
            "staff_id",
            "name",
            "contract_type",
            "fte",
            "team",
            "grade",
            "campus",
            "target_hours",
            "load_hours",
            "delta_hours",
            "ratio_L_over_T",
            "band",
        ]
    ].copy()

    # round for display
    for c in ["target_hours", "load_hours", "delta_hours", "ratio_L_over_T"]:
        staff_report[c] = staff_report[c].astype(float).round(3)
    staff_report = staff_report.sort_values(["band", "team", "grade", "staff_id"])
    staff_report.to_csv(out_dir / "staff_wam_band_table.csv", index=False)

    # --------------------------
    # aggregates by cohort (team/grade/campus)
    # --------------------------
    def _agg_table(key_col: str) -> pd.DataFrame:
        g = staff_report.groupby(key_col, dropna=False)
        tbl = g.agg(
            staff_count=("staff_id", "nunique"),
            mean_target=("target_hours", "mean"),
            mean_load=("load_hours", "mean"),
            mean_abs_dev=("delta_hours", lambda x: np.abs(x).mean()),
            pct_green=("band", lambda x: (x == "GREEN").mean() * 100.0),
            pct_amber=("band", lambda x: (x == "AMBER").mean() * 100.0),
            pct_red=("band", lambda x: (x == "RED").mean() * 100.0),
        ).reset_index()
        for c in [
            "mean_target",
            "mean_load",
            "mean_abs_dev",
            "pct_green",
            "pct_amber",
            "pct_red",
        ]:
            tbl[c] = tbl[c].astype(float).round(2)
        return tbl.sort_values("pct_green", ascending=False)

    team_summary = _agg_table("team")
    grade_summary = _agg_table("grade")
    campus_summary = _agg_table("campus")
    team_summary.to_csv(out_dir / "fairness_by_team.csv", index=False)
    grade_summary.to_csv(out_dir / "fairness_by_grade.csv", index=False)
    campus_summary.to_csv(out_dir / "fairness_by_campus.csv", index=False)

    # --------------------------
    # module splits
    # --------------------------
    splits = (
        A.groupby("module_id")["staff_id"].nunique().reset_index(name="distinct_staff")
    )
    splits["is_split"] = splits["distinct_staff"].apply(
        lambda k: int(k) - 1 if int(k) > 1 else 0
    )
    splits = splits.merge(
        modules[["module_id", "title", "period", "indivisible"]],
        on="module_id",
        how="left",
    )
    splits.to_csv(out_dir / "module_splits.csv", index=False)

    # --------------------------
    # stability vs baseline + % unchanged + total hours changed
    # --------------------------
    stability_staff = pd.DataFrame(
        columns=["staff_id", "delta_hours_sum", "changed_pairs"]
    )
    stability_long = pd.DataFrame(
        columns=["module_id", "duty_id", "staff_id", "delta_hours"]
    )
    unchanged_pct = None
    total_hours_changed = None

    if isinstance(baseline, pd.DataFrame) and not baseline.empty:
        merged = baseline.merge(
            A,
            on=["module_id", "duty_id", "staff_id"],
            how="outer",
            suffixes=("_base", "_ass"),
        ).fillna(0.0)
        merged["delta_hours"] = (
            merged["hours_assigned_ass"] - merged["hours_assigned_base"]
        ).abs()
        stability_long = merged[
            ["module_id", "duty_id", "staff_id", "delta_hours"]
        ].copy()
        stability_staff = (
            merged.groupby("staff_id")
            .agg(
                delta_hours_sum=("delta_hours", "sum"),
                changed_pairs=("delta_hours", lambda x: (x > 0).sum()),
            )
            .reset_index()
            .sort_values("delta_hours_sum", ascending=False)
        )
        stability_long.to_csv(out_dir / "stability_pairs_long.csv", index=False)
        stability_staff.to_csv(out_dir / "stability_by_staff.csv", index=False)

        # % unchanged assignments (Hamming over union of triples)
        key = ["module_id", "duty_id", "staff_id"]
        B = baseline[key + ["hours_assigned"]].copy()
        C = A[key + ["hours_assigned"]].copy()
        B["assigned_base"] = (B["hours_assigned"] > 0.0).astype(int)
        C["assigned"] = (C["hours_assigned"] > 0.0).astype(int)
        union = pd.concat([B[key], C[key]], ignore_index=True).drop_duplicates()
        M = (
            union.merge(B[key + ["assigned_base"]], on=key, how="left")
            .merge(C[key + ["assigned"]], on=key, how="left")
            .fillna({"assigned_base": 0, "assigned": 0})
        )
        same = int((M["assigned_base"] == M["assigned"]).sum())
        unchanged_pct = 100.0 * same / float(len(M)) if len(M) else None
        total_hours_changed = float(stability_staff["delta_hours_sum"].sum())

    # --------------------------
    # KPIs: L1, NMD, RMSE by cohort
    # --------------------------
    eps = 1e-6
    # L1 fairness (sum of absolute deviations)
    L1_fairness = float(np.abs(staff_df["delta_hours"]).sum())

    # NMD: mean over staff of |delta| / (|target| + eps)
    staff_df["rel_dev"] = np.abs(staff_df["delta_hours"]) / (
        np.abs(staff_df["target_hours"]) + eps
    )
    NMD = float(staff_df["rel_dev"].mean()) if len(staff_df) else float("nan")

    # RMSE by team / grade on (L - T)
    def _rmse(series: pd.Series) -> float:
        x = np.asarray(series, dtype=float)
        return float(np.sqrt((x**2).mean())) if x.size else 0.0

    rmse_by_team = (
        staff_df.groupby("team", dropna=False)["delta_hours"]
        .apply(_rmse)
        .reset_index(name="rmse_delta")
    )
    rmse_by_grade = (
        staff_df.groupby("grade", dropna=False)["delta_hours"]
        .apply(_rmse)
        .reset_index(name="rmse_delta")
    )
    rmse_by_team.to_csv(out_dir / "rmse_by_team.csv", index=False)
    rmse_by_grade.to_csv(out_dir / "rmse_by_grade.csv", index=False)

    # --------------------------
    # Excel workbook
    # --------------------------
    xlsx = out_dir / "report.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xl:
        # Summary sheet (headline KPIs)
        total_required = float(duties["hours_required"].sum())
        total_assigned = float(A["hours_assigned"].sum())
        green_pct = round((staff_report["band"] == "GREEN").mean() * 100.0, 2)
        amber_or_green_pct = round((staff_report["band"] != "RED").mean() * 100.0, 2)
        kpis = pd.DataFrame(
            {
                "metric": [
                    "total_required_hours",
                    "total_assigned_hours",
                    "staff_count",
                    "%GREEN",
                    "%AMBER_or_Green",
                    "L1_fairness_hours",
                    "NMD",
                    "%unchanged_assignments",
                    "total_hours_changed",
                ],
                "value": [
                    total_required,
                    total_assigned,
                    int(staff_report.shape[0]),
                    green_pct,
                    amber_or_green_pct,
                    round(L1_fairness, 2),
                    None if np.isnan(NMD) else round(NMD, 4),
                    None if unchanged_pct is None else round(float(unchanged_pct), 2),
                    None
                    if total_hours_changed is None
                    else round(float(total_hours_changed), 2),
                ],
            }
        )
        kpis.to_excel(xl, sheet_name="Summary", index=False)

        # Existing sheets
        staff_report.to_excel(xl, sheet_name="StaffBands", index=False)
        team_summary.to_excel(xl, sheet_name="TeamFairness", index=False)
        grade_summary.to_excel(xl, sheet_name="GradeFairness", index=False)
        campus_summary.to_excel(xl, sheet_name="CampusFairness", index=False)
        splits.to_excel(xl, sheet_name="ModuleSplits", index=False)
        if not stability_staff.empty:
            stability_staff.to_excel(xl, sheet_name="StabilityByStaff", index=False)
            stability_long.to_excel(xl, sheet_name="StabilityPairs", index=False)
        # NEW: RMSE detail sheets
        rmse_by_team.to_excel(xl, sheet_name="RMSE_Team", index=False)
        rmse_by_grade.to_excel(xl, sheet_name="RMSE_Grade", index=False)

    # --------------------------
    # JSON summary (write both names for compatibility)
    # --------------------------
    summary = {
        "staff_count": int(staff_report.shape[0]),
        "pct_green": float(round((staff_report["band"] == "GREEN").mean() * 100.0, 2)),
        "pct_amber_or_green": float(
            round((staff_report["band"] != "RED").mean() * 100.0, 2)
        ),
        "L1_fairness_hours": round(L1_fairness, 2),
        "NMD": None if np.isnan(NMD) else round(NMD, 4),
        "unchanged_pct": None
        if unchanged_pct is None
        else round(float(unchanged_pct), 2),
        "total_hours_changed": None
        if total_hours_changed is None
        else round(float(total_hours_changed), 2),
        "team_best": team_summary.sort_values("pct_green", ascending=False)
        .head(1)
        .to_dict(orient="records"),
        "grade_best": grade_summary.sort_values("pct_green", ascending=False)
        .head(1)
        .to_dict(orient="records"),
        "has_baseline": bool(isinstance(baseline, pd.DataFrame) and not baseline.empty),
    }
    (out_dir / "report_summary.json").write_text(json.dumps(summary, indent=2))
    # also write a generic name some of your analysis scripts expect
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # --------------------------
    # convenience exports: staff loads & module coverage (kept for backward compat)
    # --------------------------
    staff_loads = staff_report[
        [
            "staff_id",
            "name",
            "team",
            "grade",
            "campus",
            "target_hours",
            "load_hours",
            "delta_hours",
            "ratio_L_over_T",
            "band",
        ]
    ].copy()
    staff_loads.to_csv(out_dir / "staff_loads.csv", index=False)

    required = (
        duties.groupby(["module_id", "duty_id"])["hours_required"].sum().reset_index()
    )
    assigned = A.groupby(["module_id", "duty_id"])["hours_assigned"].sum().reset_index()
    modcov = required.merge(assigned, on=["module_id", "duty_id"], how="left").fillna(
        {"hours_assigned": 0.0}
    )
    modcov["coverage_ok"] = modcov["hours_assigned"].round(6) == modcov[
        "hours_required"
    ].round(6)
    modcov.to_csv(out_dir / "module_coverage.csv", index=False)

    # --------------------------
    # objective parity check (optional)
    # --------------------------
    try:
        from Solver.Optimisation import Evaluator, Problem as _Problem

        P2 = _Problem(Path(instance_dir), cfg).load()
        eval2 = Evaluator(P2)
        recomputed = eval2.objective(A)
        prior_path = out_dir.parent / "objective_breakdown.json"
        if prior_path.exists():
            prior = json.loads(prior_path.read_text())
            eps_chk = 1e-6
            keys = sorted(set(recomputed.keys()) & set(prior.keys()))
            diffs = {k: float(recomputed[k]) - float(prior[k]) for k in keys}
            ok = all(abs(diffs[k]) <= eps_chk for k in keys)
            summary["objective_parity"] = {
                "status": ("ok" if ok else "mismatch"),
                "diffs": diffs,
            }
        else:
            summary["objective_parity"] = {"status": "no_prior"}
    except Exception as e:
        summary["objective_parity"] = {"status": "error", "error": str(e)}

    return summary


# --------------------------
# CLI
# --------------------------
def _parse_args(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="AWLA Reporting (WAM bands, fairness by team/grade, stability, KPIs)"
    )
    ap.add_argument("--instance", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--assignment", required=True)
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    summary = generate_reports(
        args.instance, args.config, args.assignment, args.out_dir
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
