
"""
AWLA Checker (end-to-end)

Reads:
- Dataset/instances/<instance>/*.csv  (staff, modules, duties, ...)
- Resolved config (YAML)
- assignment.csv  (module_id, duty_id, staff_id, hours_assigned)

Outputs:
- violations.csv (hard & soft entries with magnitudes and costs)
- objective_breakdown.json (term-by-term totals and weighted sum)
- report.md (human summary)
"""

from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import yaml

def load_csvs(inst_dir: Path) -> Dict[str, pd.DataFrame]:
    def rd(name): return pd.read_csv(inst_dir/name)
    data = {
        "staff": rd("staff.csv"),
        "modules": rd("modules.csv"),
        "duties": rd("duties.csv"),
        "elig": rd("eligibility.csv"),
        "avail": rd("availability.csv"),
        "prefs": rd("preferences.csv"),
        "conf": rd("conflicts_staff_pairs.csv"),
        "prealloc": rd("preallocations.csv") if (inst_dir/"preallocations.csv").exists() else pd.DataFrame(columns=["staff_id","category","hours","notes"]),
        "fair_over": rd("fairness_targets.csv") if (inst_dir/"fairness_targets.csv").exists() else pd.DataFrame(columns=["staff_id","target_hours_override"]),
        "baseline": rd("baseline_plan.csv") if (inst_dir/"baseline_plan.csv").exists() else pd.DataFrame(columns=["module_id","duty_id","staff_id","hours_assigned"]),
    }
    return data

def compute_targets(cfg: Dict[str,Any], data: Dict[str,pd.DataFrame]) -> Dict[str, float]:
    staff = data["staff"].copy()
    # base targets by contract
    base_TR = float(cfg["wam_adapter"]["default_target_hours"]["TR"])
    base_TS = float(cfg["wam_adapter"]["default_target_hours"]["TS"])
    base_map = {"TR": base_TR, "TS": base_TS}
    staff["base"] = staff["contract_type"].map(base_map).fillna(base_TR)
    staff["target"] = staff["base"] * staff["fte"]
    # apply person overrides
    if not data["fair_over"].empty:
        ov = data["fair_over"].dropna()
        if "target_hours_override" in ov.columns:
            ov = ov.set_index("staff_id")["target_hours_override"].to_dict()
            staff["target"] = staff.apply(lambda r: ov.get(r["staff_id"], r["target"]), axis=1)
    # credit preallocations if configured
    if cfg.get("wam_adapter",{}).get("preallocations_mode","credit") == "credit":
        pre = data["prealloc"].groupby("staff_id")["hours"].sum().to_dict()
        staff["target"] = staff.apply(lambda r: r["target"] - pre.get(r["staff_id"], 0.0), axis=1)
    # sanity: lower bound at zero
    staff["target"] = staff["target"].clip(lower=0.0)
    return dict(zip(staff["staff_id"], staff["target"]))

def assignment_to_loads(ass: pd.DataFrame) -> Dict[str,float]:
    return ass.groupby("staff_id")["hours_assigned"].sum().to_dict()

def split_count_per_module(ass: pd.DataFrame) -> Dict[str,int]:
    # number of distinct staff per module minus 1
    return (ass.groupby("module_id")["staff_id"].nunique() - 1).clip(lower=0).to_dict()

def make_period_map(modules: pd.DataFrame) -> Dict[str,str]:
    return dict(zip(modules["module_id"], modules["period"]))

def availability_value(avail: pd.DataFrame, staff_id: str, period: str) -> float:
    r = avail[(avail["staff_id"]==staff_id) & (avail["period"]==period)]
    if r.empty: return 1.0
    return float(r["availability"].iloc[0])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_dir", required=True)
    ap.add_argument("--resolved_config", required=True)
    ap.add_argument("--assignment_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    inst_dir = Path(args.instance_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(Path(args.resolved_config).read_text())
    data = load_csvs(inst_dir)

    # Load assignment
    ass = pd.read_csv(args.assignment_csv)
    required_cols = {"module_id","duty_id","staff_id","hours_assigned"}
    if not required_cols.issubset(set(ass.columns)):
        raise ValueError(f"assignment.csv must have columns: {required_cols}")

    # Prep lookups
    modules = data["modules"]; duties = data["duties"]; staff = data["staff"]
    elig = data["elig"]; avail = data["avail"]; prefs = data["prefs"]; conf = data["conf"]
    period_of = make_period_map(modules)
    hours_required = duties.set_index("duty_id")["hours_required"].to_dict()
    module_of_duty = duties.set_index("duty_id")["module_id"].to_dict()
    indivisible_modules = set(modules.loc[modules["indivisible"]==1, "module_id"].tolist())
    max_hours = staff.set_index("staff_id")["max_hours"].fillna(float("inf")).to_dict()
    targets = compute_targets(cfg, data)

    # Eligibility map (duty level). If empty in data, allow all.
    elig_map = {(row["staff_id"], row["duty_id"]): int(row["eligible"]) for _,row in elig.iterrows()}
    def is_eligible(sid: str, did: str) -> bool:
        key = (sid, did)
        if key in elig_map:
            return elig_map[key] == 1
        # if not present, fall back to module-level eligibility rows (staff,module, empty duty)
        mod = module_of_duty.get(did, "")
        module_rows = elig[(elig["staff_id"]==sid) & (elig["module_id"]==mod) & (elig["duty_id"].fillna("")=="")]
        if not module_rows.empty:
            return int(module_rows["eligible"].iloc[0]) == 1
        # default allow if no eligibility specified at all
        return True

    # Build violations list
    rows = []
    def add(code, kind, entity, magnitude, weight, cost, detail):
        rows.append({"code":code, "type":kind, "entity":entity, "magnitude":magnitude, "weight":weight, "cost":cost, "detail":detail})

    # ---------------- HARD: H1 Coverage ----------------
    assigned_sum = ass.groupby("duty_id")["hours_assigned"].sum().to_dict()
    for did, H in hours_required.items():
        got = float(assigned_sum.get(did, 0.0))
        delta = round(got - H, 6)
        if abs(delta) > 1e-6:
            add("H1_COVERAGE","HARD", did, delta, None, None, f"Assigned {got}, required {H}")

    # ---------------- HARD: H2 Eligibility ----------------
    for i,row in ass.iterrows():
        sid = row["staff_id"]; did = row["duty_id"]; hrs = float(row["hours_assigned"])
        if hrs <= 0: continue
        if not is_eligible(sid, did):
            add("H2_ELIGIBILITY","HARD", f"{sid}|{did}", hrs, None, None, "Assigned but not eligible")

    # ---------------- HARD: H3 Capacity ----------------
    loads = assignment_to_loads(ass)
    for sid, L in loads.items():
        cap = float(max_hours.get(sid, float("inf")))
        if L - cap > 1e-6:
            add("H3_CAPACITY","HARD", sid, round(L-cap,6), None, None, f"Load {L} exceeds cap {cap}")

    # ---------------- HARD: H4 Availability zero ----------------
    for i,row in ass.iterrows():
        sid = row["staff_id"]; did = row["duty_id"]; hrs = float(row["hours_assigned"])
        if hrs <= 0: continue
        mod = module_of_duty.get(did, "")
        period = period_of.get(mod, "")
        ava = availability_value(avail, sid, period)
        if ava == 0.0:
            add("H4_UNAVAILABLE","HARD", f"{sid}|{did}", hrs, None, None, f"Availability is 0 in period {period}")

    # ---------------- HARD: H5 Duty integrity for indivisible modules ----------------
    if len(indivisible_modules) > 0:
        assign_group = ass.groupby(["module_id","staff_id"])["hours_assigned"].sum().reset_index()
        for m in indivisible_modules:
            sub = assign_group[assign_group["module_id"]==m]
            staff_count = (sub["hours_assigned"]>0).sum()
            if staff_count > 1:
                add("H5_INDIVISIBLE","HARD", m, staff_count, None, None, "Module marked indivisible but assigned to multiple staff")

    # ---------------- HARD: H7 Conflicts (cannot_co_teach) ----------------
    hard_pairs = set()
    for _,r in conf.iterrows():
        if str(r.get("conflict_type","")).strip() == "cannot_co_teach":
            a = r["staff_id_1"]; b = r["staff_id_2"]
            hard_pairs.add(tuple(sorted([a,b])))
    if hard_pairs:
        mod_staff = ass.groupby(["module_id","staff_id"])["hours_assigned"].sum().reset_index()
        for m, df in mod_staff.groupby("module_id"):
            staff_list = set(df.loc[df["hours_assigned"]>0, "staff_id"].tolist())
            for a in staff_list:
                for b in staff_list:
                    if a < b and (a,b) in hard_pairs:
                        add("H7_CONFLICT_PAIR","HARD", f"{m}|{a}&{b}", 1, None, None, "Hard conflict pair co-teaching same module")

    # ---------------- SOFT TERMS ----------------
    W = cfg["weights"]
    # S1 Fairness L1
    fairness_cost = 0.0
    fairness_rows = []
    for sid in staff["staff_id"]:
        L = float(loads.get(sid, 0.0))
        T = float(compute_targets(cfg, data).get(sid, 0.0))
        resid = L - T
        mag = abs(resid)
        cost = W["fairness_L1"] * mag
        fairness_cost += cost
        fairness_rows.append((sid, mag, W["fairness_L1"], cost, f"L={L:.2f}, T={T:.2f}"))
    for sid, mag, w, c, detail in fairness_rows:
        add("S1_FAIRNESS","SOFT", sid, round(mag,6), w, round(c,6), detail)

    # S2 Preferences (per-hour penalties/rewards)
    pref_cost = 0.0
    if not prefs.empty:
        # Build helpers
        duty_to_mod = module_of_duty
        period_map = period_of
        # For faster lookups build assignment expansions with module and period
        aa = ass.copy()
        aa["module_id"] = aa["module_id"]
        aa["period"] = aa["module_id"].map(period_map)
        # staff_module
        sm = prefs[prefs["pref_type"]=="staff_module"]
        for _,r in sm.iterrows():
            sid, mid, pen = r["staff_id"], r["module_id"], float(r["penalty"])
            hrs = aa[(aa["staff_id"]==sid) & (aa["module_id"]==mid)]["hours_assigned"].sum()
            if hrs != 0:
                cost = pen * hrs * W["pref"]
                pref_cost += cost
                add("S2_PREF_SM","SOFT", f"{sid}|{mid}", round(hrs,6), W["pref"], round(cost,6), f"pen={pen}")
        # staff_duty
        sd = prefs[prefs["pref_type"]=="staff_duty"]
        for _,r in sd.iterrows():
            sid, did, pen = r["staff_id"], r["duty_id"], float(r["penalty"])
            hrs = aa[(aa["staff_id"]==sid) & (aa["duty_id"]==did)]["hours_assigned"].sum()
            if hrs != 0:
                cost = pen * hrs * W["pref"]
                pref_cost += cost
                add("S2_PREF_SD","SOFT", f"{sid}|{did}", round(hrs,6), W["pref"], round(cost,6), f"pen={pen}")
        # staff_period
        sp = prefs[prefs["pref_type"]=="staff_period"]
        for _,r in sp.iterrows():
            sid, per, pen = r["staff_id"], r["period"], float(r["penalty"])
            hrs = aa[(aa["staff_id"]==sid) & (aa["period"]==per)]["hours_assigned"].sum()
            if hrs != 0:
                cost = pen * hrs * W["pref"]
                pref_cost += cost
                add("S2_PREF_SP","SOFT", f"{sid}|{per}", round(hrs,6), W["pref"], round(cost,6), f"pen={pen}")

    # S3 Team balance (L1 around team mean)
    team_of = staff.set_index("staff_id")["team"].to_dict()
    team_cost = 0.0
    for team, sids in staff.groupby("team")["staff_id"]:
        Ls = [float(loads.get(s, 0.0)) for s in sids]
        if len(Ls)==0: continue
        meanL = float(np.mean(Ls))
        disp = sum(abs(L-meanL) for L in Ls)
        cost = W["team_balance"] * disp
        team_cost += cost
        # add per-staff entries
        for s in sids:
            L = float(loads.get(s, 0.0))
            add("S3_TEAM_BAL","SOFT", f"{team}|{s}", round(abs(L-meanL),6), W["team_balance"], round(W["team_balance"]*abs(L-meanL),6), f"team_mean={meanL:.2f}")

    # S4 Splitting penalty
    split_counts = split_count_per_module(ass)
    split_cost = 0.0
    for m, sc in split_counts.items():
        if sc > 0:
            c = W["split"] * sc
            split_cost += c
            add("S4_SPLIT","SOFT", m, int(sc), W["split"], round(c,6), "extra staff beyond 1")

    # S5 Stability vs baseline (eligible-only if configured)
    stability_cost = 0.0
    base = data["baseline"]
    if not base.empty:
        include_only_elig = cfg.get("checker",{}).get("baseline_stability_include_only_eligible", True)
        # Merge baseline with assignment, compute delta hours
        merged = base.merge(ass, on=["module_id","duty_id","staff_id"], how="outer", suffixes=("_base","_ass")).fillna(0.0)
        merged["delta"] = (merged["hours_assigned_ass"] - merged["hours_assigned_base"]).abs()
        for _, r in merged.iterrows():
            sid, did = r["staff_id"], r["duty_id"]
            if include_only_elig:
                # reuse eligibility check
                # If there is no duty id (NaN from outer merge), skip
                if not isinstance(did, str) or did=="":
                    continue
                if not is_eligible(sid, did):
                    continue
            if r["delta"] > 0:
                c = W["stability"] * float(r["delta"])
                stability_cost += c
                add("S5_STABILITY","SOFT", f"{sid}|{did}", round(float(r["delta"]),6), W["stability"], round(c,6), "")

    # S9 Soft conflicts (line_manager_pairing_avoid, personal_conflict)
    soft_cost = 0.0
    soft_types = set(["line_manager_pairing_avoid","personal_conflict"])
    if not data["conf"].empty:
        mod_staff = ass.groupby(["module_id","staff_id"])["hours_assigned"].sum().reset_index()
        for m, df in mod_staff.groupby("module_id"):
            staff_list = set(df.loc[df["hours_assigned"]>0, "staff_id"].tolist())
            for _, r in data["conf"].iterrows():
                ctype = str(r.get("conflict_type","")).strip()
                if ctype in soft_types:
                    a, b = r["staff_id_1"], r["staff_id_2"]
                    if a in staff_list and b in staff_list:
                        w = float(r.get("weight", 1.0))
                        cost = w  # one unit per module where co-teaching occurs
                        soft_cost += cost
                        add("S9_SOFT_CONFLICT","SOFT", f"{m}|{a}&{b}", 1, w, cost, ctype)

    # Objective breakdown
    breakdown = {
        "fairness_L1": round(fairness_cost, 6),
        "preferences": round(pref_cost, 6),
        "team_balance": round(team_cost, 6),
        "split": round(split_cost, 6),
        "stability": round(stability_cost, 6),
        "soft_conflicts": round(soft_cost, 6),
    }
    breakdown["weighted_sum"] = round(sum(breakdown.values()), 6)

    # Write outputs
    vio = pd.DataFrame(rows)
    vio.to_csv(out_dir/"violations.csv", index=False)
    (out_dir/"objective_breakdown.json").write_text(json.dumps(breakdown, indent=2))

    # Human report
    hard = vio[vio["type"]=="HARD"]
    soft = vio[vio["type"]=="SOFT"]
    lines = []
    lines.append("# AWLA Checker Report\n")
    lines.append(f"- Hard violations: **{hard.shape[0]}**")
    lines.append(f"- Soft entries: **{soft.shape[0]}**")
    lines.append("## Objective breakdown")
    for k,v in breakdown.items():
        lines.append(f"- {k}: **{v}**")
    (out_dir/"report.md").write_text("\n".join(lines))

    print(f"Wrote: {out_dir/'violations.csv'}")
    print(f"Wrote: {out_dir/'objective_breakdown.json'}")
    print(f"Wrote: {out_dir/'report.md'}")

if __name__ == "__main__":
    main()
