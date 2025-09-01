
"""
Naive assignment generator (for testing the checker).

Greedy heuristic:
- For each module (respecting indivisible flag):
  - pick an eligible staff with availability>0 and remaining capacity to cover duties,
    preferring staff currently under target.
- Falls back to splitting if needed.
Produces assignment.csv.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

def load_cfg(p: Path) -> dict:
    return yaml.safe_load(p.read_text())

def main(instance_dir: str, resolved_config: str, out_assignment: str):
    inst = Path(instance_dir)
    staff = pd.read_csv(inst/"staff.csv")
    modules = pd.read_csv(inst/"modules.csv")
    duties = pd.read_csv(inst/"duties.csv")
    elig = pd.read_csv(inst/"eligibility.csv")
    avail = pd.read_csv(inst/"availability.csv")
    conf = pd.read_csv(inst/"conflicts_staff_pairs.csv")
    cfg = load_cfg(Path(resolved_config))

    # helper maps
    max_hours = staff.set_index("staff_id")["max_hours"].fillna(float("inf")).to_dict()
    # targets
    base_TR = float(cfg["wam_adapter"]["default_target_hours"]["TR"])
    base_TS = float(cfg["wam_adapter"]["default_target_hours"]["TS"])
    base_map = {"TR": base_TR, "TS": base_TS}
    staff["target"] = staff["contract_type"].map(base_map).fillna(base_TR) * staff["fte"]
    prealloc_path = inst/"preallocations.csv"
    if prealloc_path.exists():
        pre = pd.read_csv(prealloc_path).groupby("staff_id")["hours"].sum().to_dict()
        staff["target"] = staff.apply(lambda r: r["target"] - pre.get(r["staff_id"], 0.0), axis=1)

    period = modules.set_index("module_id")["period"].to_dict()
    indivisible = set(modules.loc[modules["indivisible"]==1, "module_id"].tolist())
    elig1 = elig[elig["eligible"]==1][["staff_id","duty_id"]].drop_duplicates()

    # current loads
    load = {sid: 0.0 for sid in staff["staff_id"]}

    def availability_ok(sid, mod):
        r = avail[(avail["staff_id"]==sid) & (avail["period"]==period[mod])]
        if r.empty: return True
        return float(r["availability"].iloc[0]) > 0.0

    # Generate assignment rows
    rows = []

    # Assign indivisible modules first
    for m in modules["module_id"]:
        duties_m = duties[duties["module_id"]==m]
        total_m = float(duties_m["hours_required"].sum())
        if m in indivisible:
            # find a single staff to take all hours
            candidates = []
            for sid in staff["staff_id"]:
                ok = True
                if not availability_ok(sid, m): ok=False
                # must be eligible for ALL duties
                for _,d in duties_m.iterrows():
                    if (sid, d["duty_id"]) not in set(map(tuple, elig1.to_records(index=False))):
                        ok=False; break
                if ok:
                    rem = max_hours.get(sid, float("inf")) - load[sid]
                    if rem >= total_m:
                        gap = (load[sid] - float(staff.loc[staff["staff_id"]==sid,"target"].iloc[0]))
                        candidates.append((gap, sid))
            if candidates:
                sid = sorted(candidates, key=lambda x: (x[0], x[1]))[0][1]
                for _,d in duties_m.iterrows():
                    rows.append({"module_id":m,"duty_id":d["duty_id"],"staff_id":sid,"hours_assigned":float(d.get("hours_required_eff", d["hours_required"]))})
                load[sid] += total_m
                continue  # next module
        # if not indivisible or no single candidate, assign per duty
        for _,d in duties_m.iterrows():
            did = d["duty_id"]; H = float(d.get("hours_required_eff", d["hours_required"]))
            cand = []
            for sid in staff["staff_id"]:
                if (sid, did) in set(map(tuple, elig1.to_records(index=False))) and availability_ok(sid, m):
                    rem = max_hours.get(sid, float("inf")) - load[sid]
                    if rem > 0:
                        gap = (load[sid] - float(staff.loc[staff["staff_id"]==sid,"target"].iloc[0]))
                        cand.append((gap, rem, sid))
            if not cand:
                # last resort: assign to least loaded staff ignoring availability
                sid = sorted([(load[s], s) for s in staff["staff_id"]])[0][1]
                rows.append({"module_id":m,"duty_id":did,"staff_id":sid,"hours_assigned":H})
                load[sid] += H
            else:
                # prefer under-target, enough remaining capacity
                cand.sort(key=lambda x: (x[0], -x[1]))
                sid = cand[0][2]
                hrs = min(H, max_hours.get(sid, float("inf")) - load[sid])
                if hrs < H and len(cand) > 1:
                    # split remainder to next candidate
                    rows.append({"module_id":m,"duty_id":did,"staff_id":sid,"hours_assigned":hrs})
                    load[sid] += hrs
                    remH = H - hrs
                    sid2 = cand[1][2]
                    rows.append({"module_id":m,"duty_id":did,"staff_id":sid2,"hours_assigned":remH})
                    load[sid2] += remH
                else:
                    rows.append({"module_id":m,"duty_id":did,"staff_id":sid,"hours_assigned":H})
                    load[sid] += H

    out = pd.DataFrame(rows)
    out.to_csv(out_assignment, index=False)
    print(f"Wrote {out_assignment}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_dir", required=True)
    ap.add_argument("--resolved_config", required=True)
    ap.add_argument("--out_assignment", required=True)
    args = ap.parse_args()
    main(args.instance_dir, args.resolved_config, args.out_assignment)
