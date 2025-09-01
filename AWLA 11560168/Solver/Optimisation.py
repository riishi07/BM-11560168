"""
Solver/Optimisation.py
======================

End-to-end optimiser for AWLA:
1) ILP seed (Tier-1 fairness + essentials) using PuLP backends (CBC/Gurobi/HiGHS).
2) Hyper-heuristic / Large Neighbourhood Search (LNS) with LAHC acceptor
   and neighbourhood operators: reassign, swap, block_move, split_merge.
3) Outputs: assignment.csv, objective_breakdown.json, optimise_meta.json

This module consumes a validated `Problem` (from Solver.Problem).
"""

from __future__ import annotations

import os
import shutil
import inspect
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

# Problem loader
from Solver.Problem import Problem

# Optional: PuLP backends
try:
    import pulp

    HAS_PULP = True
except Exception:
    HAS_PULP = False


# =============================================================================
# Backend factories (robust to PuLP version differences)
# =============================================================================


def _make_cbc_solver(
    time_limit: int | float | None, mip_gap: float | None, threads: int | None = None
):
    """
    CBC factory that:
      - handles keyword drift (timeLimit/maximumSeconds, fracGap/gapRel/ratioGap/mipGap)
      - pins the CBC binary (CBC_PATH or PATH)
      - works around PuLP's flaky available() on macOS/conda by trusting an executable path
    """
    if not HAS_PULP:
        raise RuntimeError("PuLP is required for CBC. `pip install pulp`")

    sig = inspect.signature(pulp.PULP_CBC_CMD)
    kwargs: Dict[str, object] = {"msg": 0}

    # time limit (name varies by PuLP version)
    if time_limit is not None:
        if "timeLimit" in sig.parameters:
            kwargs["timeLimit"] = int(time_limit)
        elif "maximumSeconds" in sig.parameters:
            kwargs["maximumSeconds"] = int(time_limit)

    # threads if supported
    if threads and "threads" in sig.parameters:
        kwargs["threads"] = int(threads)

    # relative mip gap (name varies)
    if mip_gap is not None:
        gap = float(mip_gap)
        for gk in ("fracGap", "gapRel", "ratioGap", "mipGap"):
            if gk in sig.parameters:
                kwargs[gk] = gap
                break
        else:
            # pass raw CBC option as fallback
            kwargs["options"] = [f"ratioGap={gap}"]

    # Prefer explicit CBC binary path
    cbc_path = os.environ.get("CBC_PATH") or shutil.which("cbc")

    # Subclass to override available() and trust an executable binary path
    class _CBC_CMD_FORCE(pulp.PULP_CBC_CMD):
        def available(self) -> bool:  # type: ignore[override]
            try:
                ok = super().available()
                if ok:
                    return True
            except Exception:
                pass
            p = (
                getattr(self, "path", None)
                or os.environ.get("CBC_PATH")
                or shutil.which("cbc")
            )
            return bool(p and os.path.exists(p) and os.access(p, os.X_OK))

    solver = _CBC_CMD_FORCE(**kwargs)
    # Assign path attribute even if not in __init__ signature
    if cbc_path:
        try:
            setattr(solver, "path", cbc_path)
        except Exception:
            pass
    return solver


def _make_highs_solver(time_limit: int | float | None):
    if not HAS_PULP:
        raise RuntimeError("PuLP is required for HiGHS. `pip install pulp`")
    sig = inspect.signature(pulp.HiGHS_CMD)
    kwargs: Dict[str, object] = {"msg": 0}
    if time_limit is not None and "timeLimit" in sig.parameters:
        kwargs["timeLimit"] = int(time_limit)
    return pulp.HiGHS_CMD(**kwargs)


def _make_gurobi_solver(time_limit: int | float | None, mip_gap: float | None):
    if not HAS_PULP:
        raise RuntimeError("PuLP is required for Gurobi. `pip install pulp`")
    try:
        try:
            return pulp.GUROBI_CMD(
                msg=0,
                timeLimit=int(time_limit or 0),
                options=[("MIPGap", float(mip_gap or 0.0))],
            )
        except TypeError:
            return pulp.GUROBI_CMD(
                msg=0,
                timeLimit=int(time_limit or 0),
                options=[f"MIPGap={float(mip_gap or 0.0)}"],
            )
    except Exception as e:
        raise RuntimeError(f"Gurobi not available: {e}")


# =============================================================================
# Utility data structures
# =============================================================================


@dataclass
class OptimiseResult:
    assignment: pd.DataFrame
    objective_breakdown: Dict[str, float]
    meta: Dict[str, Any]


# =============================================================================
# Evaluator (objective + hard checks)
# =============================================================================


class Evaluator:
    def __init__(self, problem: Problem):
        self.P = problem
        self.cfg = self.P.config
        self.W = self.cfg.get("weights", {})
        self.soft_conflict_types = {"line_manager_pairing_avoid", "personal_conflict"}

        self.modules = self.P.get_dataframe("modules")
        self.duties = self.P.get_dataframe("duties")
        self.staff = self.P.get_dataframe("staff")
        self.elig = self.P.get_dataframe("eligibility")
        self.avail = self.P.get_dataframe("availability")
        self.prefs = self.P.get_dataframe("preferences")
        self.conf = self.P.get_dataframe("conflicts_staff_pairs")
        self.baseline = self.P.get_dataframe("baseline_plan")

        self.period_of = dict(zip(self.modules["module_id"], self.modules["period"]))
        self.hours_required = (
            self.duties.set_index("duty_id")["hours_required_eff"]
            if "hours_required_eff" in self.duties.columns
            else self.duties.set_index("duty_id")["hours_required"]
        ).to_dict()
        self.module_of_duty = self.duties.set_index("duty_id")["module_id"].to_dict()
        self.indivisible_modules = set(
            self.modules.loc[self.modules["indivisible"] == 1, "module_id"].tolist()
        )
        self.max_hours = (
            self.staff.set_index("staff_id")["max_hours"].fillna(float("inf")).to_dict()
        )
        self.targets = self.P.targets

        # Eligibility maps
        self.elig_map = {
            (r["staff_id"], r["duty_id"]): int(r["eligible"])
            for _, r in self.elig.iterrows()
        }
        self.elig_module_rows: Dict[tuple, int] = {}
        em = self.elig[self.elig["duty_id"].fillna("").astype(str) == ""]
        if not em.empty:
            for _, r in em.iterrows():
                self.elig_module_rows[(r["staff_id"], r["module_id"])] = int(
                    r["eligible"]
                )

    def is_eligible(self, sid: str, did: str) -> bool:
        key = (sid, did)
        if key in self.elig_map:
            return self.elig_map[key] == 1
        mod = self.module_of_duty.get(did, "")
        if mod and (sid, mod) in self.elig_module_rows:
            return self.elig_module_rows[(sid, mod)] == 1
        return True

    def availability(self, sid: str, module_id: str) -> float:
        per = self.period_of.get(module_id, "")
        r = self.avail[(self.avail["staff_id"] == sid) & (self.avail["period"] == per)]
        if r.empty:
            return 1.0
        return float(r["availability"].iloc[0])

    def fairness_raw(self, A: pd.DataFrame) -> float:
        loads = A.groupby("staff_id")["hours_assigned"].sum()
        total = 0.0
        for sid in self.staff["staff_id"]:
            L = float(loads.get(sid, 0.0))
            T = float(self.targets.get(sid, 0.0))
            total += abs(L - T)
        return float(total)

    # ---- Hard checks ----
    def hard_violations(self, A: pd.DataFrame) -> List[str]:
        msgs: List[str] = []
        # H1 coverage
        s = A.groupby("duty_id")["hours_assigned"].sum()
        for did, H in self.hours_required.items():
            if abs(float(s.get(did, 0.0)) - float(H)) > 1e-6:
                msgs.append(
                    f"H1 coverage: duty {did} sum {s.get(did, 0.0)} != required {H}"
                )
        # H2 eligibility
        for _, r in A.iterrows():
            if r["hours_assigned"] <= 0:
                continue
            if not self.is_eligible(r["staff_id"], r["duty_id"]):
                msgs.append(
                    f"H2 eligibility: {r['staff_id']} not eligible for {r['duty_id']}"
                )
        # H3 capacity
        loads = A.groupby("staff_id")["hours_assigned"].sum()
        for sid, L in loads.items():
            cap = float(self.max_hours.get(sid, float("inf")))
            if L - cap > 1e-6:
                msgs.append(f"H3 capacity: {sid} load {L} > cap {cap}")
        # H4 availability=0
        for _, r in A.iterrows():
            if r["hours_assigned"] <= 0:
                continue
            if self.availability(r["staff_id"], r["module_id"]) == 0.0:
                msgs.append(
                    f"H4 availability: {r['staff_id']} unavailable for module {r['module_id']}"
                )
        # H5 indivisible → one staff
        if self.indivisible_modules:
            mod_staff = (
                A.groupby(["module_id", "staff_id"])["hours_assigned"]
                .sum()
                .reset_index()
            )
            for m in self.indivisible_modules:
                sub = mod_staff[mod_staff["module_id"] == m]
                count = (sub["hours_assigned"] > 0).sum()
                if count > 1:
                    msgs.append(f"H5 indivisible: module {m} assigned to {count} staff")
        # H7 cannot_co_teach (if present in your data)
        return msgs

    # ---- Objective ----
    def objective(self, A: pd.DataFrame) -> Dict[str, float]:
        W = self.W
        loads = A.groupby("staff_id")["hours_assigned"].sum()
        fairness = 0.0
        for sid in self.staff["staff_id"]:
            L = float(loads.get(sid, 0.0))
            T = float(self.targets.get(sid, 0.0))
            fairness += abs(L - T) * float(W.get("fairness_L1", 0.0))

        # Preferences
        pref_total = 0.0
        if not self.prefs.empty:
            P_ = A.copy()
            P_["period"] = P_["module_id"].map(self.period_of)
            # staff_module
            sm = self.prefs[self.prefs["pref_type"] == "staff_module"]
            if not sm.empty:
                key = P_.groupby(["staff_id", "module_id"])["hours_assigned"].sum()
                for _, r in sm.iterrows():
                    hrs = float(key.get((r["staff_id"], r["module_id"]), 0.0))
                    if hrs != 0:
                        pref_total += (
                            float(W.get("pref", 0.0)) * float(r["penalty"]) * hrs
                        )
            # staff_duty
            sd = self.prefs[self.prefs["pref_type"] == "staff_duty"]
            if not sd.empty:
                key = P_.groupby(["staff_id", "duty_id"])["hours_assigned"].sum()
                for _, r in sd.iterrows():
                    hrs = float(key.get((r["staff_id"], r["duty_id"]), 0.0))
                    if hrs != 0:
                        pref_total += (
                            float(W.get("pref", 0.0)) * float(r["penalty"]) * hrs
                        )
            # staff_period
            sp = self.prefs[self.prefs["pref_type"] == "staff_period"]
            if not sp.empty:
                key = P_.groupby(["staff_id", "period"])["hours_assigned"].sum()
                for _, r in sp.iterrows():
                    hrs = float(key.get((r["staff_id"], r["period"]), 0.0))
                    if hrs != 0:
                        pref_total += (
                            float(W.get("pref", 0.0)) * float(r["penalty"]) * hrs
                        )

        # Team balance
        tb = 0.0
        for _, sids in self.staff.groupby("team"):
            Ls = [float(loads.get(s, 0.0)) for s in sids["staff_id"]]
            if len(Ls) == 0:
                continue
            meanL = float(np.mean(Ls))
            disp = sum(abs(L - meanL) for L in Ls)
            tb += float(self.W.get("team_balance", 0.0)) * disp

        # Splitting (by (module,duty))
        split_total = 0.0
        if not A.empty:
            distinct_staff = A.groupby(["module_id", "duty_id"])["staff_id"].nunique()
            for _, k in distinct_staff.items():
                if int(k) > 1:
                    split_total += float(self.W.get("split", 0.0)) * (int(k) - 1)

        # Stability (eligible-only if configured)
        stability = 0.0
        if not self.baseline.empty:
            include_only_elig = self.cfg.get("checker", {}).get(
                "baseline_stability_include_only_eligible", True
            )
            merged = self.baseline.merge(
                A,
                on=["module_id", "duty_id", "staff_id"],
                how="outer",
                suffixes=("_base", "_ass"),
            ).fillna(0.0)
            for _, r in merged.iterrows():
                sid, did = r["staff_id"], r["duty_id"]
                if include_only_elig:
                    if not isinstance(did, str) or did == "":
                        continue
                    if not self.is_eligible(sid, did):
                        continue
                delta = abs(
                    float(r["hours_assigned_ass"]) - float(r["hours_assigned_base"])
                )
                if delta > 0:
                    stability += float(self.W.get("stability", 0.0)) * delta

        # Soft conflicts (lightweight placeholder; your detailed version can be slotted in)
        soft_total = 0.0
        if not self.conf.empty:
            mod_staff = (
                A.groupby(["module_id", "staff_id"])["hours_assigned"]
                .sum()
                .reset_index()
            )
            for m, df in mod_staff.groupby("module_id"):
                _ = set(df.loc[df["hours_assigned"] > 0, "staff_id"].tolist())
                # sum conflict penalties here if you maintain them per (s1,s2,m)
        soft_total *= float(self.W.get("soft_conflicts", 1.0))

        breakdown = {
            "fairness_L1": round(fairness, 6),
            "preferences": round(pref_total, 6),
            "team_balance": round(tb, 6),
            "split": round(split_total, 6),
            "stability": round(stability, 6),
            "soft_conflicts": round(soft_total, 6),
        }
        breakdown["weighted_sum"] = round(sum(breakdown.values()), 6)
        return breakdown


# =============================================================================
# ILP Seed (Tier-1 fairness + essentials)
# =============================================================================


class ILPSeed:
    def __init__(self, problem: Problem):
        if not HAS_PULP:
            raise RuntimeError("PuLP is required for ILP seed. `pip install pulp`.")
        self.P = problem
        self.cfg = self.P.config
        self.W = self.cfg.get("weights", {})
        self.model = None
        self.vars_h: Dict[tuple, Any] = {}
        self.vars_y: Dict[tuple, Any] = {}
        self.d_plus: Dict[str, Any] = {}
        self.d_minus: Dict[str, Any] = {}

        self.modules = self.P.get_dataframe("modules")
        self.duties = self.P.get_dataframe("duties")
        self.staff = self.P.get_dataframe("staff")
        self.elig = self.P.get_dataframe("eligibility")
        self.avail = self.P.get_dataframe("availability")
        self.conf = self.P.get_dataframe("conflicts_staff_pairs")

        self.period_of = dict(zip(self.modules["module_id"], self.modules["period"]))
        self.hours_required = (
            self.duties.set_index("duty_id")["hours_required_eff"]
            if "hours_required_eff" in self.duties.columns
            else self.duties.set_index("duty_id")["hours_required"]
        ).to_dict()
        self.module_of_duty = self.duties.set_index("duty_id")["module_id"].to_dict()
        self.indivisible_modules = set(
            self.modules.loc[self.modules["indivisible"] == 1, "module_id"].tolist()
        )
        self.max_hours = (
            self.staff.set_index("staff_id")["max_hours"].fillna(float("inf")).to_dict()
        )
        self.targets = self.P.targets

        self.elig_map = {
            (r["staff_id"], r["duty_id"]): int(r["eligible"])
            for _, r in self.elig.iterrows()
        }
        self.elig_module_rows: Dict[tuple, int] = {}
        em = self.elig[self.elig["duty_id"].fillna("").astype(str) == ""]
        if not em.empty:
            for _, r in em.iterrows():
                self.elig_module_rows[(r["staff_id"], r["module_id"])] = int(
                    r["eligible"]
                )

    def is_eligible(self, sid: str, did: str) -> bool:
        if (sid, did) in self.elig_map:
            return self.elig_map[(sid, did)] == 1
        mod = self.module_of_duty.get(did, "")
        if mod and (sid, mod) in self.elig_module_rows:
            return self.elig_module_rows[(sid, mod)] == 1
        return True

    def availability(self, sid: str, module_id: str) -> float:
        per = self.period_of.get(module_id, "")
        r = self.avail[(self.avail["staff_id"] == sid) & (self.avail["period"] == per)]
        if r.empty:
            return 1.0
        return float(r["availability"].iloc[0])

    def build(self):
        self.model = pulp.LpProblem("AWLA_ILP_Seed", pulp.LpMinimize)

        # h_{s,d} variables for eligible/available pairs
        for _, d in self.duties.iterrows():
            did = d["duty_id"]
            mod = d["module_id"]
            H = float(d.get("hours_required_eff", d["hours_required"]))
            for sid in self.staff["staff_id"]:
                if not self.is_eligible(sid, did):
                    continue
                if self.availability(sid, mod) == 0.0:
                    continue
                v = pulp.LpVariable(
                    f"h_{sid}__{did}", lowBound=0, upBound=H, cat="Continuous"
                )
                self.vars_h[(sid, did)] = v

        # y_{s,m} binary (teaches module) for indivisible & conflicts linkage
        for m in self.modules["module_id"].tolist():
            for sid in self.staff["staff_id"]:
                self.vars_y[(sid, m)] = pulp.LpVariable(
                    f"y_{sid}__{m}", lowBound=0, upBound=1, cat="Binary"
                )

        # fairness slacks
        for sid in self.staff["staff_id"]:
            self.d_plus[sid] = pulp.LpVariable(
                f"dplus_{sid}", lowBound=0, cat="Continuous"
            )
            self.d_minus[sid] = pulp.LpVariable(
                f"dminus_{sid}", lowBound=0, cat="Continuous"
            )

        # coverage: sum_s h_{s,d} == H_d
        for _, d in self.duties.iterrows():
            did = d["duty_id"]
            H = float(d.get("hours_required_eff", d["hours_required"]))
            terms = [
                self.vars_h[(sid, did)]
                for sid in self.staff["staff_id"]
                if (sid, did) in self.vars_h
            ]
            self.model += (pulp.lpSum(terms) == H), f"coverage_{did}"

        # link h to y (if not teaching module, cannot get hours in that module)
        for _, d in self.duties.iterrows():
            did = d["duty_id"]
            H = float(d.get("hours_required_eff", d["hours_required"]))
            m = d["module_id"]
            for sid in self.staff["staff_id"]:
                v = self.vars_h.get((sid, did))
                if v is not None:
                    self.model += v <= H * self.vars_y[(sid, m)], f"link_hy_{sid}_{did}"

        # indivisible modules: exactly one teacher
        for m in self.indivisible_modules:
            self.model += (
                pulp.lpSum(self.vars_y[(sid, m)] for sid in self.staff["staff_id"])
                == 1,
                f"indivisible_{m}",
            )

        # capacity: sum_d h_{s,d} <= max_hours
        for sid in self.staff["staff_id"]:
            cap = float(self.max_hours.get(sid, float("inf")))
            if math.isfinite(cap):
                terms = [
                    self.vars_h[(sid, did)]
                    for did in self.duties["duty_id"]
                    if (sid, did) in self.vars_h
                ]
                self.model += pulp.lpSum(terms) <= cap, f"capacity_{sid}"

        # fairness balance: sum_d h_{s,d} - T_s = dplus - dminus
        for sid in self.staff["staff_id"]:
            T = float(self.targets.get(sid, 0.0))
            terms = [
                self.vars_h[(sid, did)]
                for did in self.duties["duty_id"]
                if (sid, did) in self.vars_h
            ]
            self.model += (
                pulp.lpSum(terms) - T == self.d_plus[sid] - self.d_minus[sid],
                f"fair_bal_{sid}",
            )

        # objective: minimise fairness L1
        wF = float(self.W.get("fairness_L1", 1.0))
        self.model += (
            wF
            * pulp.lpSum(
                [self.d_plus[sid] + self.d_minus[sid] for sid in self.staff["staff_id"]]
            ),
            "obj_fairness",
        )

    def solve(
        self,
        time_limit: int = 300,
        mip_gap: float = 0.0,
        threads: Optional[int] = None,
        backend: str = "auto",
    ) -> pd.DataFrame:
        if self.model is None:
            self.build()

        backend = (backend or "auto").lower()

        def _choose_solver(bk: str):
            if bk == "cbc":
                return _make_cbc_solver(
                    time_limit=time_limit, mip_gap=mip_gap, threads=threads
                )
            if bk == "gurobi":
                return _make_gurobi_solver(time_limit, mip_gap)
            if bk == "highs":
                return _make_highs_solver(time_limit)
            if bk == "auto":
                # prefer CBC; fallback to HiGHS
                try:
                    return _make_cbc_solver(
                        time_limit=time_limit, mip_gap=mip_gap, threads=threads
                    )
                except Exception:
                    return _make_highs_solver(time_limit)
            # default: try CBC
            return _make_cbc_solver(
                time_limit=time_limit, mip_gap=mip_gap, threads=threads
            )

        solver = _choose_solver(backend)

        # Try solve; on PuLP "Not Available" for CBC, auto-fallback to HiGHS
        try:
            status = self.model.solve(solver)
        except Exception as e:
            msg = str(e)
            # Auto fallback only if we attempted CBC / AUTO
            if ("Not Available" in msg or "permissions on cbc" in msg) and backend in (
                "cbc",
                "auto",
            ):
                # fallback to HiGHS
                solver = _make_highs_solver(time_limit)
                status = self.model.solve(solver)
            else:
                raise

        # basic status guard (PuLP status code varies; keep permissive)
        if (
            pulp.LpStatus[status]
            not in (
                "Optimal",
                "Integer Feasible",
                "Not Solved",
                "Undefined",
                "Infeasible",
            )
            and status != 1
        ):
            raise RuntimeError(f"ILP solve failed with status: {pulp.LpStatus[status]}")

        rows: List[Dict[str, object]] = []
        for (sid, did), var in self.vars_h.items():
            val = float(pulp.value(var) or 0.0)
            if val > 1e-8:
                m = self.module_of_duty.get(did, "")
                rows.append(
                    {
                        "module_id": m,
                        "duty_id": did,
                        "staff_id": sid,
                        "hours_assigned": val,
                    }
                )
        return pd.DataFrame(rows)


# =============================================================================
# LNS with LAHC
# =============================================================================


class LNSHyper:
    def __init__(
        self,
        problem: Problem,
        evaluator: Evaluator,
        seed_assignment: pd.DataFrame,
        rng: random.Random,
        fairness_cap: float | None = None,
    ):
        self.P = problem
        self.E = evaluator
        self.A = seed_assignment.copy()
        self.rng = rng
        self.bestA = self.A.copy()
        self.bestCost = self.E.objective(self.A)["weighted_sum"]
        self.fairness_cap = fairness_cap

        self.hours_required = self.E.hours_required
        self.module_of_duty = self.E.module_of_duty
        self.indivisible_modules = self.E.indivisible_modules
        self.max_hours = self.E.max_hours
        self.targets = self.E.targets

        # optional: will be set by orchestrator
        self.run_dir: Optional[Path] = None

    def loads(self, A: pd.DataFrame) -> Dict[str, float]:
        return A.groupby("staff_id")["hours_assigned"].sum().to_dict()

    def feasible(self, A: pd.DataFrame) -> bool:
        return len(self.E.hard_violations(A)) == 0

    def reassign(self, A: pd.DataFrame) -> pd.DataFrame:
        if A.empty:
            return A
        row = A.sample(1, random_state=self.rng.randint(0, 10**9)).iloc[0]
        did, mid, sid_from, hrs = (
            row["duty_id"],
            row["module_id"],
            row["staff_id"],
            float(row["hours_assigned"]),
        )
        elig_staff = [
            s
            for s in self.P.get_dataframe("staff")["staff_id"]
            if self.E.is_eligible(s, did) and self.E.availability(s, mid) > 0.0
        ]
        if not elig_staff:
            return A
        sid_to = self.rng.choice(elig_staff)
        if sid_to == sid_from:
            return A
        delta = hrs * self.rng.uniform(0.3, 0.7)
        loads = self.loads(A)
        cap_to = float(self.max_hours.get(sid_to, float("inf")))
        free_to = cap_to - float(loads.get(sid_to, 0.0))
        if free_to <= 1e-6:
            return A
        delta = min(delta, free_to)

        B = A.copy()
        idx = B.index[
            (B["duty_id"] == did)
            & (B["staff_id"] == sid_from)
            & (B["module_id"] == mid)
        ][0]
        B.at[idx, "hours_assigned"] = float(B.at[idx, "hours_assigned"]) - delta
        mask = (
            (B["duty_id"] == did) & (B["staff_id"] == sid_to) & (B["module_id"] == mid)
        )
        if mask.any():
            j = B.index[mask][0]
            B.at[j, "hours_assigned"] = float(B.at[j, "hours_assigned"]) + delta
        else:
            B = pd.concat(
                [
                    B,
                    pd.DataFrame(
                        [
                            {
                                "module_id": mid,
                                "duty_id": did,
                                "staff_id": sid_to,
                                "hours_assigned": delta,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        B = B[B["hours_assigned"] > 1e-8].reset_index(drop=True)
        return B

    def swap(self, A: pd.DataFrame) -> pd.DataFrame:
        if len(A) < 2:
            return A
        r1, r2 = A.sample(2, random_state=self.rng.randint(0, 10**9)).to_dict(
            orient="records"
        )
        if r1["staff_id"] == r2["staff_id"]:
            return A
        per1 = self.E.period_of.get(r1["module_id"], "")
        per2 = self.E.period_of.get(r2["module_id"], "")
        if per1 != per2:
            return A
        if not self.E.is_eligible(
            r1["staff_id"], r2["duty_id"]
        ) or not self.E.is_eligible(r2["staff_id"], r1["duty_id"]):
            return A
        if (
            self.E.availability(r1["staff_id"], r2["module_id"]) == 0.0
            or self.E.availability(r2["staff_id"], r1["module_id"]) == 0.0
        ):
            return A
        delta = min(
            float(r1["hours_assigned"]), float(r2["hours_assigned"])
        ) * self.rng.uniform(0.3, 0.7)
        if delta <= 1e-6:
            return A

        B = A.copy()

        def add_hours(df, duty_id, module_id, staff_id, change):
            mask = (
                (df["duty_id"] == duty_id)
                & (df["module_id"] == module_id)
                & (df["staff_id"] == staff_id)
            )
            if mask.any():
                idx = df.index[mask][0]
                df.at[idx, "hours_assigned"] = (
                    float(df.at[idx, "hours_assigned"]) + change
                )
            else:
                df.loc[len(df)] = {
                    "module_id": module_id,
                    "duty_id": duty_id,
                    "staff_id": staff_id,
                    "hours_assigned": change,
                }

        add_hours(B, r1["duty_id"], r1["module_id"], r1["staff_id"], -delta)
        add_hours(B, r1["duty_id"], r1["module_id"], r2["staff_id"], +delta)
        add_hours(B, r2["duty_id"], r2["module_id"], r2["staff_id"], -delta)
        add_hours(B, r2["duty_id"], r2["module_id"], r1["staff_id"], +delta)
        B = B[B["hours_assigned"] > 1e-8].reset_index(drop=True)
        return B

    def block_move(self, A: pd.DataFrame) -> pd.DataFrame:
        if A.empty:
            return A
        m = self.rng.choice(A["module_id"].unique().tolist())
        duties_m = A[A["module_id"] == m]["duty_id"].unique().tolist()
        candidates: List[str] = []
        for sid in self.P.get_dataframe("staff")["staff_id"]:
            if self.E.availability(sid, m) == 0.0:
                continue
            if all(self.E.is_eligible(sid, d) for d in duties_m):
                candidates.append(sid)
        if not candidates:
            return A
        sid_to = self.rng.choice(candidates)
        loads = self.loads(A)
        cap_to = float(self.max_hours.get(sid_to, float("inf")))
        free_to = cap_to - float(loads.get(sid_to, 0.0))
        total_module = float(A.loc[A["module_id"] == m, "hours_assigned"].sum())
        if free_to + 1e-6 < total_module:
            return A

        B = A.copy()
        B = B[B["module_id"] != m]
        required = self.P.get_dataframe("duties")
        required = required[required["module_id"] == m]
        for _, r in required.iterrows():
            B.loc[len(B)] = {
                "module_id": m,
                "duty_id": r["duty_id"],
                "staff_id": sid_to,
                "hours_assigned": float(
                    r.get("hours_required_eff", r["hours_required"])
                ),
            }
        return B.reset_index(drop=True)

    def split_merge(self, A: pd.DataFrame) -> pd.DataFrame:
        if A.empty:
            return A
        m = self.rng.choice(A["module_id"].unique().tolist())
        sub = A[A["module_id"] == m]
        staff_set = sub["staff_id"].unique().tolist()
        if len(staff_set) > 1:
            return self.block_move(A)
        row = sub.sample(1, random_state=self.rng.randint(0, 10**9)).iloc[0]
        did, sid_from, hrs = (
            row["duty_id"],
            row["staff_id"],
            float(row["hours_assigned"]),
        )
        elig_staff = [
            s
            for s in self.P.get_dataframe("staff")["staff_id"]
            if self.E.is_eligible(s, did)
            and self.E.availability(s, m) > 0.0
            and s != sid_from
        ]
        if not elig_staff:
            return A
        sid_to = self.rng.choice(elig_staff)
        delta = hrs * self.rng.uniform(0.3, 0.6)
        loads = self.loads(A)
        cap_to = float(self.max_hours.get(sid_to, float("inf")))
        free_to = cap_to - float(loads.get(sid_to, 0.0))
        if free_to <= 1e-6:
            return A
        delta = min(delta, free_to)

        B = A.copy()
        idx = B.index[
            (B["module_id"] == m) & (B["duty_id"] == did) & (B["staff_id"] == sid_from)
        ][0]
        B.at[idx, "hours_assigned"] = float(B.at[idx, "hours_assigned"]) - delta
        mask = (B["module_id"] == m) & (B["duty_id"] == did) & (B["staff_id"] == sid_to)
        if mask.any():
            j = B.index[mask][0]
            B.at[j, "hours_assigned"] = float(B.at[j, "hours_assigned"]) + delta
        else:
            B.loc[len(B)] = {
                "module_id": m,
                "duty_id": did,
                "staff_id": sid_to,
                "hours_assigned": delta,
            }
        return B[B["hours_assigned"] > 1e-8].reset_index(drop=True)

    def step(self, A: pd.DataFrame) -> pd.DataFrame:
        op = self.rng.choice(["reassign", "swap", "block_move", "split_merge"])
        if op == "reassign":
            return self.reassign(A)
        if op == "swap":
            return self.swap(A)
        if op == "block_move":
            return self.block_move(A)
        return self.split_merge(A)

    # --------- LNS run with convergence logging ----------
    def run(
        self,
        seconds: int = 600,
        lahc_length: int = 2000,
        log_every: int = 50,
    ) -> pd.DataFrame:
        """
        Late-Acceptance Hill-Climbing over LNS neighbourhoods with
        ε-fairness guard and periodic convergence logging to CSV.
        """
        start = time.time()
        current = self.A.copy()
        cur_cost = self.E.objective(current)["weighted_sum"]
        history = [cur_cost] * max(1, int(lahc_length))
        best = self.bestA.copy()
        best_cost = float(self.bestCost)

        # For logging
        logs: List[Dict[str, float | int]] = []
        accepted = 0
        it = 0
        end_time = start + max(1, int(seconds))

        while time.time() < end_time:
            it += 1
            cand = self.step(current)
            if not self.feasible(cand):
                continue

            # ε-guard fairness: keep within ILP optimum + ε hours
            if self.fairness_cap is not None:
                if self.E.fairness_raw(cand) > float(self.fairness_cap):
                    continue

            cand_cost = self.E.objective(cand)["weighted_sum"]
            idx = it % len(history)
            accepted_move = False
            if cand_cost <= history[idx] or cand_cost <= cur_cost:
                current = cand
                cur_cost = cand_cost
                history[idx] = cand_cost
                accepted += 1
                accepted_move = True
                if cand_cost < best_cost - 1e-9:
                    best = cand
                    best_cost = cand_cost

            # periodic log (iteration-level)
            if (it % max(1, int(log_every))) == 0:
                logs.append(
                    {
                        "iter": it,
                        "elapsed_sec": float(time.time() - start),
                        "current_cost": float(cur_cost),
                        "best_cost": float(best_cost),
                        "accepted_move": int(accepted_move),
                        "accept_rate": float(accepted / it),
                    }
                )

        # final log row
        logs.append(
            {
                "iter": it,
                "elapsed_sec": float(time.time() - start),
                "current_cost": float(cur_cost),
                "best_cost": float(best_cost),
                "accepted_move": 0,
                "accept_rate": float(accepted / max(1, it)),
            }
        )

        # write convergence CSV if run_dir is provided
        if getattr(self, "run_dir", None):
            try:
                df = pd.DataFrame(logs)
                df.to_csv(Path(self.run_dir) / "lns_history.csv", index=False)
            except Exception:
                pass

        self.bestA = best
        self.bestCost = best_cost
        return best

    # --------- /PATCHED -------------------------------------------


# =============================================================================
# Orchestrator
# =============================================================================


def solve_end_to_end(
    instance_dir: Path | str, resolved_config_path: Path | str, run_dir: Path | str
) -> OptimiseResult:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(resolved_config_path).read_text())
    P = Problem(Path(instance_dir), cfg).load()

    evalr = Evaluator(P)

    meta: Dict[str, Any] = {"seed": int(cfg.get("solver", {}).get("seed", 42))}
    rng = random.Random(meta["seed"])

    # ILP Seed
    if not HAS_PULP:
        raise RuntimeError("PuLP not installed. `pip install pulp`")

    ilp = ILPSeed(P)
    ilp_time = int(cfg.get("solver", {}).get("ilp_time_sec", 300))
    mip_gap = float(cfg.get("solver", {}).get("ilp_mip_gap", 0.0))
    backend = (cfg.get("solver", {}).get("ilp", {}) or {}).get(
        "backend", cfg.get("solver", {}).get("backend", "auto")
    )
    threads = (cfg.get("solver", {}).get("ilp", {}) or {}).get("threads", None)

    t0 = time.time()
    A0 = ilp.solve(
        time_limit=ilp_time, mip_gap=mip_gap, threads=threads, backend=backend
    )
    meta["ilp_seconds"] = round(time.time() - t0, 3)
    meta["ilp_rows"] = int(A0.shape[0])

    # Validate hard constraints for seed
    hard = evalr.hard_violations(A0)
    if hard:
        raise RuntimeError(
            f"ILP seed violated hard constraints: {hard[:5]}{'...' if len(hard) > 5 else ''}"
        )

    # Lexicographic fairness guard for LNS
    F_star = evalr.fairness_raw(A0)
    total_T = float(sum(evalr.targets.values()))
    eps = float(cfg.get("solver", {}).get("lexi", {}).get("epsilon_fair", 0.01))
    eps_hours = eps * total_T if eps < 1.0 else eps
    fairness_cap = F_star + eps_hours

    # LNS
    hh_time = int(cfg.get("solver", {}).get("hh_time_sec", 600))
    lahc_len = int(cfg.get("solver", {}).get("lahc_length", 2000))
    t1 = time.time()
    lns = LNSHyper(P, evalr, A0, rng, fairness_cap=fairness_cap)
    # NEW: provide run_dir so LNS can write lns_history.csv
    lns.run_dir = run_dir
    A_best = lns.run(seconds=hh_time, lahc_length=lahc_len)
    meta["hh_seconds"] = round(time.time() - t1, 3)

    breakdown = evalr.objective(A_best)

    # Outputs
    outA = run_dir / "assignment.csv"
    A_best.to_csv(outA, index=False)
    (run_dir / "objective_breakdown.json").write_text(json.dumps(breakdown, indent=2))
    (run_dir / "optimise_meta.json").write_text(json.dumps(meta, indent=2))

    return OptimiseResult(assignment=A_best, objective_breakdown=breakdown, meta=meta)


# =============================================================================
# CLI
# =============================================================================


def _parse_args(argv: Optional[List[str]] = None):
    import argparse

    ap = argparse.ArgumentParser(description="AWLA Optimiser (ILP seed + HH/LNS)")
    ap.add_argument(
        "--instance", required=True, help="Path to instance folder containing CSVs"
    )
    ap.add_argument("--config", required=True, help="Path to resolved YAML config")
    ap.add_argument("--run_dir", required=True, help="Where to write outputs")
    ap.add_argument("--log", default="INFO")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = _parse_args(argv)
    res = solve_end_to_end(args.instance, args.config, args.run_dir)
    print(
        json.dumps(
            {"objective_breakdown": res.objective_breakdown, "meta": res.meta}, indent=2
        )
    )


if __name__ == "__main__":
    main()
