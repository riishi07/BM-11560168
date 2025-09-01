"""Problem module for AWLA (Academic Workload Allocation)

Production‑ready loaders, schema/PK/FK validation, feasibility gate,
WAM adapter (targets), and pre‑allocation accounting.

This module is the authoritative entry point for turning a CSV+YAML
*instance package* into an in‑memory, validated `Problem` object that the
optimiser can consume.

=========================
Quick start (from CLI)
=========================
$ python -m Solver.Problem \
    --instance Dataset/instances/awla_synth_instance_v1 \
    --config Experiments/runs/my_run/resolved_config.yaml \
    --out Experiments/runs/my_run/problem_summary.json

- Validates the dataset + config
- Computes staff targets using the WAM adapter (TR/TS base × FTE × overrides),
  with optional preallocation crediting
- Performs a feasibility check (required hours ≤ total capacity)
- Writes a compact JSON summary with dataset hashes & key metrics

=========================
How targets are computed
=========================
Let base(TR) and base(TS) come from `config['wam_adapter']['default_target_hours']`.
For each staff s:
  T_s = base(contract_type_s) × FTE_s
If `fairness_targets.csv` has an override for s, it takes precedence.
If `wam_adapter.preallocations_mode == "credit"`, then we subtract
preallocated policy hours recorded in `preallocations.csv`:
  T_s ← max(0, T_s − prealloc_hours(s))

=========================
Data contract (required CSVs)
=========================
- staff.csv:        staff_id, name, contract_type, fte, grade, team, campus, line_manager_id, target_hours, max_hours
- modules.csv:      module_id, title, credits, hours_required_total, period, campus, programme, indivisible
- duties.csv:       module_id, duty_id, duty_desc, hours_required, duty_type
- eligibility.csv:  staff_id, module_id, duty_id(±empty), eligible, competency_level
- availability.csv: staff_id, period, availability ∈ {0, 0.5, 1}
- preferences.csv:  pref_type ∈ {staff_module, staff_duty, staff_period}, staff_id, module_id, duty_id, period, penalty
- conflicts_staff_pairs.csv: staff_id_1, staff_id_2, conflict_type, weight
Optional:
- fairness_targets.csv: staff_id, target_hours_override
- preallocations.csv:   staff_id, category, hours ≥ 0, notes
- baseline_plan.csv:    module_id, duty_id, staff_id, hours_assigned ≥ 0

This module checks PK/FK integrity and basic domain constraints and raises
specific exceptions with actionable messages.

Author: AWLA Team
License: MIT
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import hashlib
import json
import logging

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class ProblemError(Exception):
    """Base class for problem errors."""


class ProblemSchemaError(ProblemError):
    """Raised when required columns are missing or PK/FK checks fail."""


class ProblemFeasibilityError(ProblemError):
    """Raised when the instance fails the pre‑solve feasibility gate."""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _file_hash(path: Path, algo: str = "sha1") -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"{algo}:{h.hexdigest()}"


def _require_columns(df: pd.DataFrame, required: Iterable[str], fname: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ProblemSchemaError(f"{fname}: missing required columns: {missing}")


def _assert_unique(df: pd.DataFrame, cols: Iterable[str], fname: str, label: str) -> None:
    dups = df.duplicated(list(cols), keep=False)
    if dups.any():
        sample = df.loc[dups, list(cols)].head(5).to_dict(orient="records")
        raise ProblemSchemaError(
            f"{fname}: duplicate {label} rows detected (showing up to 5): {sample}"
        )


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProblemSummary:
    instance_dir: str
    config_path: str
    data_hashes: Mapping[str, str]
    n_staff: int
    n_modules: int
    n_duties: int
    periods: List[str]
    required_hours_total: float
    staff_capacity_total: float
    min_eligible_per_duty: int

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)


class Problem:
    """Validated AWLA problem loaded from CSVs + resolved config.

    Attributes
    ----------
    config : dict
        Resolved configuration (merged defaults→instance→overrides).
    instance_dir : Path
        Folder containing the dataset CSVs.
    data : dict[str, pd.DataFrame]
        Loaded dataframes keyed by logical name (e.g., "staff", "modules").
    targets : dict[str, float]
        Effective per‑staff target hours after applying WAM adapter + overrides
        and preallocation credit (if enabled).
    summary : ProblemSummary
        High‑level metrics and file hashes for provenance.
    """

    # ---- Required schemas (minimum viable set) ----
    _REQUIRED = {
        "staff": [
            "staff_id",
            "name",
            "contract_type",
            "fte",
            "grade",
            "team",
            "campus",
            "line_manager_id",
            "target_hours",
            "max_hours",
        ],
        "modules": [
            "module_id",
            "title",
            "credits",
            "hours_required_total",
            "period",
            "campus",
            "programme",
            "indivisible",
        ],
        "duties": [
            "module_id",
            "duty_id",
            "duty_desc",
            "hours_required",
            "duty_type",
        ],
        "eligibility": [
            "staff_id",
            "module_id",
            "duty_id",
            "eligible",
            "competency_level",
        ],
        "availability": ["staff_id", "period", "availability"],
        "preferences": ["pref_type", "staff_id", "module_id", "duty_id", "period", "penalty"],
        "conflicts_staff_pairs": ["staff_id_1", "staff_id_2", "conflict_type", "weight"],
    }

    _OPTIONAL = {
        "preallocations": ["staff_id", "category", "hours", "notes"],
        "fairness_targets": ["staff_id", "target_hours_override"],
        "baseline_plan": ["module_id", "duty_id", "staff_id", "hours_assigned"],
    }

    def __init__(self, instance_dir: Path, config: Mapping[str, Any]):
        self.instance_dir = Path(instance_dir)
        self.config = dict(config)
        self.data: Dict[str, pd.DataFrame] = {}
        self.targets: Dict[str, float] = {}
        self.summary: Optional[ProblemSummary] = None
        self._log = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @classmethod
    def from_paths(cls, instance: Path | str, config_path: Path | str) -> "Problem":
        """Convenience constructor from filesystem paths.

        Parameters
        ----------
        instance : Path | str
            Folder containing the dataset CSVs.
        config_path : Path | str
            Path to a resolved YAML configuration.
        """
        with Path(config_path).open("r") as f:
            cfg = yaml.safe_load(f)
        return cls(Path(instance), cfg)

    def load(self) -> "Problem":
        """Load all CSVs into dataframes and run schema validation.

        Raises
        ------
        ProblemSchemaError
            If required files/columns are missing or PK/FK checks fail.
        """
        # Required files
        for name in self._REQUIRED:
            path = self.instance_dir / f"{name}.csv"
            if not path.exists():
                raise ProblemSchemaError(f"Missing required file: {path}")
            df = pd.read_csv(path)
            _require_columns(df, self._REQUIRED[name], path.name)
            self.data[name] = df
        # Optional files
        for name, cols in self._OPTIONAL.items():
            path = self.instance_dir / f"{name}.csv"
            if path.exists():
                df = pd.read_csv(path)
                _require_columns(df, cols, path.name)
            else:
                df = pd.DataFrame(columns=cols)
            self.data[name] = df

        self._normalize_types()
        self._validate_pk_fk()
        self._validate_domains()
        self._validate_hours_consistency()
        self._validate_period_coverage()
        # Apply preallocations to reduce duty requirements
        self._apply_preallocations_to_requirements()

        # After structural validation, compute targets & feasibility
        self.targets = self._compute_targets()
        self._feasibility_gate()
        self.summary = self._build_summary()
        return self

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _normalize_types(self) -> None:
        """Coerce basic dtypes and trim whitespace where sensible."""
        # Strip surrounding spaces on key string columns
        for name in ["staff", "modules", "duties", "eligibility", "availability", "preferences", "conflicts_staff_pairs"]:
            df = self.data[name]
            for col in df.columns:
                if pd.api.types.is_object_dtype(df[col]):
                    self.data[name][col] = df[col].astype(str).str.strip()

        # Numeric coercions with safe fallback
        self.data["staff"]["fte"] = pd.to_numeric(self.data["staff"]["fte"], errors="coerce")
        self.data["staff"]["max_hours"] = pd.to_numeric(self.data["staff"]["max_hours"], errors="coerce")
        self.data["modules"]["hours_required_total"] = pd.to_numeric(
            self.data["modules"]["hours_required_total"], errors="coerce"
        )
        self.data["modules"]["indivisible"] = pd.to_numeric(self.data["modules"]["indivisible"], errors="coerce").fillna(0).astype(int)
        self.data["duties"]["hours_required"] = pd.to_numeric(self.data["duties"]["hours_required"], errors="coerce")
        self.data["eligibility"]["eligible"] = pd.to_numeric(self.data["eligibility"]["eligible"], errors="coerce").fillna(0).astype(int)
        self.data["eligibility"]["competency_level"] = pd.to_numeric(self.data["eligibility"]["competency_level"], errors="coerce").fillna(0).astype(int)
        self.data["availability"]["availability"] = pd.to_numeric(self.data["availability"]["availability"], errors="coerce")
        self.data["preferences"]["penalty"] = pd.to_numeric(self.data["preferences"]["penalty"], errors="coerce")
        if not self.data["preallocations"].empty:
            self.data["preallocations"]["hours"] = pd.to_numeric(self.data["preallocations"]["hours"], errors="coerce").fillna(0)
        if not self.data["fairness_targets"].empty:
            self.data["fairness_targets"]["target_hours_override"] = pd.to_numeric(
                self.data["fairness_targets"]["target_hours_override"], errors="coerce"
            )
        if not self.data["baseline_plan"].empty:
            self.data["baseline_plan"]["hours_assigned"] = pd.to_numeric(
                self.data["baseline_plan"]["hours_assigned"], errors="coerce"
            )

    def _validate_pk_fk(self) -> None:
        """Primary/foreign key checks across all core tables."""
        staff = self.data["staff"]
        modules = self.data["modules"]
        duties = self.data["duties"]
        elig = self.data["eligibility"]

        # PKs
        _assert_unique(staff, ["staff_id"], "staff.csv", "staff_id")
        _assert_unique(modules, ["module_id"], "modules.csv", "module_id")
        _assert_unique(duties, ["duty_id"], "duties.csv", "duty_id")

        # FKs
        # duties.module_id -> modules.module_id
        missing_mods = set(duties["module_id"]) - set(modules["module_id"])
        if missing_mods:
            raise ProblemSchemaError(f"duties.csv: unknown module_id(s): {sorted(missing_mods)}")

        # eligibility FKs (allow module‑level rows with empty duty_id)
        staff_ids = set(staff["staff_id"])  # for speed
        module_ids = set(modules["module_id"])
        duty_ids = set(duties["duty_id"])
        bad_staff = set(elig["staff_id"]) - staff_ids
        if bad_staff:
            raise ProblemSchemaError(f"eligibility.csv: unknown staff_id(s): {sorted(bad_staff)}")
        bad_mods = set(elig["module_id"]) - module_ids
        if bad_mods:
            raise ProblemSchemaError(f"eligibility.csv: unknown module_id(s): {sorted(bad_mods)}")
        # duty_id: permit empty → signifies module‑level eligibility
        nonempty_duty = elig["duty_id"].fillna("").astype(str) != ""
        bad_duties = set(elig.loc[nonempty_duty, "duty_id"]) - duty_ids
        if bad_duties:
            raise ProblemSchemaError(f"eligibility.csv: unknown duty_id(s): {sorted(bad_duties)}")

        # availability FKs
        avail = self.data["availability"]
        bad_staff_av = set(avail["staff_id"]) - staff_ids
        if bad_staff_av:
            raise ProblemSchemaError(f"availability.csv: unknown staff_id(s): {sorted(bad_staff_av)}")

        # preferences FKs by type
        prefs = self.data["preferences"]
        valid_types = {"staff_module", "staff_duty", "staff_period"}
        bad_types = set(prefs["pref_type"]) - valid_types
        if bad_types:
            raise ProblemSchemaError(f"preferences.csv: invalid pref_type(s): {sorted(bad_types)}")
        # staff_module
        sm = prefs[prefs["pref_type"] == "staff_module"]
        if not sm.empty:
            bad = (set(sm["staff_id"]) - staff_ids) | (set(sm["module_id"]) - module_ids)
            if bad:
                raise ProblemSchemaError("preferences.csv: staff_module rows contain unknown staff/module IDs")
        # staff_duty
        sd = prefs[prefs["pref_type"] == "staff_duty"]
        if not sd.empty:
            bad = (set(sd["staff_id"]) - staff_ids) | (set(sd["duty_id"]) - duty_ids)
            if bad:
                raise ProblemSchemaError("preferences.csv: staff_duty rows contain unknown staff/duty IDs")
        # staff_period — validate period later against config/meta

        # conflicts FKs
        conf = self.data["conflicts_staff_pairs"]
        bad_conf = (set(conf["staff_id_1"]) | set(conf["staff_id_2"])) - staff_ids
        if bad_conf:
            raise ProblemSchemaError(f"conflicts_staff_pairs.csv: unknown staff IDs: {sorted(bad_conf)}")
        # self‑pairs
        if (conf["staff_id_1"] == conf["staff_id_2"]).any():
            raise ProblemSchemaError("conflicts_staff_pairs.csv: self‑conflicts are not allowed")

        # fairness_targets
        ft = self.data["fairness_targets"]
        if not ft.empty:
            bad_ft = set(ft["staff_id"]) - staff_ids
            if bad_ft:
                raise ProblemSchemaError(f"fairness_targets.csv: unknown staff IDs: {sorted(bad_ft)}")

        # preallocations
        pre = self.data["preallocations"]
        if not pre.empty:
            bad_pre = set(pre["staff_id"]) - staff_ids
            if bad_pre:
                raise ProblemSchemaError(f"preallocations.csv: unknown staff IDs: {sorted(bad_pre)}")

        # baseline_plan
        base = self.data["baseline_plan"]
        if not base.empty:
            bad_b_staff = set(base["staff_id"]) - staff_ids
            bad_b_duty = set(base["duty_id"]) - duty_ids
            bad_b_mod = set(base["module_id"]) - module_ids
            errs = []
            if bad_b_staff:
                errs.append(f"unknown staff: {sorted(bad_b_staff)}")
            if bad_b_duty:
                errs.append(f"unknown duties: {sorted(bad_b_duty)}")
            if bad_b_mod:
                errs.append(f"unknown modules: {sorted(bad_b_mod)}")
            if errs:
                raise ProblemSchemaError("baseline_plan.csv: " + "; ".join(errs))

    def _validate_domains(self) -> None:
        """Domain checks for value ranges and enumerations."""
        cfg_periods = list(self.config.get("meta", {}).get("periods", []))
        modules = self.data["modules"]
        availability = self.data["availability"]
        preferences = self.data["preferences"]

        # Periods: if config.meta.periods is provided, enforce membership
        if cfg_periods:
            bad_mod_periods = set(modules["period"]) - set(cfg_periods)
            if bad_mod_periods:
                raise ProblemSchemaError(
                    f"modules.csv: found periods not declared in config.meta.periods: {sorted(bad_mod_periods)}"
                )
            bad_av_periods = set(availability["period"]) - set(cfg_periods)
            if bad_av_periods:
                raise ProblemSchemaError(
                    f"availability.csv: found periods not declared in config.meta.periods: {sorted(bad_av_periods)}"
                )
            sp = preferences[preferences["pref_type"] == "staff_period"]
            bad_sp = set(sp["period"]) - set(cfg_periods)
            if not sp.empty and bad_sp:
                raise ProblemSchemaError(
                    f"preferences.csv (staff_period): unknown period values: {sorted(bad_sp)}"
                )

        # Availability domain {0, 0.5, 1}
        bad_av = set(availability["availability"]) - {0.0, 0.5, 1.0}
        if bad_av:
            raise ProblemSchemaError(f"availability.csv: availability must be one of {{0, 0.5, 1}}; bad values: {sorted(bad_av)}")

        # Basic numeric sanity
        if (self.data["duties"]["hours_required"] <= 0).any():
            raise ProblemSchemaError("duties.csv: hours_required must be > 0 for all rows")
        if (self.data["modules"]["hours_required_total"] <= 0).any():
            raise ProblemSchemaError("modules.csv: hours_required_total must be > 0 for all rows")
        if (self.data["staff"]["fte"] <= 0).any():
            raise ProblemSchemaError("staff.csv: fte must be > 0 for all staff")
        if not self.data["preallocations"].empty and (self.data["preallocations"]["hours"] < 0).any():
            raise ProblemSchemaError("preallocations.csv: hours must be ≥ 0")
        if not self.data["fairness_targets"].empty and (self.data["fairness_targets"]["target_hours_override"] <= 0).any():
            raise ProblemSchemaError("fairness_targets.csv: target_hours_override must be > 0 where provided")

    def _validate_hours_consistency(self) -> None:
        """Ensure Σ duty hours per module equals modules.hours_required_total."""
        duty_sum = self.data["duties"].groupby("module_id")["hours_required"].sum().round(6)
        mod_total = self.data["modules"].set_index("module_id")["hours_required_total"].round(6)
        bad = (duty_sum != mod_total)
        if bad.any():
            diffs = (duty_sum[bad] - mod_total[bad]).to_dict()
            raise ProblemSchemaError(
                f"Duties vs modules mismatch (Σ duty hours ≠ module total) for modules: {diffs}"
            )

    def _validate_period_coverage(self) -> None:
        """Warn if availability rows are missing for some staff/periods used by modules."""
        periods_used = sorted(self.data["modules"]["period"].dropna().unique().tolist())
        avail = self.data["availability"]
        expected = len(periods_used)
        counts = avail.groupby("staff_id")["period"].nunique()
        n_missing = int((counts < expected).sum())
        if n_missing:
            self._log.warning(
                "availability.csv: %d staff lack rows for all periods %s",
                n_missing,
                periods_used,
            )

    
    
    def _apply_preallocations_to_requirements(self) -> None:
        """Reduce duty requirements by preallocated hours *only if* preallocations
        specify module_id & duty_id. Otherwise, skip (staff-level admin credits are
        already handled via target reduction)."""
        pre = self.data.get("preallocations", pd.DataFrame())
        if pre is None or pre.empty:
            return
        needed_cols = {"module_id","duty_id","hours"}
        if not needed_cols.issubset(set(pre.columns)):
            # No duty-level preallocations -> nothing to reduce here
            return
        # Aggregate by (module_id, duty_id)
        grp = pre.groupby(["module_id","duty_id"])["hours"].sum().reset_index()
        duties = self.data["duties"]
        duties = duties.merge(grp, on=["module_id","duty_id"], how="left", suffixes=("", "_pre")).fillna({"hours_pre": 0.0})
        duties["hours_required_eff"] = (duties["hours_required"].astype(float) - duties["hours_pre"].astype(float)).clip(lower=0.0)
        self.data["duties"] = duties.drop(columns=["hours_pre"])
# ------------------------------------------------------------------
    # Targets & feasibility
    # ------------------------------------------------------------------
    def _compute_targets(self) -> Dict[str, float]:
        """Compute effective per‑staff targets from config + overrides.

        Returns
        -------
        dict
            staff_id → target_hours (non‑negative floats)
        """
        cfg = self.config
        base_TR = float(cfg.get("wam_adapter", {}).get("default_target_hours", {}).get("TR", 0.0))
        base_TS = float(cfg.get("wam_adapter", {}).get("default_target_hours", {}).get("TS", 0.0))
        prealloc_mode = cfg.get("wam_adapter", {}).get("preallocations_mode", "credit")

        staff = self.data["staff"].copy()
        base_map = {"TR": base_TR, "TS": base_TS}
        staff["_base"] = staff["contract_type"].map(base_map).fillna(base_TR)
        staff["_target"] = staff["_base"] * staff["fte"].astype(float)

        # Person‑level overrides
        ft = self.data["fairness_targets"]
        if not ft.empty:
            ov = ft.dropna().set_index("staff_id")["target_hours_override"].to_dict()
            staff["_target"] = staff.apply(lambda r: ov.get(r["staff_id"], r["_target"]), axis=1)

        # Preallocations credit (if enabled)
        if prealloc_mode == "credit" and not self.data["preallocations"].empty:
            pre = self.data["preallocations"].groupby("staff_id")["hours"].sum().to_dict()
            staff["_target"] = staff.apply(lambda r: max(0.0, float(r["_target"]) - float(pre.get(r["staff_id"], 0.0))), axis=1)

        # Clip at zero (guard against negatives)
        staff["_target"] = staff["_target"].clip(lower=0.0)
        return dict(zip(staff["staff_id"], staff["_target"]))

    def _feasibility_gate(self) -> None:
        """Simple necessary condition: total required hours ≤ total staff capacity.

        Uses `staff.max_hours` when provided; if a row is NaN/empty, it is
        treated as infinite capacity (i.e., it does not constrain the check).
        """
        required = float(self.data["duties"]["hours_required"].sum())
        # Sum only finite capacities; ignore NaNs as "no cap"
        staff = self.data["staff"]
        finite_caps = pd.to_numeric(staff["max_hours"], errors="coerce").dropna()
        capacity = float(finite_caps.sum()) if not finite_caps.empty else float("inf")
        if required > capacity:
            raise ProblemFeasibilityError(
                f"Feasibility gate failed: required hours {required:.2f} exceed total finite staff capacity {capacity:.2f}"
            )

    # ------------------------------------------------------------------
    # Summary / export
    # ------------------------------------------------------------------
    def _build_summary(self) -> ProblemSummary:
        hashes: Dict[str, str] = {}
        for name in list(self._REQUIRED.keys()) + list(self._OPTIONAL.keys()):
            path = self.instance_dir / f"{name}.csv"
            if path.exists():
                hashes[name] = _file_hash(path)
        cfg_path = self.config.get("_config_path")  # may be injected by caller
        if cfg_path and Path(cfg_path).exists():
            hashes["config"] = _file_hash(Path(cfg_path))

        periods = sorted(self.data["modules"]["period"].dropna().unique().tolist())
        elig1 = self.data["eligibility"][self.data["eligibility"]["eligible"] == 1]
        elig_counts = elig1.groupby("duty_id")["staff_id"].nunique()
        min_elig = int(elig_counts.min()) if not elig_counts.empty else 0

        staff_capacity = pd.to_numeric(self.data["staff"]["max_hours"], errors="coerce").dropna().sum()

        return ProblemSummary(
            instance_dir=str(self.instance_dir),
            config_path=str(cfg_path) if cfg_path else "",
            data_hashes=hashes,
            n_staff=int(self.data["staff"].shape[0]),
            n_modules=int(self.data["modules"].shape[0]),
            n_duties=int(self.data["duties"].shape[0]),
            periods=periods,
            required_hours_total=float(self.data["duties"]["hours_required"].sum()),
            staff_capacity_total=float(staff_capacity),
            min_eligible_per_duty=min_elig,
        )

    # Convenience accessors -------------------------------------------------
    def get_dataframe(self, name: str) -> pd.DataFrame:
        return self.data[name].copy()

    def targets_series(self) -> pd.Series:
        return pd.Series(self.targets, name="target_hours")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None):
    import argparse

    ap = argparse.ArgumentParser(description="AWLA Problem loader & validator")
    ap.add_argument("--instance", required=True, help="Path to instance folder containing CSVs")
    ap.add_argument("--config", required=True, help="Path to resolved YAML config")
    ap.add_argument("--out", required=False, default="", help="Where to write problem_summary.json")
    ap.add_argument("--log", required=False, default="INFO", help="Log level (DEBUG/INFO/WARN/ERROR)")
    return ap.parse_args(argv)


def _configure_logging(level: str) -> None:
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(levelname)s %(name)s: %(message)s")


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    _configure_logging(args.log)

    cfg_path = Path(args.config)
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)
    # Inject the config path so it appears in hashes
    cfg["_config_path"] = str(cfg_path)

    prob = Problem(Path(args.instance), cfg).load()

    # Build JSON summary
    summary_json = prob.summary.to_json()
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(summary_json)
        print(f"Wrote {outp}")
    else:
        print(summary_json)


if __name__ == "__main__":
    main()
