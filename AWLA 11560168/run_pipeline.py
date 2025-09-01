#!/usr/bin/env python3
"""
run_pipeline.py — End-to-end AWLA runner.

Steps:
1) Use resolved config (from config_loader) or deep-merge on the fly (optional extension).
2) Build assignment via:
   - naive: Scripts/make_naive_assignment.py
   - solve:  Solver.Optimisation.solve_end_to_end  (ILP seed + LNS)
3) Validate with Scripts/checker.py (writes violations.csv, objective_breakdown.json)
4) Generate reports with Solver.Reports (CSV + Excel workbook)
"""
from __future__ import annotations
import argparse, json, sys, importlib.util
from pathlib import Path
import yaml

def import_module_from(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True, help="Instance folder with CSVs")
    ap.add_argument("--resolved", required=True, help="Resolved YAML config (from config_loader)")
    ap.add_argument("--run_dir", required=True, help="Output folder for artefacts")
    ap.add_argument("--method", choices=["naive","solve"], default="naive")
    args = ap.parse_args()

    instance = Path(args.instance)
    resolved = Path(args.resolved)
    run_dir = Path(args.run_dir); run_dir.mkdir(parents=True, exist_ok=True)

    # 1) Assignment
    assignment_csv = run_dir/"assignment.csv"
    if args.method == "naive":
        mk = import_module_from(Path("Scripts/make_naive_assignment.py"), "make_naive")
        mk.main(str(instance), str(resolved), str(assignment_csv))
    else:
        # Optimiser path
        sys.path.insert(0, str(Path(".")))
        from Solver.Optimisation import solve_end_to_end
        res = solve_end_to_end(instance, resolved, run_dir)
        # already wrote assignment + objective breakdown; continue

    # 2) Checker
    ck = import_module_from(Path("Scripts/checker.py"), "checker")
    # Simulate cli: checker.main uses argparse.read from sys.argv
    import sys as _sys
    argv_backup = _sys.argv
    _sys.argv = ["checker.py",
        "--instance_dir", str(instance),
        "--resolved_config", str(resolved),
        "--assignment_csv", str(assignment_csv),
        "--out_dir", str(run_dir)
    ]
    ck.main()
    _sys.argv = argv_backup

    # 3) Reports
    sys.path.insert(0, str(Path(".")))
    from Solver.Reports import generate_reports
    rep_dir = run_dir/"report"
    rep_dir.mkdir(parents=True, exist_ok=True)
    summary = generate_reports(instance, resolved, assignment_csv, rep_dir)

    print(json.dumps({
        "run_dir": str(run_dir),
        "report_dir": str(rep_dir),
        "report_summary": summary
    }, indent=2))

if __name__ == "__main__":
    main()
