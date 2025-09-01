# BM-11560168
Final Code for ERP
# AWLA (v1.0)

**Purpose.** AWLA builds defensible teaching/service allocations by assigning **duties** to **staff** under hard constraints (coverage, eligibility, capacity) and soft preferences, while optimising **fairness** (hours close to targets; cohort dispersion) and **stability** (minimal changes vs baseline).  
This repo contains a complete, reproducible pipeline (config → solve → check → report) with a single default seed (**42**).

## Table of Contents
- [Installation](#installation)
- [Features](#features) · [Benefits](#benefits)
- [Using AWLA](#using-awla)
- [Data Format](#data-format)
- [Outputs](#outputs)
- [Reproducibility](#reproducibility)

---

## Installation

bash
# clone your repo
git clone
cd REPO

# create an isolated environment (recommended)
python -m venv .venv
source .venv/bin/activate            
Windows: .venv\Scripts\activate

# install runtime requirements
pip install -r requirements.txt

## Features

Full pipeline: config resolution → solver (naive or optimiser) → checker → reports.

Fairness/stability KPIs: hours-based equity (L1), banding (GREEN/AMBER), cohort dispersion (RMSE), % unchanged and hours moved vs baseline.

Single-seed reproducibility: default seed = 42 baked into Configs/defaults.yaml and resolved configs.

Auditable artefacts: exports allocations, violations, objective breakdown, and an Excel/CSV report pack.

## Benefits

Defensible: transparent fairness metrics and violation logs.

Practical: stability controls to avoid disruptive reallocations.

Reproducible: deterministic seed, fixed solver budgets (if set in config), and a clean run folder.

## Using AWLA
**1) Prepare your instance**

Place the instance under Dataset/instances/<instance_name>/:

A YAML file: parameters.yaml

CSVs listed in Data Format

A working synthetic example ships with the repo:

Dataset/instances/awla_synth_instance_v1/
  parameters.yaml
  staff.csv
  modules.csv
  duties.csv
  eligibility.csv
  availability.csv
  preferences.csv
  conflicts_staff_pairs.csv
  preallocations.csv
  baseline_plan.csv
  baseline_not_eligible.csv
  fairness_targets.csv
  sanity_issues.csv

**2) Run the pipeline (naive vs optimiser)**

**2.1 Naive baseline (fast sanity check; does not require ILP):**

python run_pipeline.py \
  --instance Dataset/instances/awla_synth_instance_v1/parameters.yaml \
  --run_dir Experiments/runs/awla_seed42_naive \
  --method naive


**2.2 Optimiser (ILP seed + hyper-heuristic / LNS):**

python run_pipeline.py \
  --instance Dataset/instances/awla_synth_instance_v1/parameters.yaml \
  --run_dir Experiments/runs/awla_seed42_solve \
  --method solve

## Data Format

Minimum viable CSVs (extend freely):

**staff.csv**
staff_id, name, role, grade, team, fte, min_load, max_load, target_hours

**modules.csv**
module_id, module_code, name, level, team, semester

**duties.csv**
duty_id, module_id, activity_type, period, required_hours

**eligibility.csv**
staff_id, duty_id, period, eligible (0/1)

**availability.csv**
staff_id, period, available (0/1 or numeric weight)

**preferences.csv (optional)**
staff_id, duty_id, preference_weight (higher = preferred)

**conflicts_staff_pairs.csv (optional)**
staff_id_1, staff_id_2 (avoid pairing/splitting where required)

**preallocations.csv (optional)**
staff_id, duty_id, period, hours_fixed

**baseline_plan.csv (optional but recommended for stability KPIs)**
staff_id, duty_id, period, hours_baseline

**fairness_targets.csv**
staff_id, target_hours

##Keep IDs consistent across files. Use hours (or points) consistently with the unit declared in config.

## Outputs

Each run writes a dedicated folder under --run_dir, including:
assignment.csv — final allocation (staff × duty × period with hours)
violations.csv — hard/soft checks with magnitudes and costs
objective_breakdown.json — term-by-term objective totals + weighted sum
report/ — CSV/Excel summaries

## Reproducibility

Seed: fixed at 42 by default (set in Configs/defaults.yaml and/or the resolved config).
Config resolution: written to the run directory to freeze the exact settings used.
Solver budgets: if present in your configs (e.g., ILP time-limit, LNS time, LAHC length), they are recorded alongside the run metadata.
Environment: capture versions with pip freeze > requirements-freeze.txt if publishing results.

## Support

Open a GitHub Issue with:
OS + Python version
A minimal instance (small CSVs)
The resolved config from your run directory
Command used and the error/traceback

## License
MIT

