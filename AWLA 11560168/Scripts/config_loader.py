
"""
Config loader for AWLA (Academic Workload Allocation).

- Deep-merges: defaults.yaml -> instance parameters.yaml -> CLI overrides
- Validates against JSON Schema (Configs/schema.json)
- Writes the resolved config into the run folder
"""
from __future__ import annotations
import json, copy, argparse, sys
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError as e:
    print("Please install pyyaml: pip install pyyaml", file=sys.stderr); raise

def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_yaml(p: Path) -> Dict[str, Any]:
    return yaml.safe_load(p.read_text()) if p and p.exists() else {}

def validate_schema(cfg: Dict[str, Any], schema_path: Path) -> None:
    # Try jsonschema if present; otherwise run minimal checks
    try:
        import jsonschema
        jsonschema.validate(instance=cfg, schema=json.loads(schema_path.read_text()))
    except ModuleNotFoundError:
        required_top = ["meta","weights","fairness","wam_adapter","solver","checker","reports","logging"]
        missing = [k for k in required_top if k not in cfg]
        if missing:
            raise ValueError(f"Config missing top-level sections: {missing}")
        # Minimal sanity checks
        if cfg["meta"].get("units") not in ("hours","points"):
            raise ValueError("meta.units must be 'hours' or 'points'")
        if cfg["fairness"].get("metric") not in ("L1","L2"):
            raise ValueError("fairness.metric must be 'L1' or 'L2'")
        if cfg["solver"].get("method") not in ("ilp","matheuristic","hyper"):
            raise ValueError("solver.method must be one of: ilp, matheuristic, hyper")

def load_config(defaults_path: Path, instance_path: Path, schema_path: Path, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    defaults = load_yaml(defaults_path)
    instance = load_yaml(instance_path)
    cfg = deep_merge(defaults, instance)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    validate_schema(cfg, schema_path)
    return cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defaults", type=str, required=True)
    ap.add_argument("--config", type=str, required=True, help="Instance parameters.yaml")
    ap.add_argument("--schema", type=str, required=True)
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--override", type=str, default="", help="JSON string of CLI overrides")
    args = ap.parse_args()

    defaults = Path(args.defaults)
    config = Path(args.config)
    schema = Path(args.schema)
    run_dir = Path(args.run_dir); run_dir.mkdir(parents=True, exist_ok=True)

    overrides = json.loads(args.override) if args.override else None
    cfg = load_config(defaults, config, schema, overrides)

    # Write resolved config
    resolved = run_dir / "resolved_config.yaml"
    with resolved.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Wrote {resolved}")

if __name__ == "__main__":
    main()
