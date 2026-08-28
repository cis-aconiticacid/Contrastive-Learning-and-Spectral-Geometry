"""Aggregate multi-seed run outputs into a single JSON + CSV with mean±std.

Inputs (per seed): a run directory containing
  - csv/version_0/metrics.csv          (training)
  - jacobian_probe.jsonl               (per-step Jacobian summary)
  - engineering_metrics.json           (peak mem, wall-clock, params)
  - eval.json                          (rollout MSE + probing)
  - eval_rollout_per_step.csv          (rollout MSE per horizon)

Output:
  {out_dir}/aggregate.json   nested mean ± std
  {out_dir}/training_curve.csv         step, total_loss_{mean,std}, ...
  {out_dir}/jacobian_curve.csv         step, spectral_{mean,std}, stable_rank_{mean,std}, ...
  {out_dir}/rollout_summary.csv        horizon, mse_{mean,std}, n_seeds

Usage:
    python aggregate.py \
      --variant baseline \
      --runs /path/to/baseline_seed0 /path/to/baseline_seed1 /path/to/baseline_seed2 \
      --out-dir aggregated/baseline
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as stats
from pathlib import Path
from typing import Dict, List


def _safe_mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], 0.0
    return stats.mean(xs), stats.stdev(xs)


def load_training_curve(run_dir: Path) -> Dict[int, Dict[str, float]]:
    """Returns step -> dict of metrics. Joins train + val rows by step."""
    p = run_dir / "csv" / "version_0" / "metrics.csv"
    if not p.exists():
        return {}
    by_step: Dict[int, Dict[str, float]] = {}
    with p.open() as f:
        for row in csv.DictReader(f):
            try:
                step = int(row["step"])
            except (KeyError, ValueError):
                continue
            d = by_step.setdefault(step, {})
            for k, v in row.items():
                if v in ("", None) or k == "step":
                    continue
                try:
                    d[k] = float(v)
                except ValueError:
                    pass
    return by_step


def load_jacobian(run_dir: Path) -> Dict[int, Dict[str, float]]:
    p = run_dir / "jacobian_probe.jsonl"
    if not p.exists():
        return {}
    by_step = {}
    with p.open() as f:
        for line in f:
            d = json.loads(line)
            by_step[int(d["step"])] = {
                "spectral_norm": d["mean_spectral_norm"],
                "stable_rank": d["mean_stable_rank"],
                "effective_rank": d["mean_effective_rank"],
                "n_svs_above_1pct": d["mean_n_svs_above_1pct"],
            }
    return by_step


def load_probing(run_dir: Path) -> Dict[int, Dict[str, float]]:
    p = run_dir / "probing.jsonl"
    if not p.exists():
        return {}
    by_step = {}
    with p.open() as f:
        for line in f:
            d = json.loads(line)
            row = {"overall_r2": d["overall_r2"], "mse": d["mse"]}
            for i, v in enumerate(d.get("per_dim_r2", [])):
                row[f"r2_dim{i}"] = v
            by_step[int(d["step"])] = row
    return by_step


def load_latent_cov(run_dir: Path) -> Dict[int, Dict[str, float]]:
    p = run_dir / "latent_cov.jsonl"
    if not p.exists():
        return {}
    by_step = {}
    with p.open() as f:
        for line in f:
            d = json.loads(line)
            by_step[int(d["step"])] = {
                "trace": d["trace"],
                "spectral_eigval": d["spectral_eigval"],
                "stable_rank": d["stable_rank"],
                "effective_rank": d["effective_rank"],
                "n_eigvals_above_1pct": d["n_eigvals_above_1pct"],
            }
    return by_step


def load_engineering(run_dir: Path) -> Dict[str, float]:
    p = run_dir / "engineering_metrics.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def load_eval(run_dir: Path) -> Dict:
    p = run_dir / "eval.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------

def aggregate_training(curves: List[Dict[int, Dict[str, float]]]) -> List[Dict]:
    """Aggregate per-step training metrics across seeds.

    Only keeps steps present in ALL seeds (intersection) for clean mean ± std.
    """
    if not curves:
        return []
    common_steps = sorted(set.intersection(*[set(c.keys()) for c in curves]))
    fields = sorted({k for c in curves for d in c.values() for k in d})
    rows = []
    for step in common_steps:
        row: Dict[str, float] = {"step": step, "n_seeds": len(curves)}
        for fld in fields:
            vals = [c[step].get(fld) for c in curves if step in c]
            mean, std = _safe_mean_std(vals)
            if mean is not None:
                row[f"{fld}_mean"] = mean
                row[f"{fld}_std"] = std
        rows.append(row)
    return rows


def aggregate_jacobian(curves: List[Dict[int, Dict[str, float]]]) -> List[Dict]:
    """Field-generic aggregation: discovers all numeric fields across all
    seed × step pairs and produces a {field}_mean / {field}_std row per step."""
    if not curves:
        return []
    common_steps = sorted(set.intersection(*[set(c.keys()) for c in curves]))
    # collect all field names that are numeric in at least one row
    all_fields = set()
    for c in curves:
        for d in c.values():
            for k, v in d.items():
                if isinstance(v, (int, float)):
                    all_fields.add(k)
    fields = sorted(all_fields)
    rows = []
    for step in common_steps:
        row = {"step": step, "n_seeds": len(curves)}
        for fld in fields:
            vals = [c[step].get(fld) for c in curves if step in c and fld in c[step]]
            vals = [v for v in vals if isinstance(v, (int, float))]
            if not vals: continue
            mean, std = _safe_mean_std(vals)
            if mean is not None:
                row[f"{fld}_mean"] = mean
                row[f"{fld}_std"] = std
        rows.append(row)
    return rows


def aggregate_engineering(engs: List[Dict]) -> Dict:
    out = {"n_seeds": len(engs)}
    for fld in ["predictor_params", "encoder_params", "total_params",
                "peak_gpu_memory_mb", "wall_clock_s", "global_step"]:
        vals = [e.get(fld) for e in engs if fld in e]
        mean, std = _safe_mean_std(vals)
        out[f"{fld}_mean"] = mean
        out[f"{fld}_std"] = std
    return out


def aggregate_eval(evals: List[Dict]) -> Dict:
    """Multi-step rollout MSE (per horizon) + probing R² mean±std."""
    rollout_by_h: Dict[int, List[float]] = {}
    overall_r2 = []
    per_dim_r2: List[List[float]] = []
    probe_mse = []
    for ev in evals:
        for r in ev.get("rollout_results", []):
            rollout_by_h.setdefault(r["horizon"], []).append(r["mse_per_step"])
        if "probe" in ev:
            overall_r2.append(ev["probe"]["overall_r2"])
            per_dim_r2.append(ev["probe"]["per_dim_r2"])
            probe_mse.append(ev["probe"]["mse"])
    rollout_summary = []
    for h in sorted(rollout_by_h):
        vals = rollout_by_h[h]
        mean, std = _safe_mean_std(vals)
        rollout_summary.append({"horizon": h, "mse_mean": mean, "mse_std": std,
                                "n_seeds": len(vals)})
    probe = {"n_seeds": len(overall_r2)}
    overall_mean, overall_std = _safe_mean_std(overall_r2)
    probe["overall_r2_mean"] = overall_mean
    probe["overall_r2_std"] = overall_std
    pm_mean, pm_std = _safe_mean_std(probe_mse)
    probe["mse_mean"] = pm_mean
    probe["mse_std"] = pm_std
    if per_dim_r2 and all(len(r) == len(per_dim_r2[0]) for r in per_dim_r2):
        per_dim_summary = []
        for d in range(len(per_dim_r2[0])):
            vals = [r[d] for r in per_dim_r2]
            m, s = _safe_mean_std(vals)
            per_dim_summary.append({"dim": d, "r2_mean": m, "r2_std": s})
        probe["per_dim"] = per_dim_summary
    return {"rollout": rollout_summary, "probe": probe}


def write_csv(rows: List[Dict], path: Path):
    if not rows:
        return
    # Collect every field that appears in any row
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="variant tag for the JSON")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="per-seed run directories (each containing csv/, jacobian_probe.jsonl, etc.)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [Path(r) for r in args.runs]

    train_curves = [load_training_curve(r) for r in runs]
    jac_curves = [load_jacobian(r) for r in runs]
    probe_curves = [load_probing(r) for r in runs]
    cov_curves = [load_latent_cov(r) for r in runs]
    engs = [load_engineering(r) for r in runs]
    evals = [load_eval(r) for r in runs]

    train_agg = aggregate_training(train_curves)
    jac_agg = aggregate_jacobian(jac_curves)
    probe_agg = aggregate_jacobian(probe_curves)  # same shape; reuse
    cov_agg = aggregate_jacobian(cov_curves)
    eng_agg = aggregate_engineering(engs)
    eval_agg = aggregate_eval(evals)

    # JSON dump
    summary = {
        "variant": args.variant,
        "n_seeds": len(runs),
        "run_dirs": [str(r) for r in runs],
        "engineering": eng_agg,
        "eval": eval_agg,
        "training_curve_n_steps": len(train_agg),
        "jacobian_curve_n_steps": len(jac_agg),
    }
    (out_dir / "aggregate.json").write_text(json.dumps(summary, indent=2))
    write_csv(train_agg, out_dir / "training_curve.csv")
    write_csv(jac_agg, out_dir / "jacobian_curve.csv")
    write_csv(probe_agg, out_dir / "probing_curve.csv")
    write_csv(cov_agg, out_dir / "latent_cov_curve.csv")
    write_csv(eval_agg["rollout"], out_dir / "rollout_summary.csv")

    print(f"Aggregated {len(runs)} seeds into {out_dir}/")
    print(f"  training_curve.csv:  {len(train_agg)} rows")
    print(f"  jacobian_curve.csv:  {len(jac_agg)} rows")
    print(f"  probing_curve.csv:   {len(probe_agg)} rows")
    print(f"  latent_cov_curve.csv: {len(cov_agg)} rows")
    print(f"  rollout_summary.csv: {len(eval_agg['rollout'])} horizons")
    print(f"  engineering: predictor_params {eng_agg.get('predictor_params_mean')} "
          f"peak_mb {eng_agg.get('peak_gpu_memory_mb_mean'):.1f} "
          f"wall_s {eng_agg.get('wall_clock_s_mean'):.1f}")


if __name__ == "__main__":
    main()
