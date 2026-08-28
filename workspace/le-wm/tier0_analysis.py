"""Tier-0 follow-up analyses on existing v2 / v3 probe JSONL.

No new forward passes. Reads latent_cov.jsonl + jacobian_probe.jsonl already
collected during training, computes:

  1. Participation Ratio (PR) over training from top-32 latent eigenvalues
     PR = (Σλᵢ)² / Σλᵢ²  — standard "effective dimension" metric.
     Top-32 covers ≥99% of trace at the final step → PR_{top-32} ≈ PR_{full}.

  2. Predictor Jacobian spectral gap from top-32 singular values
     gap2  = σ₁ / σ₂
     gap10 = σ₁ / σ_10
     Quantifies "dominant mode advantage" beyond bulk stable_rank.

  3. SV decay analysis from top-32 singular values
     Power-law fit log(σ_i) = a - β log(i) over i ∈ [1, 32].
     β is a standard intrinsic-dim proxy: large β → fast decay, low effective dim.

Outputs:
  refine-logs/tier0_results.json   — all numbers (per variant × seed × step)
  refine-logs/tier0_pr_vs_step.png
  refine-logs/tier0_sv_decay_final.png
  refine-logs/tier0_spectral_gap.png
"""
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V2_ROOT = Path("/workspace/lewm_autodl_results_v2")
V3_ROOT = Path("/workspace/lewm_autodl_results_v3")
OUT_DIR = Path("/workspace/le-wm/refine-logs")
OUT_JSON = OUT_DIR / "tier0_results.json"

V2_RUNS = {
    "baseline":  ["baseline_seed0", "baseline_seed1", "baseline_seed2"],
    "uvlowr_r4": ["uvlowr_r4_seed0", "uvlowr_r4_seed1", "uvlowr_r4_seed2"],
    "randdiff":  ["randdiff_seed0", "randdiff_seed1", "randdiff_seed2"],
}
V3_RUNS = {
    "baseline":  ["frozen_baseline_seed0", "frozen_baseline_seed1"],
    "uvlowr_r4": ["frozen_uvlowr_r4_seed0", "frozen_uvlowr_r4_seed1"],
    "randdiff":  ["frozen_randdiff_seed0", "frozen_randdiff_seed1"],
}
COLORS = {"baseline": "C0", "uvlowr_r4": "C1", "randdiff": "C2"}


def participation_ratio(eig):
    e = np.asarray(eig, dtype=np.float64)
    e = e[e > 0]
    if e.size == 0:
        return 0.0
    return float((e.sum() ** 2) / (e ** 2).sum())


def load_latent_cov(run_dir):
    """Return list of dicts {step, pr, trace, sr, er, eigvals[32]}."""
    path = run_dir / "latent_cov.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in open(path):
        d = json.loads(line)
        eig = d.get("top_eigvals", d.get("eigvals", []))
        trace = d.get("trace", float(np.sum(eig)))
        rows.append({
            "step": int(d["step"]),
            "pr_top32": participation_ratio(eig),
            "trace_full": float(trace),
            "trace_top32_frac": float(np.sum(eig)) / max(trace, 1e-12),
            "stable_rank_lcov": float(d.get("stable_rank", np.nan)),
            "effective_rank_lcov": float(d.get("effective_rank", np.nan)),
            "eigvals": list(eig),
        })
    return rows


def load_jacobian(run_dir):
    """Return list of dicts {step, mean_top_svs[32], σ₁, σ₂, σ₁₀, sr, er}."""
    path = run_dir / "jacobian_probe.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in open(path):
        d = json.loads(line)
        samples = d.get("samples", [])
        if not samples:
            continue
        sv_per_sample = np.array([s["top_svs"] for s in samples])  # (n_samples, 32)
        mean_svs = sv_per_sample.mean(axis=0)                       # (32,)
        sr = [s["stable_rank"] for s in samples]
        rows.append({
            "step": int(d["step"]),
            "mean_top_svs": mean_svs.tolist(),
            "sigma_1":  float(mean_svs[0]),
            "sigma_2":  float(mean_svs[1]),
            "sigma_10": float(mean_svs[9]),
            "sigma_20": float(mean_svs[19]),
            "sigma_32": float(mean_svs[-1]),
            "gap_1_to_2":  float(mean_svs[0] / max(mean_svs[1], 1e-12)),
            "gap_1_to_10": float(mean_svs[0] / max(mean_svs[9], 1e-12)),
            "gap_1_to_20": float(mean_svs[0] / max(mean_svs[19], 1e-12)),
            "mean_stable_rank": float(np.mean(sr)),
            "mean_effective_rank": float(d.get("mean_effective_rank", np.nan)),
        })
    return rows


def power_law_exponent(svs):
    """Fit log(σ_i) = a − β log(i), return β. i ∈ [1, len(svs)]."""
    svs = np.asarray(svs, dtype=np.float64)
    mask = svs > 0
    if mask.sum() < 4:
        return float("nan")
    i = np.arange(1, len(svs) + 1)[mask]
    s = svs[mask]
    A = np.vstack([np.log(i), np.ones_like(i, dtype=float)]).T
    slope, _ = np.linalg.lstsq(A, np.log(s), rcond=None)[0]
    return float(-slope)


def collect(root, runs_by_variant):
    out = {}
    for variant, run_dirs in runs_by_variant.items():
        out[variant] = []
        for rd in run_dirs:
            p = root / rd
            lcov = load_latent_cov(p)
            jac  = load_jacobian(p)
            if not lcov and not jac:
                continue
            out[variant].append({
                "run": rd,
                "latent_cov": lcov,
                "jacobian": jac,
            })
    return out


def agg_by_step(rows_per_seed, key):
    """Mean ± std across seeds at each shared step."""
    if not rows_per_seed:
        return [], [], []
    steps_sets = [set(r["step"] for r in rows) for rows in rows_per_seed]
    common = sorted(set.intersection(*steps_sets))
    means, stds = [], []
    for st in common:
        vals = []
        for rows in rows_per_seed:
            for r in rows:
                if r["step"] == st:
                    vals.append(r[key])
                    break
        vals = np.array(vals, dtype=float)
        means.append(float(np.nanmean(vals)))
        stds.append(float(np.nanstd(vals)))
    return common, means, stds


def main():
    print("[tier0] collecting v2 ...")
    v2 = collect(V2_ROOT, V2_RUNS)
    print("[tier0] collecting v3 ...")
    v3 = collect(V3_ROOT, V3_RUNS)

    summary = {"v2": {}, "v3": {}, "comparison": {}}

    # --- 1. PR over training ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, label, data in [(axes[0], "v2 (40K joint)", v2),
                            (axes[1], "v3 (8K frozen)", v3)]:
        for variant, runs in data.items():
            lcov_per_seed = [r["latent_cov"] for r in runs]
            steps, means, stds = agg_by_step(lcov_per_seed, "pr_top32")
            if not steps:
                continue
            means_arr = np.array(means); stds_arr = np.array(stds)
            ax.plot(steps, means_arr, label=variant, color=COLORS[variant], lw=1.8)
            ax.fill_between(steps, means_arr - stds_arr, means_arr + stds_arr,
                            alpha=0.18, color=COLORS[variant])
            summary[label.split()[0]].setdefault("pr_top32", {})[variant] = {
                "steps": steps,
                "mean": [float(x) for x in means_arr],
                "std":  [float(x) for x in stds_arr],
                "final_mean": float(means_arr[-1]),
                "final_std":  float(stds_arr[-1]),
            }
        ax.set_xlabel("step")
        ax.set_title(f"PR (top-32) over training — {label}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Participation Ratio")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "tier0_pr_vs_step.png", dpi=140)
    plt.close()
    print(f"[tier0] saved -> {OUT_DIR/'tier0_pr_vs_step.png'}")

    # --- 2. Predictor Jacobian spectral gap at final step + trajectory ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    gap_summary = {}
    for ax, label, data in [(axes[0], "v2 (40K joint)", v2),
                            (axes[1], "v3 (8K frozen)", v3)]:
        gap_summary[label.split()[0]] = {}
        for variant, runs in data.items():
            jac_per_seed = [r["jacobian"] for r in runs]
            steps, means, stds = agg_by_step(jac_per_seed, "gap_1_to_10")
            if not steps:
                continue
            means_arr = np.array(means); stds_arr = np.array(stds)
            ax.plot(steps, means_arr, label=variant, color=COLORS[variant], lw=1.8)
            ax.fill_between(steps, means_arr - stds_arr, means_arr + stds_arr,
                            alpha=0.18, color=COLORS[variant])
            _, g2,  g2_std  = agg_by_step(jac_per_seed, "gap_1_to_2")
            _, g10, g10_std = agg_by_step(jac_per_seed, "gap_1_to_10")
            _, g20, g20_std = agg_by_step(jac_per_seed, "gap_1_to_20")
            gap_summary[label.split()[0]][variant] = {
                "final_step": steps[-1],
                "gap_1_to_2_final":  (float(g2[-1]),  float(g2_std[-1])),
                "gap_1_to_10_final": (float(g10[-1]), float(g10_std[-1])),
                "gap_1_to_20_final": (float(g20[-1]), float(g20_std[-1])),
            }
        ax.set_xlabel("step")
        ax.set_ylabel("σ₁ / σ₁₀")
        ax.set_title(f"Predictor Jacobian spectral gap σ₁/σ₁₀ — {label}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "tier0_spectral_gap.png", dpi=140)
    plt.close()
    summary["v2"]["jacobian_spectral_gap_final"] = gap_summary["v2"]
    summary["v3"]["jacobian_spectral_gap_final"] = gap_summary["v3"]
    print(f"[tier0] saved -> {OUT_DIR/'tier0_spectral_gap.png'}")

    # --- 3. SV decay curve (final step) + power-law exponent β ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    decay_summary = {}
    for ax, label, data in [(axes[0], "v2 (40K joint, final)", v2),
                            (axes[1], "v3 (8K frozen, final)", v3)]:
        decay_summary[label.split()[0]] = {}
        for variant, runs in data.items():
            # take final step from each seed, average across seeds
            final_svs_per_seed = []
            for r in runs:
                if r["jacobian"]:
                    final_svs_per_seed.append(np.array(r["jacobian"][-1]["mean_top_svs"]))
            if not final_svs_per_seed:
                continue
            stack = np.stack(final_svs_per_seed)  # (n_seeds, 32)
            mean_svs = stack.mean(axis=0)
            std_svs  = stack.std(axis=0)
            i = np.arange(1, len(mean_svs) + 1)
            ax.plot(i, mean_svs, "o-", label=variant, color=COLORS[variant], lw=1.5, ms=4)
            ax.fill_between(i, mean_svs - std_svs, mean_svs + std_svs,
                            alpha=0.18, color=COLORS[variant])
            betas = [power_law_exponent(s) for s in stack]
            decay_summary[label.split()[0]][variant] = {
                "mean_svs": mean_svs.tolist(),
                "std_svs":  std_svs.tolist(),
                "power_law_beta_mean": float(np.mean(betas)),
                "power_law_beta_std":  float(np.std(betas)),
                "n_seeds": len(betas),
            }
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("singular-value index i")
        ax.set_ylabel(r"$\sigma_i$")
        ax.set_title(f"SV decay (log-log, final step) — {label}")
        ax.grid(which="both", alpha=0.3)
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "tier0_sv_decay_final.png", dpi=140)
    plt.close()
    summary["v2"]["sv_decay_final"] = decay_summary["v2"]
    summary["v3"]["sv_decay_final"] = decay_summary["v3"]
    print(f"[tier0] saved -> {OUT_DIR/'tier0_sv_decay_final.png'}")

    # --- write JSON ---
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"[tier0] saved -> {OUT_JSON}")

    # --- print headline numbers ---
    print("\n" + "=" * 72)
    print("HEADLINE NUMBERS")
    print("=" * 72)
    for cond in ["v2", "v3"]:
        print(f"\n--- {cond.upper()} ---")
        pr = summary[cond]["pr_top32"]
        gap = summary[cond]["jacobian_spectral_gap_final"]
        dec = summary[cond]["sv_decay_final"]
        for variant in ["baseline", "uvlowr_r4", "randdiff"]:
            if variant not in pr:
                continue
            g = gap[variant]
            d = dec[variant]
            print(f"  {variant:<12}: "
                  f"PR_final={pr[variant]['final_mean']:6.2f} ± {pr[variant]['final_std']:.2f}   "
                  f"σ₁/σ₂={g['gap_1_to_2_final'][0]:.3f} ± {g['gap_1_to_2_final'][1]:.3f}   "
                  f"σ₁/σ₁₀={g['gap_1_to_10_final'][0]:.3f} ± {g['gap_1_to_10_final'][1]:.3f}   "
                  f"β={d['power_law_beta_mean']:.3f} ± {d['power_law_beta_std']:.3f}")


if __name__ == "__main__":
    main()
