"""Tier-0 cross-metric correlation matrix.

For each (variant, seed, condition) data point, take the final-step value of:
  PR        — Participation Ratio of latent eigenvalues (from latent_cov.jsonl)
  σ₁        — Predictor Jacobian top singular value (from jacobian_probe.jsonl)
  F²        — Predictor Jacobian Frobenius² (= Σσᵢ²)
  nuclear   — Predictor Jacobian nuclear norm ≈ Σ_{i=1..32} σᵢ (top-32 truncation)
  SR        — Stable rank = F² / σ₁²
  gap       — Spectral gap σ₁ / σ₁₀
  β         — Power-law decay exponent of {σᵢ} via lstsq on log(σ) ~ a − β log(i)

Pearson correlation across n = 15 data points (v2: 3×3 seeds; v3: 3×2 seeds).
Also reports v2-only and v3-only correlations.

Output:
  refine-logs/tier0_correlation_matrix.json
  refine-logs/tier0_correlation_matrix.png
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V2_ROOT = Path("/workspace/lewm_autodl_results_v2")
V3_ROOT = Path("/workspace/lewm_autodl_results_v3")
OUT_DIR = Path("/workspace/le-wm/refine-logs")

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
METRICS = ["PR", "σ₁", "F²", "nuclear", "SR", "gap", "β"]


def participation_ratio(eig):
    e = np.asarray(eig, dtype=np.float64)
    e = e[e > 0]
    return float((e.sum() ** 2) / (e ** 2).sum()) if e.size else 0.0


def power_law_beta(svs):
    svs = np.asarray(svs, dtype=np.float64)
    mask = svs > 0
    if mask.sum() < 4: return float("nan")
    i = np.arange(1, len(svs) + 1)[mask]
    s = svs[mask]
    A = np.vstack([np.log(i), np.ones_like(i, dtype=float)]).T
    slope, _ = np.linalg.lstsq(A, np.log(s), rcond=None)[0]
    return float(-slope)


def final_record(jsonl_path):
    last = None
    for line in open(jsonl_path):
        last = json.loads(line)
    return last


def per_run_metrics(run_dir):
    lc = final_record(run_dir / "latent_cov.jsonl")
    jc = final_record(run_dir / "jacobian_probe.jsonl")
    if lc is None or jc is None:
        return None
    # PR from top-32 latent eigvals
    pr = participation_ratio(lc["top_eigvals"])
    # Predictor-Jacobian: average top_svs across samples per step
    sv_per_sample = np.array([s["top_svs"] for s in jc["samples"]])    # (n_samples, 32)
    mean_svs = sv_per_sample.mean(axis=0)
    fb_per_sample = np.array([s["frobenius_norm_sq"] for s in jc["samples"]])
    sigma_1 = float(mean_svs[0])
    sigma_10 = float(mean_svs[9])
    F2 = float(fb_per_sample.mean())
    nuclear = float(mean_svs.sum())          # truncated to top-32 (typically captures >95%)
    SR = F2 / max(sigma_1 ** 2, 1e-12)
    gap = sigma_1 / max(sigma_10, 1e-12)
    beta = power_law_beta(mean_svs)
    return {
        "PR": pr, "σ₁": sigma_1, "F²": F2,
        "nuclear": nuclear, "SR": SR, "gap": gap, "β": beta,
    }


def collect(root, runs):
    data = []
    for variant, run_names in runs.items():
        for rn in run_names:
            rd = root / rn
            m = per_run_metrics(rd)
            if m is None:
                print(f"[corr] missing files in {rd}; skipping")
                continue
            d = {"variant": variant, "run": rn, "condition": root.name}
            d.update(m)
            data.append(d)
    return data


def corr_matrix(rows, metrics=METRICS):
    M = np.array([[r[m] for m in metrics] for r in rows])
    # Pearson
    c = np.corrcoef(M.T)
    return c, M


def plot_grid(corrs, labels, out):
    """corrs: list of (title, C, n).  3 panels: v2-only, v3-only, pooled."""
    n_panels = len(corrs)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.4))
    if n_panels == 1: axes = [axes]
    for ax, (title, C, n) in zip(axes, corrs):
        im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{C[i,j]:+.2f}", ha="center", va="center", fontsize=9,
                        color="black" if abs(C[i,j]) < 0.6 else "white")
        ax.set_title(f"{title} (n={n})")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.04)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()


def main():
    v2 = collect(V2_ROOT, V2_RUNS)
    v3 = collect(V3_ROOT, V3_RUNS)
    pooled = v2 + v3
    print(f"[corr] v2 n={len(v2)}  v3 n={len(v3)}  pooled n={len(pooled)}")
    # show raw values
    print("\n--- per-run final-step metric table ---")
    print("condition variant   run                       PR     σ₁     F²    nuclear   SR     gap   β")
    for r in pooled:
        print(f"  {r['condition']:<22} {r['variant']:<9} {r['run']:<25} "
              f"{r['PR']:6.2f} {r['σ₁']:6.3f} {r['F²']:7.2f} {r['nuclear']:7.2f} "
              f"{r['SR']:6.2f} {r['gap']:5.3f} {r['β']:5.3f}")

    C_v2, _ = corr_matrix(v2)
    C_v3, _ = corr_matrix(v3)
    C_pool, _ = corr_matrix(pooled)

    plot_grid(
        [("v2 (joint 40K)", C_v2, len(v2)),
         ("v3 (frozen 8K)", C_v3, len(v3)),
         ("pooled (v2+v3)", C_pool, len(pooled))],
        METRICS,
        OUT_DIR / "tier0_correlation_matrix.png",
    )
    print(f"[corr] saved -> {OUT_DIR / 'tier0_correlation_matrix.png'}")

    json_out = {
        "metrics_order": METRICS,
        "v2_only":  {"n": len(v2),  "data": v2,  "corr": C_v2.tolist()},
        "v3_only":  {"n": len(v3),  "data": v3,  "corr": C_v3.tolist()},
        "pooled":   {"n": len(pooled), "data": pooled, "corr": C_pool.tolist()},
    }
    (OUT_DIR / "tier0_correlation_matrix.json").write_text(json.dumps(json_out, indent=2))
    print(f"[corr] saved -> {OUT_DIR / 'tier0_correlation_matrix.json'}")

    print("\n--- strong correlations (|r| > 0.7) in pooled (n={}) ---".format(len(pooled)))
    for i, m1 in enumerate(METRICS):
        for j, m2 in enumerate(METRICS):
            if j <= i: continue
            r = C_pool[i, j]
            if abs(r) > 0.7:
                print(f"  {m1:<8} vs {m2:<8}  r = {r:+.3f}")

    print("\n--- v2 vs v3 correlation differences ---")
    for i, m1 in enumerate(METRICS):
        for j, m2 in enumerate(METRICS):
            if j <= i: continue
            r2, r3 = C_v2[i, j], C_v3[i, j]
            if abs(r2 - r3) > 0.5:
                print(f"  {m1:<8} vs {m2:<8}  v2={r2:+.3f}  v3={r3:+.3f}  Δ={r3-r2:+.3f}")


if __name__ == "__main__":
    main()
