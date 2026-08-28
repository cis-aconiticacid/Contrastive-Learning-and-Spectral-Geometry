"""Tier-1: compute all 6 metrics on dumped trajectories.

Inputs:  /workspace/le-wm/refine-logs/tier1_traj/{variant}_traj.npz
         keys: predicted_z (N, H+1, D), encoded_z (N, H+1, D),
               state_gt (N, H+1, 7), actions (N, H+1, 2)

Outputs:
  refine-logs/tier1_traj/predicted_metrics.json
  refine-logs/tier1_traj/predicted_metrics_summary.csv
  refine-logs/tier1_decay_curves.png
  refine-logs/tier1_dmd_spectrum.png
  refine-logs/tier1_velocity_distributions.png
  refine-logs/tier1_correlation_matrix.png
  refine-logs/tier1_findings.md
"""
import json
from pathlib import Path
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

TRAJ_DIR = Path("/workspace/le-wm/refine-logs/tier1_traj")
OUT_DIR  = Path("/workspace/le-wm/refine-logs")
EPS = 1e-8

VARIANTS = ["tier1_baseline", "tier1_uvlowr", "tier1_randdiff"]
SHORT = {"tier1_baseline": "baseline", "tier1_uvlowr": "uvlowr_r4", "tier1_randdiff": "randdiff"}
COLORS = {"baseline": "C0", "uvlowr_r4": "C1", "randdiff": "C2", "encoded_GT": "k"}

HORIZONS_FOR_PROBE = [0, 1, 3, 5, 10, 15, 20]


# -------------------------------------------------------------------- #
#  Per-trajectory metrics
# -------------------------------------------------------------------- #
def velocities(z):
    """z: (N, T, D) -> v: (N, T-1, D)"""
    return z[:, 1:] - z[:, :-1]


def tlps(z):
    """Temporal latent-path straightness per trajectory.
    mean cosine similarity between consecutive velocities."""
    v = velocities(z)                                            # (N, T-1, D)
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True) + EPS
    u = v / v_norm
    cos = (u[:, :-1] * u[:, 1:]).sum(-1)                         # (N, T-2)
    return cos.mean(axis=1), cos                                 # per-traj mean, per-(traj,t)


def curvature(z):
    """Discrete curvature κ_t = ||v_{t+1} - v_t|| / (||v_t||^2 + ε)."""
    v = velocities(z)                                            # (N, T-1, D)
    dv = v[:, 1:] - v[:, :-1]                                    # (N, T-2, D)
    num = np.linalg.norm(dv, axis=-1)
    den = (v[:, :-1] ** 2).sum(-1) + EPS
    k = num / den                                                # (N, T-2)
    return k.mean(axis=1), k


def velocity_acceleration(z):
    v = velocities(z)
    a = v[:, 1:] - v[:, :-1]
    v_norm = np.linalg.norm(v, axis=-1)
    a_norm = np.linalg.norm(a, axis=-1)
    return v_norm.flatten(), a_norm.flatten()


def dmd_eigenvalues(z, rank_cap=32):
    """Companion-DMD on pooled snapshots from all trajectories.
    z: (N, T, D).  Returns eigenvalues of reduced A_tilde."""
    N, T, D = z.shape
    # build X, Y by pooling all trajectories
    X = z[:, :-1].reshape(-1, D).T                               # (D, N*(T-1))
    Y = z[:, 1:].reshape(-1, D).T                                # (D, N*(T-1))
    # truncated SVD of X
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    r = min(rank_cap, S.shape[0])
    U_r  = U[:, :r]
    S_r  = S[:r]
    Vt_r = Vt[:r, :]
    A_tilde = U_r.T @ Y @ Vt_r.T @ np.diag(1.0 / (S_r + EPS))     # (r, r)
    eigs = np.linalg.eigvals(A_tilde)
    return eigs, A_tilde


def two_nn_id(points, n_max=4000, rng=None):
    """TwoNN intrinsic dimensionality estimator (Facco et al. 2017)."""
    rng = rng or np.random.default_rng(0)
    P = points.shape[0]
    if P > n_max:
        idx = rng.choice(P, size=n_max, replace=False)
        pts = points[idx]
    else:
        pts = points
    # pairwise distances
    d = np.sqrt(np.maximum(0.0,
        (pts ** 2).sum(-1, keepdims=True) + (pts ** 2).sum(-1)[None, :]
        - 2 * pts @ pts.T))
    np.fill_diagonal(d, np.inf)
    d_sorted = np.sort(d, axis=1)
    r1, r2 = d_sorted[:, 0], d_sorted[:, 1]
    mu = (r2 + EPS) / (r1 + EPS)
    mu = np.sort(mu[mu > 1])
    if mu.size < 10:
        return float("nan"), float("nan")
    # F(mu) = 1 - mu^{-d}  =>  log(1 - F̂) = -d * log(mu)
    Fhat = np.arange(1, len(mu) + 1) / (len(mu) + 1)
    y = -np.log(1 - Fhat)
    x = np.log(mu)
    slope, intercept, r, _, _ = stats.linregress(x, y)
    return float(slope), float(r ** 2)


def probe_R2_at_horizon(latent_pred, state_gt, h, train_frac=0.8, alpha=1.0):
    """Ridge probe latent_pred[:, h] -> state_gt[:, h] (7-d state)."""
    z = latent_pred[:, h]                                        # (N, D)
    s = state_gt[:, h]                                           # (N, 7)
    N = z.shape[0]
    rng = np.random.default_rng(42 + h)
    perm = rng.permutation(N)
    n_tr = int(N * train_frac)
    tr, te = perm[:n_tr], perm[n_tr:]
    Z_tr, Z_te = z[tr], z[te]
    S_tr, S_te = s[tr], s[te]
    # closed-form ridge: W = (Z^T Z + αI)^-1 Z^T S
    ZtZ = Z_tr.T @ Z_tr + alpha * np.eye(Z_tr.shape[1])
    W = np.linalg.solve(ZtZ, Z_tr.T @ S_tr)
    pred = Z_te @ W
    ss_res = ((S_te - pred) ** 2).sum()
    ss_tot = ((S_te - S_te.mean(0)) ** 2).sum() + EPS
    return float(1.0 - ss_res / ss_tot)


# -------------------------------------------------------------------- #
#  Aggregation
# -------------------------------------------------------------------- #
def compute_metrics_for_traj(z, state_gt, label):
    """Compute all metrics on a (N, T, D) trajectory tensor."""
    tlps_per, _   = tlps(z)
    curv_per, _   = curvature(z)
    v_norm, a_norm = velocity_acceleration(z)
    eigs, _ = dmd_eigenvalues(z)
    mag = np.abs(eigs)
    pts = z.reshape(-1, z.shape[-1])
    d_id, fit_r2 = two_nn_id(pts)
    R2_h = {h: probe_R2_at_horizon(z, state_gt, h) for h in HORIZONS_FOR_PROBE}
    return {
        "label": label,
        "n_traj": int(z.shape[0]),
        "horizon": int(z.shape[1] - 1),
        "tlps_mean":  float(tlps_per.mean()),
        "tlps_std":   float(tlps_per.std()),
        "tlps_per_traj": tlps_per.tolist(),
        "curv_mean":  float(curv_per.mean()),
        "curv_std":   float(curv_per.std()),
        "curv_per_traj": curv_per.tolist(),
        "v_mean":     float(v_norm.mean()),
        "v_std":      float(v_norm.std()),
        "v_p95":      float(np.percentile(v_norm, 95)),
        "a_mean":     float(a_norm.mean()),
        "a_std":      float(a_norm.std()),
        "a_p95":      float(np.percentile(a_norm, 95)),
        "av_ratio":   float(a_norm.mean() / (v_norm.mean() + EPS)),
        "dmd_lambda_max":  float(mag.max()),
        "dmd_lambda_top5": sorted(mag.tolist(), reverse=True)[:5],
        "n_unstable":      int((mag > 1.0).sum()),
        "n_marginal":      int(((mag >= 0.99) & (mag <= 1.01)).sum()),
        "spectral_abscissa": float(np.log(mag.max() + EPS)),
        "dmd_eigs_real":   [float(e.real) for e in eigs],
        "dmd_eigs_imag":   [float(e.imag) for e in eigs],
        "twonn_id":     d_id,
        "twonn_fit_r2": fit_r2,
        "R2_per_h":     R2_h,
        "vel_norms":    v_norm.tolist(),
        "acc_norms":    a_norm.tolist(),
    }


def paired_test(per_traj_a, per_traj_b):
    a = np.asarray(per_traj_a, dtype=float)
    b = np.asarray(per_traj_b, dtype=float)
    if len(a) != len(b) or len(a) < 3:
        return float("nan"), float("nan")
    t, p_t = stats.ttest_rel(a, b)
    try:
        _, p_w = stats.wilcoxon(a, b)
    except ValueError:
        p_w = float("nan")
    return float(p_t), float(p_w)


# -------------------------------------------------------------------- #
#  Plotting
# -------------------------------------------------------------------- #
def plot_decay_curves(results, out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    # R²_h
    ax = axes[0]
    for v_short, m in results.items():
        if "predicted" not in m:
            continue
        R = m["predicted"]["R2_per_h"]
        ax.plot(list(R.keys()), list(R.values()), "o-",
                color=COLORS[v_short], label=v_short, lw=2)
    R_gt = next(iter(results.values()))["encoded_gt"]["R2_per_h"]
    ax.plot(list(R_gt.keys()), list(R_gt.values()), "k--", label="encoded GT", lw=1.5, alpha=0.8)
    ax.set_xlabel("horizon h"); ax.set_ylabel("R² (state probe)")
    ax.set_title("Probing R² vs horizon")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # TLPS — show overall mean as bar
    ax = axes[1]
    labels = []
    means  = []
    stds   = []
    for v_short, m in results.items():
        if "predicted" in m:
            labels.append(v_short)
            means.append(m["predicted"]["tlps_mean"])
            stds.append(m["predicted"]["tlps_std"])
    labels.append("encoded_GT")
    means.append(next(iter(results.values()))["encoded_gt"]["tlps_mean"])
    stds.append(next(iter(results.values()))["encoded_gt"]["tlps_std"])
    colors = [COLORS[v] for v in labels]
    ax.bar(labels, means, yerr=stds, color=colors, alpha=0.85, capsize=4)
    ax.set_ylabel("mean TLPS (cos of adjacent velocities)")
    ax.set_title("Predicted-trajectory TLPS")
    ax.axhline(0, color="grey", lw=0.5)
    ax.grid(axis="y", alpha=0.3)

    # |λ_max|
    ax = axes[2]
    for v_short, m in results.items():
        if "predicted" in m:
            ax.bar(v_short, m["predicted"]["dmd_lambda_max"],
                   color=COLORS[v_short], alpha=0.85)
    ax.bar("encoded_GT",
           next(iter(results.values()))["encoded_gt"]["dmd_lambda_max"],
           color="k", alpha=0.85)
    ax.axhline(1.0, color="red", lw=1, ls="--", label="unit circle")
    ax.set_ylabel("|λ_max|")
    ax.set_title("Dominant DMD eigenvalue")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()


def plot_dmd_spectrum(results, out):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), sharex=True, sharey=True)
    panels = list(results.items())
    panels.append(("encoded_GT", {"predicted": next(iter(results.values()))["encoded_gt"]}))
    for ax, (label, m) in zip(axes, panels):
        d = m["predicted"]
        re = np.array(d["dmd_eigs_real"])
        im = np.array(d["dmd_eigs_imag"])
        ax.scatter(re, im, s=24, alpha=0.7, color=COLORS.get(label, "k"))
        # unit circle
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(th), np.sin(th), "r--", lw=1, alpha=0.8)
        ax.axhline(0, color="grey", lw=0.4)
        ax.axvline(0, color="grey", lw=0.4)
        ax.set_aspect("equal")
        ax.set_title(f"{label}  (|λ_max|={d['dmd_lambda_max']:.3f}, n_unstable={d['n_unstable']})")
        ax.set_xlabel("Re(λ)"); ax.set_ylabel("Im(λ)")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()


def plot_velocity_distributions(results, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    bins_v = np.linspace(0, max(np.percentile(m["predicted"]["vel_norms"], 99)
                                 for m in results.values()) * 1.05, 50)
    bins_a = np.linspace(0, max(np.percentile(m["predicted"]["acc_norms"], 99)
                                 for m in results.values()) * 1.05, 50)
    for v_short, m in results.items():
        if "predicted" not in m: continue
        axes[0].hist(m["predicted"]["vel_norms"], bins=bins_v, alpha=0.5,
                     label=v_short, color=COLORS[v_short], density=True)
        axes[1].hist(m["predicted"]["acc_norms"], bins=bins_a, alpha=0.5,
                     label=v_short, color=COLORS[v_short], density=True)
    gt = next(iter(results.values()))["encoded_gt"]
    axes[0].hist(gt["vel_norms"], bins=bins_v, alpha=0.5,
                 label="encoded_GT", color="k", histtype="step", density=True, lw=2)
    axes[1].hist(gt["acc_norms"], bins=bins_a, alpha=0.5,
                 label="encoded_GT", color="k", histtype="step", density=True, lw=2)
    axes[0].set_xlabel("‖v‖"); axes[0].set_ylabel("density"); axes[0].set_title("Velocity norm")
    axes[1].set_xlabel("‖a‖"); axes[1].set_title("Acceleration norm")
    for ax in axes: ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()


def plot_correlation_matrix(per_traj_table, out):
    """per_traj_table: dict[(variant, metric_name)] = (n_traj,) array."""
    metrics = ["tlps", "curv", "v_mean", "a_mean", "av_ratio"]
    rows = []
    labels = []
    for v, table in per_traj_table.items():
        for m in metrics:
            rows.append(table[m])
            labels.append(f"{v}.{m}")
    X = np.stack(rows)                                              # (M, N)
    # within-variant cross-metric correlation, averaged across variants
    short = ["tlps", "curv", "v", "a", "av_ratio"]
    n_metrics = len(short)
    avg_corr = np.zeros((n_metrics, n_metrics))
    n_var = len(per_traj_table)
    for v, table in per_traj_table.items():
        X_v = np.stack([table[m] for m in metrics])                 # (M, N)
        c = np.corrcoef(X_v)
        avg_corr += c
    avg_corr /= n_var

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(avg_corr, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    ax.set_xticks(range(n_metrics)); ax.set_yticks(range(n_metrics))
    ax.set_xticklabels(short, rotation=30, ha="right")
    ax.set_yticklabels(short)
    for i in range(n_metrics):
        for j in range(n_metrics):
            ax.text(j, i, f"{avg_corr[i,j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black" if abs(avg_corr[i,j]) < 0.6 else "white")
    ax.set_title("Per-trajectory metric × metric correlation\n(average across 3 variants)")
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()


# -------------------------------------------------------------------- #
#  Main
# -------------------------------------------------------------------- #
def main():
    results = OrderedDict()
    per_traj_table = OrderedDict()
    encoded_gt_metrics = None

    for variant in VARIANTS:
        path = TRAJ_DIR / f"{variant}_traj.npz"
        if not path.exists():
            print(f"[metrics] MISSING {path}; skipping {variant}")
            continue
        data = np.load(path, allow_pickle=True)
        pred_z = data["predicted_z"]                                 # (N, H+1, D)
        enc_z  = data["encoded_z"]                                   # (N, H+1, D)
        state  = data["state_gt"]                                    # (N, H+1, 7)
        print(f"[metrics] === {variant}  shapes: pred {pred_z.shape}  enc {enc_z.shape}  state {state.shape}")

        m_pred = compute_metrics_for_traj(pred_z, state, label=f"predicted_{variant}")
        if encoded_gt_metrics is None:
            m_gt = compute_metrics_for_traj(enc_z, state, label="encoded_GT")
            encoded_gt_metrics = m_gt
        v_short = SHORT[variant]
        results[v_short] = {"predicted": m_pred, "encoded_gt": encoded_gt_metrics}

        per_traj_table[v_short] = {
            "tlps":     np.array(m_pred["tlps_per_traj"]),
            "curv":     np.array(m_pred["curv_per_traj"]),
            "v_mean":   np.array([np.mean(np.linalg.norm(velocities(pred_z[i:i+1]), axis=-1))
                                  for i in range(pred_z.shape[0])]),
            "a_mean":   np.array([np.mean(np.linalg.norm(velocities(velocities(pred_z[i:i+1])), axis=-1))
                                  for i in range(pred_z.shape[0])]),
            "av_ratio": None,
        }
        per_traj_table[v_short]["av_ratio"] = (
            per_traj_table[v_short]["a_mean"] / (per_traj_table[v_short]["v_mean"] + EPS)
        )

    if not results:
        print("[metrics] No variants found — nothing to compute. Did dump finish?")
        return

    # --- paired tests vs baseline ---
    paired = {}
    if "baseline" in per_traj_table:
        for v_short, table in per_traj_table.items():
            if v_short == "baseline":
                continue
            paired[v_short] = {}
            for metric in ["tlps", "curv", "v_mean", "a_mean"]:
                p_t, p_w = paired_test(per_traj_table["baseline"][metric], table[metric])
                paired[v_short][metric] = {"p_ttest": p_t, "p_wilcoxon": p_w}

    # --- summary table ---
    summary_rows = []
    header = "variant,tlps_mean,tlps_std,curv_mean,curv_std,|λ_max|,n_unstable,n_marginal,twonn_id,v_mean,a_mean,av_ratio,R2_h0,R2_h5,R2_h10,R2_h20".split(",")
    summary_rows.append(",".join(header))
    for v_short, m in results.items():
        p = m["predicted"]
        row = [
            v_short,
            f"{p['tlps_mean']:.4f}", f"{p['tlps_std']:.4f}",
            f"{p['curv_mean']:.5f}", f"{p['curv_std']:.5f}",
            f"{p['dmd_lambda_max']:.4f}",
            f"{p['n_unstable']}", f"{p['n_marginal']}",
            f"{p['twonn_id']:.3f}",
            f"{p['v_mean']:.4f}",
            f"{p['a_mean']:.4f}",
            f"{p['av_ratio']:.4f}",
            f"{p['R2_per_h'][0]:.4f}", f"{p['R2_per_h'][5]:.4f}",
            f"{p['R2_per_h'][10]:.4f}", f"{p['R2_per_h'][20]:.4f}",
        ]
        summary_rows.append(",".join(row))
    # encoded GT row
    p = encoded_gt_metrics
    summary_rows.append(",".join([
        "encoded_GT",
        f"{p['tlps_mean']:.4f}", f"{p['tlps_std']:.4f}",
        f"{p['curv_mean']:.5f}", f"{p['curv_std']:.5f}",
        f"{p['dmd_lambda_max']:.4f}",
        f"{p['n_unstable']}", f"{p['n_marginal']}",
        f"{p['twonn_id']:.3f}",
        f"{p['v_mean']:.4f}",
        f"{p['a_mean']:.4f}",
        f"{p['av_ratio']:.4f}",
        f"{p['R2_per_h'][0]:.4f}", f"{p['R2_per_h'][5]:.4f}",
        f"{p['R2_per_h'][10]:.4f}", f"{p['R2_per_h'][20]:.4f}",
    ]))
    csv_path = TRAJ_DIR / "predicted_metrics_summary.csv"
    csv_path.write_text("\n".join(summary_rows) + "\n")
    print(f"[metrics] saved -> {csv_path}")

    # --- JSON dump ---
    json_path = TRAJ_DIR / "predicted_metrics.json"
    json_dump = {
        "variants": {v: {"predicted": results[v]["predicted"]} for v in results},
        "encoded_gt": encoded_gt_metrics,
        "paired_tests_vs_baseline": paired,
    }
    # strip very large arrays from JSON (per-traj kept; per-(traj,t) was inside helper outputs)
    for v in json_dump["variants"]:
        d = json_dump["variants"][v]["predicted"]
        d.pop("vel_norms", None); d.pop("acc_norms", None)
    json_dump["encoded_gt"].pop("vel_norms", None)
    json_dump["encoded_gt"].pop("acc_norms", None)
    json_path.write_text(json.dumps(json_dump, indent=2))
    print(f"[metrics] saved -> {json_path}")

    # --- figures ---
    plot_decay_curves(results, OUT_DIR / "tier1_decay_curves.png")
    plot_dmd_spectrum(results, OUT_DIR / "tier1_dmd_spectrum.png")
    # re-load vel/acc for histograms (we stripped them above)
    full = OrderedDict()
    for v_short in results:
        # recompute lightweight
        variant_long = [k for k, vs in SHORT.items() if vs == v_short][0]
        data = np.load(TRAJ_DIR / f"{variant_long}_traj.npz", allow_pickle=True)
        v_norm, a_norm = velocity_acceleration(data["predicted_z"])
        full[v_short] = {"predicted": {"vel_norms": v_norm, "acc_norms": a_norm,
                                       "tlps_mean": results[v_short]["predicted"]["tlps_mean"],
                                       "dmd_lambda_max": results[v_short]["predicted"]["dmd_lambda_max"],
                                       "n_unstable": results[v_short]["predicted"]["n_unstable"],
                                       "dmd_eigs_real": results[v_short]["predicted"]["dmd_eigs_real"],
                                       "dmd_eigs_imag": results[v_short]["predicted"]["dmd_eigs_imag"]}}
    # encoded GT vel/acc:
    data = np.load(TRAJ_DIR / f"{VARIANTS[0]}_traj.npz", allow_pickle=True)
    v_norm, a_norm = velocity_acceleration(data["encoded_z"])
    for v_short in full:
        full[v_short]["encoded_gt"] = {"vel_norms": v_norm, "acc_norms": a_norm,
                                       "tlps_mean": encoded_gt_metrics["tlps_mean"],
                                       "dmd_lambda_max": encoded_gt_metrics["dmd_lambda_max"],
                                       "n_unstable": encoded_gt_metrics["n_unstable"],
                                       "dmd_eigs_real": encoded_gt_metrics["dmd_eigs_real"],
                                       "dmd_eigs_imag": encoded_gt_metrics["dmd_eigs_imag"]}
    plot_velocity_distributions(full, OUT_DIR / "tier1_velocity_distributions.png")
    plot_correlation_matrix(per_traj_table, OUT_DIR / "tier1_correlation_matrix.png")

    # --- print summary ---
    print("\n" + "=" * 96)
    print("TIER-1 PREDICTED-TRAJECTORY HEADLINE NUMBERS")
    print("=" * 96)
    print("\n".join(summary_rows))

    # --- decision gates ---
    print("\n" + "=" * 96)
    print("DECISION GATES (G1-G5)")
    print("=" * 96)
    if "baseline" in results and "randdiff" in results:
        rdb = results["baseline"]["predicted"]
        rdr = results["randdiff"]["predicted"]
        g1_diff = rdb["tlps_mean"] - rdr["tlps_mean"]
        p_g1 = paired["randdiff"]["tlps"]["p_ttest"] if "randdiff" in paired else float("nan")
        print(f"G1  randdiff TLPS < baseline by ≥0.05?  diff={g1_diff:+.4f}  p={p_g1:.4f}  "
              f"=> {'PASS' if g1_diff >= 0.05 and p_g1 < 0.05 else 'fail'}")
        g2_diff = rdr["n_unstable"] - rdb["n_unstable"]
        print(f"G2  randdiff n_unstable - baseline ≥ 2?  diff={g2_diff:+d}  "
              f"=> {'PASS' if g2_diff >= 2 else 'fail'}")
        r0b, r5b, r10b = rdb["R2_per_h"][0], rdb["R2_per_h"][5], rdb["R2_per_h"][10]
        r0r, r5r, r10r = rdr["R2_per_h"][0], rdr["R2_per_h"][5], rdr["R2_per_h"][10]
        g3_h0_tie = abs(r0r - r0b) < 0.05
        g3_h5_drop = (r5b - r5r) > 0.05
        print(f"G3  R²_h0 tied AND R²_h5 randdiff↓?  h0={r0b:.3f}/{r0r:.3f}  "
              f"h5={r5b:.3f}/{r5r:.3f}  => {'PASS' if (g3_h0_tie and g3_h5_drop) else 'fail'}")
    if "baseline" in results and "uvlowr_r4" in results and "randdiff" in results:
        ids = [results[v]["predicted"]["twonn_id"] for v in ["baseline", "uvlowr_r4", "randdiff"]]
        spread = max(ids) - min(ids)
        print(f"G4  TwoNN ID spread ≥ 1.5?  IDs={ids}  spread={spread:.3f}  "
              f"=> {'PASS' if spread >= 1.5 else 'fail'}")
    # G5 — between TLPS, |λ_max|, R²-decay-slope: pull from results
    if results:
        per_var = []
        for v_short, m in results.items():
            p = m["predicted"]
            R = p["R2_per_h"]
            # decay slope: simple linear fit log R² vs h on h ∈ [0, 20]
            hs = np.array(list(R.keys()), float)
            r2 = np.array(list(R.values()), float)
            r2 = np.clip(r2, 1e-3, 1.0)
            slope = np.polyfit(hs, r2, 1)[0]
            per_var.append([p["tlps_mean"], p["dmd_lambda_max"], slope])
        per_var = np.array(per_var)
        if per_var.shape[0] >= 3:
            c = np.corrcoef(per_var.T)
            pair_tlps_lam = c[0, 1]
            pair_tlps_slope = c[0, 2]
            pair_lam_slope = c[1, 2]
            print(f"G5  cross-metric |r| of (TLPS, |λ_max|, R²-slope) — "
                  f"r(TLPS,|λ_max|)={pair_tlps_lam:.2f}  "
                  f"r(TLPS,slope)={pair_tlps_slope:.2f}  "
                  f"r(|λ_max|,slope)={pair_lam_slope:.2f}  "
                  f"=> "
                  f"{'PASS (consolidate to 1)' if all(abs(x) > 0.7 for x in [pair_tlps_lam, pair_tlps_slope, pair_lam_slope]) else 'fail (keep distinct)'}")


if __name__ == "__main__":
    main()
