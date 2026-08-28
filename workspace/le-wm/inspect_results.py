"""Quick inspection of the 40K-step run results.

Reads aggregated/{baseline,uvlowr_r4,randdiff}/* and produces a structured
summary table comparing variants on:
  - final-step training loss / sigreg / grad_norm (mean ± std)
  - probing R² overall + per-dim time series (initial / mid / final + per-dim final)
  - Jacobian spectrum trajectory
  - Latent covariance trajectory
  - rollout MSE per horizon
  - per-weight SVD on final ckpt (which Linears actually have low rank)
  - encoder Jacobian spectrum on final ckpt
  - MLP probe vs Ridge probe gap

Usage: python inspect_results.py --base /workspace/lewm_autodl_results
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import statistics as stats


def fmt(m, s, prec=4):
    if m is None: return "n/a"
    return f"{m:.{prec}f}±{s:.{prec}f}" if s else f"{m:.{prec}f}"


def load_aggregate(path):
    return json.loads(path.read_text()) if path.exists() else {}


def load_curve(path):
    if not path.exists(): return []
    return list(csv.DictReader(path.open()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/workspace/lewm_autodl_results")
    args = ap.parse_args()
    base = Path(args.base)
    variants = ["baseline", "uvlowr_r4", "randdiff"]

    print(f"\n{'='*100}\nLeWM 40K-step run — inspection\n{'='*100}\n")

    # --- Engineering ---
    print("--- Engineering (mean over 3 seeds) ---")
    print(f"{'variant':>14s} | {'params':>10s} | {'peak_mb':>9s} | {'wall_s':>9s}")
    for v in variants:
        agg = load_aggregate(base / f'aggregated/{v}/aggregate.json')
        e = agg.get('engineering', {})
        params = e.get('predictor_params_mean')
        peak = e.get('peak_gpu_memory_mb_mean')
        wall = e.get('wall_clock_s_mean')
        if params is not None:
            print(f"{v:>14s} | {int(params):>10d} | {peak:>9.1f} | {wall:>9.1f}")
    print()

    # --- Final training loss + grad norm ---
    print("--- Final-step training (mean ± std over 3 seeds) ---")
    print(f"{'variant':>14s} | {'pred_loss':>16s} | {'sigreg':>16s} | {'grad_norm_pre_clip':>22s}")
    for v in variants:
        rows = load_curve(base / f'aggregated/{v}/training_curve.csv')
        if not rows: continue
        last = [r for r in rows if r.get('fit/pred_loss_mean')][-1]
        gn = last.get('fit/grad_norm_pre_clip_mean')
        gns = last.get('fit/grad_norm_pre_clip_std', 0)
        print(f"{v:>14s} | "
              f"{fmt(float(last['fit/pred_loss_mean']), float(last.get('fit/pred_loss_std',0)), 5):>16s} | "
              f"{(fmt(float(last['fit/sigreg_loss_mean']), float(last.get('fit/sigreg_loss_std',0)), 4) if last.get('fit/sigreg_loss_mean') else 'n/a'):>16s} | "
              f"{fmt(float(gn), float(gns), 4) if gn else 'n/a':>22s}")
    print()

    # --- Probing R² time series (steps 0, mid, final) ---
    print("--- Probing R² time series (mean over seeds) ---")
    for v in variants:
        rows = load_curve(base / f'aggregated/{v}/probing_curve.csv')
        if not rows: continue
        steps_data = [(int(r['step']),
                       float(r.get('overall_r2_mean', 0)),
                       float(r.get('overall_r2_std', 0))) for r in rows]
        print(f"  {v}: steps " + " | ".join(
            f"{s}={m:.3f}±{sd:.3f}" for s, m, sd in steps_data[:3]
        ) + " ... " + " | ".join(
            f"{s}={m:.3f}±{sd:.3f}" for s, m, sd in steps_data[-3:]
        ))

    # --- Per-dim probe at final step ---
    print("\n--- Per-dim probe R² at final step (mean over seeds) ---")
    print("Dim labels: [agent_x, agent_y, block_x, block_y, block_angle, agent_vx, agent_vy]")
    for v in variants:
        rows = load_curve(base / f'aggregated/{v}/probing_curve.csv')
        if not rows: continue
        last = rows[-1]
        per_dim = []
        for d in range(7):
            k = f'r2_dim{d}_mean'
            if k in last:
                per_dim.append(float(last[k]))
        if per_dim:
            print(f"  {v:>12s}: {[round(x,2) for x in per_dim]}")
    print()

    # --- Jacobian time series ---
    print("--- Predictor Jacobian: stable_rank trajectory (mean over seeds) ---")
    for v in variants:
        rows = load_curve(base / f'aggregated/{v}/jacobian_curve.csv')
        if not rows: continue
        steps = [(int(r['step']), float(r['stable_rank_mean'])) for r in rows]
        print(f"  {v:>12s}: " + " ".join(f"{s}:{sr:.0f}" for s, sr in steps[:1] + steps[len(steps)//2:len(steps)//2+1] + steps[-1:]))
    print()

    # --- Latent cov time series ---
    print("--- Encoder latent covariance: stable_rank trajectory ---")
    for v in variants:
        rows = load_curve(base / f'aggregated/{v}/latent_cov_curve.csv')
        if not rows: continue
        steps = [(int(r['step']), float(r['stable_rank_mean']), float(r['effective_rank_mean']))
                 for r in rows]
        first, mid, last = steps[0], steps[len(steps)//2], steps[-1]
        print(f"  {v:>12s}: step0 sr={first[1]:.1f} er={first[2]:.1f}  "
              f"mid sr={mid[1]:.1f} er={mid[2]:.1f}  "
              f"final sr={last[1]:.1f} er={last[2]:.1f}")
    print()

    # --- Rollout MSE ---
    print("--- Multi-step rollout MSE (mean ± std over 3 seeds) ---")
    headers = ["horizon"] + variants
    print(f"{'horizon':>8s}  " + "  ".join(f"{v:>20s}" for v in variants))
    horizons = set()
    for v in variants:
        for r in load_curve(base / f'aggregated/{v}/rollout_summary.csv'):
            horizons.add(int(r['horizon']))
    for h in sorted(horizons):
        line = f"h={h:>5d}  "
        for v in variants:
            rows = load_curve(base / f'aggregated/{v}/rollout_summary.csv')
            r = next((r for r in rows if int(r['horizon']) == h), None)
            if r and 'mse_mean' in r:
                line += f"  {fmt(float(r['mse_mean']), float(r.get('mse_std',0)), 5):>18s}"
            else:
                line += f"  {'n/a':>18s}"
        print(line)
    print()

    # --- Per-weight SVD (predictor) at final ckpt, seed 0 ---
    print("--- Per-weight SVD on final ckpt (seed 0 only, picking key layers) ---")
    for v in variants:
        fa_path = base / f'{v}_seed0/final_analyses.json'
        if not fa_path.exists(): continue
        fa = json.loads(fa_path.read_text())
        ws = fa.get('weight_svd', {})
        # Pick representative layers
        keys = ['transformer.layers.0.attn.to_qkv',
                'transformer.layers.0.attn.to_out.0',
                'transformer.layers.0.mlp.net.1' if v == 'baseline' or v == 'randdiff' else 'transformer.layers.0.mlp.w1_a',
                'transformer.layers.0.adaLN_modulation.1']
        line = f"  {v:>12s}: "
        for k in keys:
            if k in ws:
                s = ws[k]
                line += f"{k.split('.')[-1]}={s['stable_rank']:.0f} "
        print(line)
    print()

    # --- Encoder Jacobian at final ckpt ---
    print("--- Encoder Jacobian (input pixels → 192-d latent) on final ckpt, seed 0 ---")
    for v in variants:
        fa_path = base / f'{v}_seed0/final_analyses.json'
        if not fa_path.exists(): continue
        fa = json.loads(fa_path.read_text())
        ej = fa.get('encoder_jacobian', {})
        if 'mean_stable_rank' in ej:
            print(f"  {v:>12s}: spectral={ej['mean_spectral_norm']:.3f}  "
                  f"sr={ej['mean_stable_rank']:.2f}  "
                  f"er={ej['mean_effective_rank']:.2f}")
    print()

    # --- MLP probe vs Ridge ---
    print("--- MLP probe vs Ridge on final ckpt (seed 0) ---")
    for v in variants:
        fa_path = base / f'{v}_seed0/final_analyses.json'
        if not fa_path.exists(): continue
        fa = json.loads(fa_path.read_text())
        cmp = fa.get('probe_comparison', {})
        ridge = cmp.get('ridge', {})
        mlp = cmp.get('mlp', {})
        if mlp:
            print(f"  {v:>12s}: Ridge={ridge.get('overall_r2',0):.3f}  MLP={mlp['overall_r2']:.3f}  "
                  f"Δ={cmp['delta_overall']:+.3f}  "
                  f"per-dim Δ={[round(x,2) for x in cmp.get('delta_per_dim',[])]}")
    print()


if __name__ == "__main__":
    main()
