"""Post-hoc spectral analysis: predictor latent->latent Jacobian eigendecomp + TwoNN.

EXPERIMENT_PLAN_v5 requires |lambda_max|, n_unstable, spectral_abscissa, and TwoNN
intrinsic dim. The training-time JacobianProbeCallback only computes SVD of the
*end-to-end* (predictor + pred_proj, state-dim output) Jacobian, which is non-square
and has no eigenvalues. This script fixes that by computing the *latent-only*
predictor Jacobian (D x D, square) on a saved checkpoint and writing the missing
fields back to final_analyses.json.

Usage:
    python spectral_postprocess.py <ckpt_path> [--n-samples 6] [--twonn-n 2048]

Output: appends/overwrites keys in <ckpt_dir>/final_analyses.json:
    spectral_eig: {mean_lambda_max, mean_n_unstable, mean_spectral_abscissa, samples: [...]}
    twonn_intrinsic_dim: float
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.func import jacrev

sys.path.insert(0, str(Path(__file__).parent))
from final_analyses import build_world_model_and_load
from eval_rollout import build_dataset


def predictor_latent_jacobian(predictor, emb_prefix, e_last, act_in):
    """Jacobian d predictor(emb_full)[:, -1] / d e_last, returning D x D matrix.

    emb_prefix: (T-1, D) fixed context (no grad)
    e_last:     (D,)    the variable last context frame
    act_in:     (T, A_emb) fixed actions
    """
    def fn(e):
        emb_full = torch.cat([emb_prefix, e.unsqueeze(0)], dim=0).unsqueeze(0)  # (1, T, D)
        out = predictor(emb_full, act_in.unsqueeze(0))                          # (1, T, D)
        return out[0, -1]                                                       # (D,)
    J = jacrev(fn)(e_last)
    return J  # shape (D, D)


def analyze_eigendecomp(J: torch.Tensor):
    """Return dict with lambda_max, n_unstable, spectral_abscissa from a square J."""
    Jc = J.detach().cpu().double()
    eigvals = torch.linalg.eigvals(Jc)  # complex
    mag = eigvals.abs()
    real = eigvals.real
    return {
        "lambda_max":         float(mag.max().item()),
        "n_unstable":         int((mag > 1.0).sum().item()),
        "spectral_abscissa":  float(real.max().item()),
        "n_complex":          int((eigvals.imag.abs() > 1e-9).sum().item()),
        "shape": list(J.shape),
    }


def twonn_intrinsic_dim(X: torch.Tensor):
    """Facco et al. (2017) Two-NN intrinsic dimensionality.

    X: (N, D) latents.
    Returns scalar intrinsic dim estimate.
    """
    X = X.detach().cpu()
    # pairwise distances
    D = torch.cdist(X, X)
    D.fill_diagonal_(float("inf"))
    # for each point, take r1 (nearest) and r2 (2nd-nearest)
    sorted_D, _ = D.sort(dim=1)
    r1 = sorted_D[:, 0]
    r2 = sorted_D[:, 1]
    # filter out degenerate cases
    mask = (r1 > 1e-9) & (r2 > r1)
    mu = (r2[mask] / r1[mask])
    # max-likelihood estimate (Facco eq. 4): d = N / sum(log mu)
    return float(mu.numel() / torch.log(mu).sum().item())


def collect_latents(wm, dataset, n_samples, batch_size, device, seed=0,
                     source="encoder"):
    """Encode n_samples frames using strided cross-trajectory sampling.

    Earlier version reshaped (B, T, D) -> (B*T, D) which packed T adjacent frames
    from the same trajectory together. Adjacent-frame latents are near-identical
    -> r1 ~ 0 -> TwoNN's mu = r2/r1 blows up -> d_intrinsic underestimated (~0.45).
    Fix: shuffle=True and take only the last frame of each window per batch.

    source:
      "encoder"   -> encoder output emb[:, -1] (data-prior intrinsic dim;
                     identical across variants when encoder is frozen)
      "predictor" -> predictor next-step prediction pred[:, -1] (the manifold
                     that randdiff/uvlowr actually contract)
    """
    from torch.utils.data import DataLoader
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, generator=g)
    latents = []
    for batch in loader:
        if sum(l.shape[0] for l in latents) >= n_samples:
            break
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        if "action" in batch:
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)
        with torch.no_grad():
            info = wm.encode(batch)
            if source == "encoder":
                z = info["emb"][:, -1]  # (B, D)
            elif source == "predictor":
                pred = wm.predictor(info["emb"], info["act_emb"])  # (B, T, D)
                z = pred[:, -1]  # next-step prediction
            else:
                raise ValueError(f"unknown source={source!r}")
        latents.append(z.cpu())
    return torch.cat(latents, dim=0)[:n_samples]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="*_epoch_1_object.ckpt path")
    ap.add_argument("--n-samples", type=int, default=6,
                    help="number of fixed inputs for Jacobian eigendecomp")
    ap.add_argument("--twonn-n",   type=int, default=2048,
                    help="number of latents for TwoNN")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    print(f"=== spectral_postprocess {ckpt_path.name} ===", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm, cfg = build_world_model_and_load(ckpt_path, device)
    wm.eval()

    history_size = int(cfg.wm.history_size)
    # need at least T = history_size frames for predictor input; pred_proj is bypassed
    ds = build_dataset(cfg, num_steps_override=history_size)

    # collect fixed inputs (use first n_samples examples deterministically)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=args.n_samples, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
    if "action" in batch:
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    with torch.no_grad():
        info = wm.encode(batch)
    emb_full = info["emb"][:, :history_size]      # (N, T, D)
    act_full = info["act_emb"][:, :history_size]  # (N, T, A_emb)
    D = emb_full.shape[-1]

    samples = []
    for i in range(args.n_samples):
        e_prefix = emb_full[i, :-1].detach()       # (T-1, D)
        e_last   = emb_full[i, -1].detach()        # (D,)
        a_in     = act_full[i].detach()            # (T, A_emb)
        J = predictor_latent_jacobian(wm.predictor, e_prefix, e_last, a_in)
        stats = analyze_eigendecomp(J)
        stats["sample"] = i
        samples.append(stats)
        print(f"  sample {i}: lam_max={stats['lambda_max']:.4f}  n_unstable={stats['n_unstable']}  abscissa={stats['spectral_abscissa']:.4f}", flush=True)

    summary = {
        "mean_lambda_max":        float(np.mean([s["lambda_max"]        for s in samples])),
        "mean_n_unstable":        float(np.mean([s["n_unstable"]        for s in samples])),
        "mean_spectral_abscissa": float(np.mean([s["spectral_abscissa"] for s in samples])),
        "std_lambda_max":         float(np.std ([s["lambda_max"]        for s in samples])),
        "n_samples": args.n_samples,
        "samples": samples,
    }

    # TwoNN intrinsic dim — both encoder output AND predictor output.
    # For frozen-encoder runs the encoder TwoNN is identical across variants
    # (data-prior), while predictor TwoNN reflects the manifold the predictor
    # actually contracts. The latter is the paper-relevant quantity.
    print("  collecting encoder latents for TwoNN...", flush=True)
    enc_lat = collect_latents(wm, ds, args.twonn_n, args.batch_size, device,
                               source="encoder")
    enc_twonn = twonn_intrinsic_dim(enc_lat)
    print(f"  TwoNN(encoder)   = {enc_twonn:.2f}  (N={enc_lat.shape[0]}, D={enc_lat.shape[1]})", flush=True)

    print("  collecting predictor latents for TwoNN...", flush=True)
    pred_lat = collect_latents(wm, ds, args.twonn_n, args.batch_size, device,
                                source="predictor")
    pred_twonn = twonn_intrinsic_dim(pred_lat)
    print(f"  TwoNN(predictor) = {pred_twonn:.2f}  (N={pred_lat.shape[0]}, D={pred_lat.shape[1]})", flush=True)

    out_path = ckpt_path.parent / "final_analyses.json"
    if out_path.exists():
        existing = json.loads(out_path.read_text())
    else:
        existing = {}
    existing["spectral_eig"] = summary
    # keep legacy field name pointing to encoder TwoNN for backward compat
    existing["twonn_intrinsic_dim"] = enc_twonn
    existing["twonn_n_samples"] = enc_lat.shape[0]
    existing["twonn_enc_intrinsic_dim"]  = enc_twonn
    existing["twonn_pred_intrinsic_dim"] = pred_twonn
    out_path.write_text(json.dumps(existing, indent=2))
    print(f"  wrote spectral_eig + twonn_{{enc,pred}}_intrinsic_dim -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
