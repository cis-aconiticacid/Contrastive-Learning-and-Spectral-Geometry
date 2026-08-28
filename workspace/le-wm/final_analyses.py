"""Three one-shot final-checkpoint analyses (post-training, single ckpt):

1. weight_svd  — for every Linear in the predictor, compute weight-matrix SVD,
                 report top-k SVs, stable rank, effective rank. Tells us per-layer
                 effective dimensionality.
2. encoder_jacobian — Jacobian of encoder w.r.t. raw input pixels, on N samples.
                 Expensive: input dim is 3*224*224 = 150K, output is 192. Computed
                 via vmap+jacrev. Reports SVD spectrum.
3. mlp_probe   — 2-layer MLP probe of latent → state, compared to Ridge probe.
                 If MLP >> Ridge, info is THERE but non-linearly entangled.

Usage:
    python final_analyses.py {ckpt_path} --out {dir} \
        [--skip-encoder-jac] [--skip-mlp-probe]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Reuse spectrum analysis
sys.path.insert(0, str(Path(__file__).parent))
from callbacks import _analyze_spectrum  # noqa: E402


# ---------------------------------------------------------------------------
#  1. Per-weight SVD (predictor only)
# ---------------------------------------------------------------------------

def per_weight_svd(predictor: nn.Module):
    """Return dict {layer_name -> spectrum stats} for every Linear weight."""
    out = {}
    for name, mod in predictor.named_modules():
        if isinstance(mod, nn.Linear):
            W = mod.weight.detach()
            sv = torch.linalg.svdvals(W)
            stats = _analyze_spectrum(sv)
            stats["shape"] = list(W.shape)
            stats["params"] = int(W.numel())
            out[name] = stats
    return out


# ---------------------------------------------------------------------------
#  2. Encoder Jacobian (input pixels -> latent)
# ---------------------------------------------------------------------------

def encoder_jacobian(world_model, fixed_pixels: torch.Tensor, max_samples: int = 4):
    """Compute encoder Jacobian (input -> latent) on `max_samples` images.

    Input pixels: (N, C, H, W). Output: latent (D,).
    Jacobian shape: (D, C*H*W). For ViT-Tiny D=192, C*H*W=150528 -> 28M entries
    per sample. Storing 4 such Jacobians = ~450 MB. Manageable.
    """
    encoder = world_model.encoder
    projector = world_model.projector
    encoder.eval(); projector.eval()

    def encode_one(x):
        # x: (C, H, W) -> latent (D,)
        out = encoder(x.unsqueeze(0).float(), interpolate_pos_encoding=True)
        cls = out.last_hidden_state[:, 0]
        z = projector(cls)
        return z.flatten()  # (D,)

    from torch.func import jacrev
    results = []
    n = min(max_samples, fixed_pixels.size(0))
    for i in range(n):
        x = fixed_pixels[i].detach().requires_grad_(False)
        t0 = time.time()
        J = jacrev(encode_one)(x)  # (D, C, H, W)
        J = J.reshape(J.size(0), -1)  # (D, C*H*W)
        sv = torch.linalg.svdvals(J)
        stats = _analyze_spectrum(sv)
        stats["sample"] = i
        stats["jacobian_shape"] = list(J.shape)
        stats["compute_time_s"] = time.time() - t0
        results.append(stats)
        print(f"  encoder_jac sample {i}: spectral={stats['spectral_norm']:.3f} "
              f"sr={stats['stable_rank']:.1f} er={stats['effective_rank']:.1f} "
              f"({stats['compute_time_s']:.1f}s)", flush=True)
    summary = {
        "n_samples": n,
        "mean_spectral_norm": sum(r["spectral_norm"] for r in results) / max(1, len(results)),
        "mean_stable_rank": sum(r["stable_rank"] for r in results) / max(1, len(results)),
        "mean_effective_rank": sum(r["effective_rank"] for r in results) / max(1, len(results)),
        "samples": results,
    }
    return summary


# ---------------------------------------------------------------------------
#  3. MLP probe baseline
# ---------------------------------------------------------------------------

def mlp_probe(Z_tr, S_tr, Z_te, S_te, hidden=256, lr=1e-3, epochs=200, wd=1e-4,
              device="cuda", seed=0):
    """Train a 2-layer MLP probe Z -> S, return R²."""
    torch.manual_seed(seed)
    Zt = torch.from_numpy(Z_tr).float().to(device)
    St = torch.from_numpy(S_tr).float().to(device)
    Zv = torch.from_numpy(Z_te).float().to(device)
    Sv = torch.from_numpy(S_te).float().to(device)
    D = Z_tr.shape[1]; S = S_tr.shape[1]

    net = nn.Sequential(
        nn.Linear(D, hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, S),
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    losses = []
    for ep in range(epochs):
        pred = net(Zt)
        loss = F.mse_loss(pred, St)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    net.eval()
    with torch.no_grad():
        pred_te = net(Zv).cpu().numpy()
    from sklearn.metrics import r2_score
    overall = float(r2_score(S_te, pred_te))
    per_dim = [float(r2_score(S_te[:, d], pred_te[:, d])) for d in range(S_te.shape[1])]
    return {
        "overall_r2": overall,
        "per_dim_r2": per_dim,
        "mse": float(((pred_te - S_te) ** 2).mean()),
        "hidden": hidden, "epochs": epochs, "lr": lr, "wd": wd,
        "final_train_loss": losses[-1],
    }


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------

def build_world_model_and_load(ckpt_path: Path, device):
    """Re-build JEPA world model from config and load weights."""
    from omegaconf import OmegaConf
    import stable_pretraining as spt
    import stable_worldmodel as swm
    sys.path.insert(0, str(Path(__file__).parent))
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    cfg = OmegaConf.load(ckpt_path.parent / "config.yaml")

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale, patch_size=cfg.patch_size,
        image_size=cfg.img_size, pretrained=False, use_mask_token=False,
    )
    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim
    predictor = ARPredictor(num_frames=cfg.wm.history_size,
                            input_dim=embed_dim, hidden_dim=hidden_dim, output_dim=hidden_dim,
                            **cfg.predictor)
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    projector = MLP(hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=nn.BatchNorm1d)
    pred_proj = MLP(hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=nn.BatchNorm1d)
    wm = JEPA(encoder=encoder, predictor=predictor, action_encoder=action_encoder,
              projector=projector, pred_proj=pred_proj)

    # Find weights file
    cand = next(ckpt_path.parent.glob("*_weights.ckpt"))
    print(f"loading weights from {cand}")
    sd = torch.load(cand, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("model.", "", 1) if k.startswith("model.") else k: v
          for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not k.startswith("sigreg.")}
    wm.load_state_dict(sd, strict=False)
    return wm.to(device).eval(), cfg


def collect_latent_state_pairs(wm, dataset, n_samples, batch_size, device, history_size):
    """Encode last-frame pixels, return (Z, S) numpy arrays."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=0, drop_last=False, prefetch_factor=None)
    all_z, all_s = [], []
    with torch.no_grad():
        for batch in loader:
            if sum(z.shape[0] for z in all_z) >= n_samples:
                break
            pixels = batch["pixels"][:, history_size - 1].to(device).float()
            out = wm.encoder(pixels, interpolate_pos_encoding=True)
            cls = out.last_hidden_state[:, 0]
            z = wm.projector(cls).cpu().numpy()
            s = batch["state"][:, history_size - 1].numpy()
            all_z.append(z); all_s.append(s)
    Z = np.concatenate(all_z, 0)[:n_samples]
    S = np.concatenate(all_s, 0)[:n_samples]
    return Z, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--skip-encoder-jac", action="store_true")
    ap.add_argument("--skip-mlp-probe", action="store_true")
    ap.add_argument("--encoder-jac-samples", type=int, default=4)
    ap.add_argument("--mlp-n-train", type=int, default=2048)
    ap.add_argument("--mlp-n-test", type=int, default=512)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    wm, cfg = build_world_model_and_load(ckpt_path, device)

    summary = {"ckpt": str(ckpt_path)}

    # 1. Per-weight SVD
    print("\n=== 1. Per-weight SVD on predictor ===")
    summary["weight_svd"] = per_weight_svd(wm.predictor)
    for name, s in summary["weight_svd"].items():
        print(f"  {name:>40s}  shape={s['shape']}  sr={s['stable_rank']:.2f}  er={s['effective_rank']:.2f}")

    # 2. Encoder Jacobian
    if not args.skip_encoder_jac:
        print(f"\n=== 2. Encoder Jacobian on {args.encoder_jac_samples} samples ===")
        # Build dataset for sampling
        import stable_worldmodel as swm
        import stable_pretraining as spt
        from utils import get_column_normalizer, get_img_preprocessor
        ds_cfg = dict(cfg.data.dataset)
        ds_cfg["num_steps"] = cfg.wm.history_size
        transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
        dataset = swm.data.HDF5Dataset(**ds_cfg, transform=None)
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"): continue
            transforms.append(get_column_normalizer(dataset, col, col))
        dataset.transform = spt.data.transforms.Compose(*transforms)
        # Sample N pixels
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.encoder_jac_samples,
                                             shuffle=False, num_workers=0)
        batch = next(iter(loader))
        pixels = batch["pixels"][:, cfg.wm.history_size - 1].to(device).float()
        summary["encoder_jacobian"] = encoder_jacobian(wm, pixels, args.encoder_jac_samples)
    else:
        print("\n=== 2. Encoder Jacobian — SKIPPED ===")

    # 3. MLP probe (compare to Ridge)
    if not args.skip_mlp_probe:
        print(f"\n=== 3. MLP probe vs Ridge (n_tr={args.mlp_n_train}, n_te={args.mlp_n_test}) ===")
        import stable_worldmodel as swm
        import stable_pretraining as spt
        from utils import get_column_normalizer, get_img_preprocessor
        ds_cfg = dict(cfg.data.dataset)
        ds_cfg["num_steps"] = cfg.wm.history_size
        transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
        dataset = swm.data.HDF5Dataset(**ds_cfg, transform=None)
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"): continue
            transforms.append(get_column_normalizer(dataset, col, col))
        dataset.transform = spt.data.transforms.Compose(*transforms)

        Z, S = collect_latent_state_pairs(wm, dataset,
                                          n_samples=args.mlp_n_train + args.mlp_n_test,
                                          batch_size=64, device=device,
                                          history_size=cfg.wm.history_size)
        Z_tr, S_tr = Z[:args.mlp_n_train], S[:args.mlp_n_train]
        Z_te, S_te = Z[args.mlp_n_train:], S[args.mlp_n_train:]

        # Ridge baseline
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score
        ridge = Ridge(alpha=1.0); ridge.fit(Z_tr, S_tr)
        rp = ridge.predict(Z_te)
        ridge_r2 = float(r2_score(S_te, rp))
        ridge_per = [float(r2_score(S_te[:, d], rp[:, d])) for d in range(S_te.shape[1])]
        # MLP
        mlp_res = mlp_probe(Z_tr, S_tr, Z_te, S_te, device=device)
        summary["probe_comparison"] = {
            "ridge": {"overall_r2": ridge_r2, "per_dim_r2": ridge_per,
                      "n_train": args.mlp_n_train, "n_test": args.mlp_n_test},
            "mlp": mlp_res,
            "delta_overall": mlp_res["overall_r2"] - ridge_r2,
            "delta_per_dim": [m - r for m, r in zip(mlp_res["per_dim_r2"], ridge_per)],
        }
        print(f"  Ridge overall R² = {ridge_r2:.4f}")
        print(f"  MLP   overall R² = {mlp_res['overall_r2']:.4f}  (Δ = {summary['probe_comparison']['delta_overall']:+.4f})")
        print(f"  per-dim Δ (MLP - Ridge) = " +
              str([f'{x:+.3f}' for x in summary['probe_comparison']['delta_per_dim']]))
    else:
        print("\n=== 3. MLP probe — SKIPPED ===")

    out = out_dir / "final_analyses.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
