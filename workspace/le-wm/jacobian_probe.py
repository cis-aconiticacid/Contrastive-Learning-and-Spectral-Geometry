"""
Jacobian diagnostic for the LeWM predictor.

Loads a trained checkpoint, runs a small batch through the encoder to get latent
embeddings, then computes the predictor's Jacobian at each latent and reports:

  - Top-k singular values (SV decay)
  - Stable rank ||J||_F^2 / ||J||_2^2
  - Effective rank exp( -sum p_i log p_i ), p_i = sigma_i^2 / sum sigma_j^2
  - Spectral norm ||J||_2

For a low-rank dynamical system the predictor Jacobian should have rapidly-decaying
singular values. Comparing baseline vs uv_lowr tells us whether the low-rank
architecture actually shows up in the learned Jacobian.

Usage:
    python jacobian_probe.py compare_runs/baseline/baseline_object.ckpt
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange


@torch.no_grad()
def get_sample_latents(world_model, dataset, n_samples=4, device="cuda"):
    """Encode a small batch of frames to get realistic latents."""
    indices = torch.randperm(len(dataset))[:n_samples]
    samples = [dataset[i.item()] for i in indices]
    pixels = torch.stack([s["pixels"] for s in samples]).to(device).float()
    actions = torch.stack([s["action"] for s in samples]).to(device).float()
    proprio = torch.stack([s["proprio"] for s in samples]).to(device).float()
    state = torch.stack([s["state"] for s in samples]).to(device).float()
    info = {"pixels": pixels, "action": actions, "proprio": proprio, "state": state}
    info = world_model.encode(info)
    # Use only the history window (first 3 frames)
    return info["emb"][:, :3], info["act_emb"][:, :3]


def predictor_fn(predictor, pred_proj, emb_in, act_in):
    """Wraps predictor + pred_proj as a single z_history -> z_history' map.

    emb_in: (T, D), act_in: (T, A_emb). Returns flat output (T*D,).
    """
    emb = emb_in.unsqueeze(0)  # (1, T, D)
    act = act_in.unsqueeze(0)  # (1, T, A_emb)
    out = predictor(emb, act)
    # apply prediction projector (per-token)
    out = pred_proj(rearrange(out, "b t d -> (b t) d"))
    return out.flatten()


def compute_jacobian(predictor, pred_proj, emb_in, act_in):
    """Returns Jacobian (T*D_out, T*D_in)."""
    from torch.func import jacrev
    emb_in = emb_in.detach().requires_grad_(False)
    act_in = act_in.detach().requires_grad_(False)
    fn = lambda e: predictor_fn(predictor, pred_proj, e, act_in)
    return jacrev(fn)(emb_in).reshape(-1, emb_in.numel())


def analyze_jacobian(J):
    """J: (out, in). Returns dict of spectrum stats."""
    sv = torch.linalg.svdvals(J).cpu()
    sv2 = sv.pow(2)
    fr = sv2.sum().item()                                # ||J||_F^2
    sp = sv.max().item()                                 # ||J||_2
    stable_rank = fr / max(sp ** 2, 1e-12)
    # entropy-effective rank
    p = sv2 / sv2.sum()
    entropy = -(p * (p.clamp_min(1e-12).log())).sum().item()
    eff_rank = math.exp(entropy)
    return {
        "spectral_norm": sp,
        "frobenius_norm_sq": fr,
        "stable_rank": stable_rank,
        "effective_rank": eff_rank,
        "top_svs": sv[:32].tolist(),
        "n_svs_above_1pct_max": int((sv > 0.01 * sp).sum().item()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", help="path to *_object.ckpt or weights.ckpt")
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--config", type=str, default=None,
                   help="Optional path to config.yaml (auto-detected if alongside ckpt)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- find config and load model ----
    ckpt_path = Path(args.ckpt)
    cfg_path = Path(args.config) if args.config else ckpt_path.parent / "config.yaml"

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(cfg_path)

    # Re-build the world model exactly as train.py does
    sys.path.insert(0, str(Path(__file__).parent))
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP, SIGReg
    from utils import get_column_normalizer, get_img_preprocessor

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale, patch_size=cfg.patch_size,
        image_size=cfg.img_size, pretrained=False, use_mask_token=False,
    )
    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim, hidden_dim=hidden_dim, output_dim=hidden_dim,
        **cfg.predictor,
    )
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    projector = MLP(hidden_dim, output_dim=embed_dim, hidden_dim=2048,
                    norm_fn=torch.nn.BatchNorm1d)
    predictor_proj = MLP(hidden_dim, output_dim=embed_dim, hidden_dim=2048,
                         norm_fn=torch.nn.BatchNorm1d)
    world_model = JEPA(encoder=encoder, predictor=predictor,
                       action_encoder=action_encoder, projector=projector,
                       pred_proj=predictor_proj)

    # ---- load weights ----
    # Try the *_weights.ckpt (state_dict only) first.
    # spt saves the latest weights as e.g. "baseline_weights.ckpt" (no epoch suffix).
    cand = ckpt_path.parent / f"{ckpt_path.stem.replace('_object', '')}_weights.ckpt"
    if cand.exists():
        weights_ckpt = cand
    else:
        # Fall back to the run-name-based weights file
        run_name = ckpt_path.parent.name
        weights_ckpt = ckpt_path.parent / f"{run_name}_weights.ckpt"
    if weights_ckpt.exists():
        print(f"loading weights from {weights_ckpt}")
        sd = torch.load(weights_ckpt, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        # strip "model." prefix from spt-saved state dict
        sd = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if not k.startswith("sigreg.")}
        try:
            world_model.load_state_dict(sd, strict=False)
            print("loaded weights (non-strict)")
        except Exception as e:
            print("strict load failed:", e)
    else:
        print(f"WARN: no weights file at {weights_ckpt}; using freshly initialized model")

    world_model.to(device).eval()

    # ---- get sample latents ----
    transforms = [get_img_preprocessor(source='pixels', target='pixels',
                                       img_size=cfg.img_size)]
    # build dataset without transforms first, then attach Compose at the end
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    for col in cfg.data.dataset.keys_to_load:
        if col.startswith("pixels"):
            continue
        normalizer = get_column_normalizer(dataset, col, col)
        transforms.append(normalizer)
    dataset.transform = spt.data.transforms.Compose(*transforms)

    embs, acts = get_sample_latents(world_model, dataset, n_samples=args.n_samples, device=device)
    print(f"sampled latents: emb {embs.shape}  act {acts.shape}")

    # ---- compute Jacobian per sample ----
    results = []
    n_params = sum(p.numel() for p in predictor.parameters())
    print(f"\npredictor params: {n_params:,}")
    for i in range(args.n_samples):
        J = compute_jacobian(predictor, predictor_proj, embs[i], acts[i])
        # J: (T*D_out, T*D_in)
        s = analyze_jacobian(J.detach().to("cpu"))
        s["sample"] = i
        results.append(s)
        print(f"sample {i}: J shape={J.shape}  spectral={s['spectral_norm']:.3f}  "
              f"stable_rank={s['stable_rank']:.2f}  eff_rank={s['effective_rank']:.2f}  "
              f"#sv>1%={s['n_svs_above_1pct_max']}")

    summary = {
        "ckpt": str(ckpt_path),
        "predictor_params": n_params,
        "n_samples": args.n_samples,
        "ffn_rank": cfg.predictor.get("ffn_rank") if "ffn_rank" in cfg.predictor else None,
        "samples": results,
        "mean_stable_rank": sum(r["stable_rank"] for r in results) / len(results),
        "mean_effective_rank": sum(r["effective_rank"] for r in results) / len(results),
        "mean_spectral_norm": sum(r["spectral_norm"] for r in results) / len(results),
        "mean_n_svs_above_1pct": sum(r["n_svs_above_1pct_max"] for r in results) / len(results),
    }
    print("\n=== SUMMARY ===")
    print(f"  ffn_rank:             {summary['ffn_rank']}")
    print(f"  predictor params:     {summary['predictor_params']:,}")
    print(f"  mean spectral norm:   {summary['mean_spectral_norm']:.4f}")
    print(f"  mean stable rank:     {summary['mean_stable_rank']:.2f}")
    print(f"  mean effective rank:  {summary['mean_effective_rank']:.2f}")
    print(f"  mean #SV > 1% peak:   {summary['mean_n_svs_above_1pct']:.1f}")

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
