"""R100: ε-bracket probe for v4 B1 randdiff ε sensitivity sweep.

Loads v3 Stage-1 frozen ckpt, encodes ~100 Push-T episodes, computes the
per-sample latent norm distribution on both CLS and post-projector latents.

The randdiff penalty (see train.py line 67-74) perturbs ctx_emb with a vector
of flat-unit-norm scaled by rd_eps. So the relevant scale for ε is the typical
per-sample norm ||z||₂. If frozen v3 latent has ||z|| substantially different
from what v2 had, then v3's reused ε=0.05 may have been a different
"effective" perturbation strength.

Outputs:
  refine-logs/v4_eps_bracket.json  — per-source latent norm stats + ε bracket
"""
import os, json, time
from pathlib import Path

import hdf5plugin  # noqa: F401  blosc plugin
import h5py
import numpy as np
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

H5    = "/workspace/stablewm_home/pusht_expert_train.h5"
CKPT  = "/workspace/lewm_autodl_results_v3/stage1_baseline_seed0_weights.ckpt"
OUT   = Path("/workspace/le-wm/refine-logs/v4_eps_bracket.json")
N_EPS = 100
BATCH = 32
SEED  = 0
DEV   = "cuda" if torch.cuda.is_available() else "cpu"


class Projector(nn.Module):
    def __init__(self, in_dim=192, hidden=2048, out_dim=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x): return self.net(x)


def load_enc_proj(ckpt_path):
    cfg = ViTConfig(image_size=224, patch_size=14, num_channels=3,
                    hidden_size=192, num_hidden_layers=12, num_attention_heads=3,
                    intermediate_size=768, qkv_bias=True)
    enc = ViTModel(cfg, add_pooling_layer=False)
    proj = Projector()
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    enc_sd  = {k[len("model.encoder."):]:  v for k, v in sd.items()
               if k.startswith("model.encoder.")}
    proj_sd = {k[len("model.projector."):]: v for k, v in sd.items()
               if k.startswith("model.projector.")}
    assert enc_sd, "no encoder keys in ckpt"
    assert proj_sd, "no projector keys in ckpt"
    enc.load_state_dict(enc_sd, strict=True)
    proj.load_state_dict(proj_sd, strict=True)
    enc.eval(); proj.eval()
    for p in list(enc.parameters()) + list(proj.parameters()):
        p.requires_grad_(False)
    return enc.to(DEV), proj.to(DEV)


@torch.no_grad()
def encode_eps(enc, proj, ep_indices, f):
    ep_len = f["ep_len"][:]; ep_off = f["ep_offset"][:]
    pix = f["pixels"]
    cls_all, proj_all = [], []
    for ep in ep_indices:
        o = int(ep_off[ep]); L = int(ep_len[ep])
        if L < 6:
            continue
        for start in range(0, L, BATCH):
            end = min(start + BATCH, L)
            px = pix[o+start:o+end]
            px = torch.from_numpy(px).float() / 255.0
            px = px.permute(0, 3, 1, 2).contiguous().to(DEV)
            o2 = enc(px, interpolate_pos_encoding=True)
            cls = o2.last_hidden_state[:, 0]            # (b, 192)
            zp  = proj(cls)                              # (b, 192)
            cls_all.append(cls.cpu().numpy())
            proj_all.append(zp.cpu().numpy())
    return np.concatenate(cls_all, 0), np.concatenate(proj_all, 0)


def norm_stats(Z, name):
    n = np.linalg.norm(Z, axis=1)                       # (N,)
    flat = np.linalg.norm(Z.reshape(Z.shape[0], -1), axis=1)
    s = {
        "name": name,
        "n_samples": int(Z.shape[0]),
        "dim": int(Z.shape[1]),
        "norm_mean":   float(n.mean()),
        "norm_std":    float(n.std()),
        "norm_p10":    float(np.percentile(n, 10)),
        "norm_p50":    float(np.percentile(n, 50)),
        "norm_p90":    float(np.percentile(n, 90)),
        "norm_min":    float(n.min()),
        "norm_max":    float(n.max()),
        "per_dim_std_mean": float(Z.std(0).mean()),
    }
    print(f"=== {name} (n={s['n_samples']}, d={s['dim']}) ===")
    print(f"  ||z||₂  mean={s['norm_mean']:.4f}  std={s['norm_std']:.4f}  "
          f"p10/p50/p90 = {s['norm_p10']:.3f} / {s['norm_p50']:.3f} / {s['norm_p90']:.3f}")
    print(f"  per-dim std (mean over 192 dims) = {s['per_dim_std_mean']:.4f}")
    return s


def epsilon_bracket(post_proj_mean_norm):
    """Pick 4 ε values for the v4 B1 sweep.

    Reasoning: randdiff perturbs z with a vector of flat-unit-norm scaled by ε.
    So the "fractional perturbation" is ε / ||z||₂. v2/v3 used ε=0.05 without
    rescaling. We want a bracket that spans 1/4× to 16× of v3's effective scale,
    ALWAYS including the v3 value as a control.
    """
    return [
        {"eps": 0.0125, "label": "ε_1 (¼ × v3)",  "frac": 0.0125 / post_proj_mean_norm},
        {"eps": 0.05,   "label": "ε_2 = v3",        "frac": 0.05   / post_proj_mean_norm},
        {"eps": 0.2,    "label": "ε_3 (4× v3)",   "frac": 0.2    / post_proj_mean_norm},
        {"eps": 0.8,    "label": "ε_4 (16× v3)",  "frac": 0.8    / post_proj_mean_norm},
    ]


def main():
    print(f"[R100] device={DEV}")
    print(f"[R100] ckpt={CKPT}")
    t0 = time.time()
    enc, proj = load_enc_proj(CKPT)
    print(f"[R100] loaded encoder+projector in {time.time()-t0:.1f}s")

    rng = np.random.default_rng(SEED)
    with h5py.File(H5, "r") as f:
        n_eps_total = len(f["ep_len"])
        ep_idx = rng.choice(n_eps_total, size=min(N_EPS, n_eps_total), replace=False)
        print(f"[R100] selected {len(ep_idx)} / {n_eps_total} episodes")
        t1 = time.time()
        Z_cls, Z_proj = encode_eps(enc, proj, ep_idx, f)
        print(f"[R100] encoded in {time.time()-t1:.1f}s  Z_cls={Z_cls.shape}  Z_proj={Z_proj.shape}")

    cls_stats  = norm_stats(Z_cls,  "z_cls  (encoder CLS, pre-projector)")
    proj_stats = norm_stats(Z_proj, "z_proj (post-projector — predictor input)")

    bracket = epsilon_bracket(proj_stats["norm_mean"])
    print("\n=== recommended ε bracket for B1 sweep ===")
    print(f"  reference: post-projector ||z||₂ mean = {proj_stats['norm_mean']:.4f}")
    for b in bracket:
        print(f"  {b['label']:<14}  ε={b['eps']:<7}  fractional ε/||z|| = {b['frac']:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "ckpt": CKPT,
        "n_episodes": int(len(ep_idx)),
        "z_cls":  cls_stats,
        "z_proj": proj_stats,
        "epsilon_bracket": bracket,
        "note": ("v3 used ε=0.05 unrescaled. The four ε values span 64× geometric range "
                 "around v3's value. B1 runs frozen_randdiff at each ε with seed 0, "
                 "8K steps, on v3 Stage-1 ckpt. Looking for v2-like σ₁ inflation at "
                 "any ε; absence at all 4 falsifies Anti-claim A."),
    }, indent=2))
    print(f"\n[R100] saved -> {OUT}")


if __name__ == "__main__":
    main()
