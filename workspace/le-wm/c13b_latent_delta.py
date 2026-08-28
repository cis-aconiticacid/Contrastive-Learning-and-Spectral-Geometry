"""C13b: latent-delta intrinsic dim.

The predictor maps z_t -> z_{t+1}. The thing it actually needs to model is the
distribution of delta-latents z_{t+5} - z_t, not raw state deltas.

This is C13's direct test: does latent-delta effective rank match uvlowr r=4?
"""
import os, time
import hdf5plugin  # noqa: F401  blosc
import h5py
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from transformers import ViTModel, ViTConfig

H5 = '/workspace/stablewm_home/pusht_expert_train.h5'
FRAMESKIP = 5
N_EPISODES = 200       # episodes to encode
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 0


def effective_rank(eig):
    p = np.asarray(eig, dtype=np.float64); p = p[p > 0]
    if p.size == 0: return 0.0
    p = p / p.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def participation_ratio(eig):
    e = np.asarray(eig, dtype=np.float64); e = e[e > 0]
    if e.size == 0: return 0.0
    return float((e.sum()**2) / (e**2).sum())


class Projector(nn.Module):
    """Matches le-wm MLP: Linear(in,h) -> BN1d(h) -> GELU -> Linear(h,out)."""
    def __init__(self, in_dim=192, hidden=2048, out_dim=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x): return self.net(x)


def load_encoder_and_projector(ckpt_path):
    cfg = ViTConfig(image_size=224, patch_size=14, num_channels=3,
                    hidden_size=192, num_hidden_layers=12, num_attention_heads=3,
                    intermediate_size=768, qkv_bias=True)
    enc = ViTModel(cfg, add_pooling_layer=False)
    proj = Projector()
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd: sd = sd['state_dict']
    enc_sd  = {k[len('model.encoder.'):]:  v for k, v in sd.items() if k.startswith('model.encoder.')}
    proj_sd = {k[len('model.projector.'):]: v for k, v in sd.items() if k.startswith('model.projector.')}
    enc.load_state_dict(enc_sd, strict=True)
    proj.load_state_dict(proj_sd, strict=True)
    enc.eval(); proj.eval()
    for p in list(enc.parameters()) + list(proj.parameters()): p.requires_grad_(False)
    return enc.to(DEVICE), proj.to(DEVICE)


@torch.no_grad()
def encode_episodes(enc, proj, ep_indices, f):
    """Encode all frames; return list of (L, 192) CLS and (L, 192) projected."""
    ep_len = f['ep_len'][:]; ep_off = f['ep_offset'][:]
    pixels = f['pixels']
    cls_eps, proj_eps = [], []
    BATCH = 64
    for ep in ep_indices:
        o = int(ep_off[ep]); L = int(ep_len[ep])
        if L <= FRAMESKIP: continue
        zs_cls, zs_proj = [], []
        for start in range(0, L, BATCH):
            end = min(start + BATCH, L)
            px = pixels[o+start:o+end]
            px = torch.from_numpy(px).float() / 255.0
            px = px.permute(0, 3, 1, 2).contiguous().to(DEVICE)
            o2 = enc(px, interpolate_pos_encoding=True)
            cls = o2.last_hidden_state[:, 0]  # (b, 192)
            proj_out = proj(cls) if proj is not None else cls
            zs_cls.append(cls.cpu().numpy())
            zs_proj.append(proj_out.cpu().numpy())
        cls_eps.append(np.concatenate(zs_cls, axis=0))
        proj_eps.append(np.concatenate(zs_proj, axis=0))
    return cls_eps, proj_eps


def pca_report(X, name):
    Xc = X - X.mean(0, keepdims=True)
    C = (Xc.T @ Xc) / max(Xc.shape[0]-1, 1)
    eig = np.sort(np.linalg.eigvalsh(C))[::-1]
    eig_n = eig / max(eig.sum(), 1e-12)
    cum = np.cumsum(eig_n)
    d90 = int(np.searchsorted(cum, 0.90) + 1)
    d95 = int(np.searchsorted(cum, 0.95) + 1)
    d99 = int(np.searchsorted(cum, 0.99) + 1)
    er = effective_rank(eig)
    pr = participation_ratio(eig)
    print(f"\n=== {name}  (n={X.shape[0]}, d={X.shape[1]}) ===")
    print(f"  top eig: {np.array2string(eig[:15], precision=3, suppress_small=True)}")
    print(f"  norm:    {np.array2string(eig_n[:15], precision=3, suppress_small=True)}")
    print(f"  cum:     {np.array2string(cum[:15], precision=3, suppress_small=True)}")
    print(f"  → effective_rank      = {er:.3f}")
    print(f"  → participation_ratio = {pr:.3f}")
    print(f"  → dim @ 90/95/99      = {d90}/{d95}/{d99}")
    return {'name': name, 'n': int(X.shape[0]), 'd': int(X.shape[1]),
            'eigvals': eig.tolist(), 'eigvals_normalized': eig_n.tolist(),
            'cum_var': cum.tolist(), 'effective_rank': er,
            'participation_ratio': pr, 'dim_at_90': d90,
            'dim_at_95': d95, 'dim_at_99': d99}


def main():
    rng = np.random.default_rng(SEED)
    print(f"Device: {DEVICE}")
    enc, proj = load_encoder_and_projector(
        '/workspace/stablewm_home/baseline_seed0/baseline_seed0_weights.ckpt')
    print(f"Loaded LeWM baseline_seed0 encoder + projector")

    with h5py.File(H5, 'r') as f:
        n_eps = len(f['ep_len'])
        ep_idx = rng.choice(n_eps, size=N_EPISODES, replace=False)
        t0 = time.time()
        cls_eps, proj_eps = encode_episodes(enc, proj, ep_idx, f)
        print(f"\nEncoded {len(cls_eps)} episodes in {time.time()-t0:.1f}s")

    z_cls  = np.concatenate(cls_eps, axis=0)
    z_proj = np.concatenate(proj_eps, axis=0)
    print(f"CLS  {z_cls.shape}   PROJ {z_proj.shape}")

    def deltas(ep_list):
        ds = []
        for zs in ep_list:
            if zs.shape[0] > FRAMESKIP:
                ds.append(zs[FRAMESKIP:] - zs[:-FRAMESKIP])
        return np.concatenate(ds, axis=0)

    d_cls  = deltas(cls_eps)
    d_proj = deltas(proj_eps)
    print(f"Δ-CLS {d_cls.shape}   Δ-PROJ {d_proj.shape}")

    results = {
        'cls_z':       pca_report(z_cls,  "CLS z  (encoder out, 192-d)"),
        'cls_delta':   pca_report(d_cls,  f"CLS Δ  (z_{{t+{FRAMESKIP}}}-z_t, 192-d)"),
        'proj_z':      pca_report(z_proj, "PROJ z (post-projector, 192-d)"),
        'proj_delta':  pca_report(d_proj, f"PROJ Δ (z_{{t+{FRAMESKIP}}}-z_t post-proj, 192-d)"),
    }

    out = Path('/workspace/le-wm/refine-logs/c13b_latent_delta.json')
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == '__main__':
    main()
