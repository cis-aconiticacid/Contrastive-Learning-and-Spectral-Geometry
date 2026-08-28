"""C6 control: encoder Jacobian SR for ViT-Tiny at three training stages.

(A) Random-init ViT-Tiny (no training at all)
(B) ImageNet-pretrained ViT-Tiny (timm)
(C) JEPA-trained ViT-Tiny (our LeWM baseline_seed0 → CLS only, no projector)

Compares to our v2 result (encoder+projector, SR≈3.05). Tests whether the very
low Jacobian stable rank is JEPA-specific or already present in random/ImageNet ViTs.

Output dim is 192 in all cases (ViT-Tiny CLS).
Input dim is 3*224*224 = 150528.
Samples: 4 random Push-T images for direct comparability with v2.
"""
import hdf5plugin  # blosc decompressor for Push-T pixels
import h5py
import json
import time
import math
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn

def _analyze_spectrum(sv):
    sv = sv.detach().cpu()
    sv2 = sv.pow(2)
    fr = sv2.sum().item()
    sp = sv.max().item() if sv.numel() > 0 else 0.0
    stable_rank = fr / max(sp ** 2, 1e-12)
    p = sv2 / sv2.sum().clamp_min(1e-12)
    entropy = -(p * (p.clamp_min(1e-12).log())).sum().item()
    eff_rank = math.exp(entropy)
    return {
        "spectral_norm": sp,
        "frobenius_norm_sq": fr,
        "stable_rank": stable_rank,
        "effective_rank": eff_rank,
        "n_svs_above_1pct_max": int((sv > 0.01 * sp).sum().item()),
    }


H5 = '/workspace/stablewm_home/pusht_expert_train.h5'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SAMPLES = 4
SEED = 0


def sample_pixels():
    """Sample N_SAMPLES Push-T frames, return tensor (N, 3, 224, 224) float32 in [0,1]."""
    rng = np.random.default_rng(SEED)
    import os; os.environ.setdefault('HDF5_PLUGIN_PATH', '/tmp')
    with h5py.File(H5, 'r') as f:
        N = f['pixels'].shape[0]
        idx = np.sort(rng.choice(N, size=N_SAMPLES, replace=False))
        # Direct indexed reads — avoid loading all 2.3M frames
        px = np.stack([f['pixels'][int(i)] for i in idx], axis=0)
    px = torch.from_numpy(px).float() / 255.0  # (N, H, W, C)
    px = px.permute(0, 3, 1, 2).contiguous()  # (N, C, H, W)
    return px.to(DEVICE)


def jac_spectrum(model, pixels, label):
    """Compute Jacobian SVD for each sample; return mean stats + per-sample."""
    from torch.func import jacrev
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def f_one(x):
        return model(x.unsqueeze(0)).flatten()

    samples = []
    for i in range(pixels.size(0)):
        x = pixels[i].detach()
        t0 = time.time()
        J = jacrev(f_one)(x)  # (D, C, H, W)
        J = J.reshape(J.size(0), -1)  # (D, 150528)
        sv = torch.linalg.svdvals(J.float())
        stats = _analyze_spectrum(sv)
        stats['frobenius_norm_sq'] = float((sv**2).sum().item())
        stats['top_svs'] = sv[:32].tolist()
        stats['sample'] = i
        stats['compute_time_s'] = time.time() - t0
        samples.append(stats)
        print(f"  [{label}] sample {i}: σ₁={stats['spectral_norm']:.3f} "
              f"SR={stats['stable_rank']:.2f} ER={stats['effective_rank']:.2f} "
              f"F²={stats['frobenius_norm_sq']:.1f} ({stats['compute_time_s']:.1f}s)",
              flush=True)
    return {
        'n_samples': len(samples),
        'mean_spectral_norm': sum(s['spectral_norm'] for s in samples) / len(samples),
        'mean_stable_rank':   sum(s['stable_rank']   for s in samples) / len(samples),
        'mean_effective_rank':sum(s['effective_rank']for s in samples) / len(samples),
        'mean_frobenius_sq':  sum(s['frobenius_norm_sq'] for s in samples) / len(samples),
        'samples': samples,
    }


def load_lewm_encoder():
    """Load LeWM baseline_seed0 encoder (CLS head only, no projector)."""
    ckpt_path = '/workspace/stablewm_home/baseline_seed0/baseline_seed0_weights.ckpt'
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    else:
        sd = ckpt
    # ckpt keys look like: model.encoder.{embeddings,encoder}.layer...
    enc_sd = {}
    for k, v in sd.items():
        if not k.startswith('model.encoder.'):
            continue
        new_k = k[len('model.encoder.'):]
        enc_sd[new_k] = v
    print(f"  LeWM ckpt: {len(enc_sd)} encoder keys")
    # the encoder in stable_pretraining is a HuggingFace ViTModel
    from transformers import ViTModel, ViTConfig
    # LeWM uses patch_size=14 (224/14 = 16 → 257 tokens incl CLS)
    cfg = ViTConfig(image_size=224, patch_size=14, num_channels=3,
                    hidden_size=192, num_hidden_layers=12, num_attention_heads=3,
                    intermediate_size=768, qkv_bias=True)
    enc = ViTModel(cfg, add_pooling_layer=False)
    missing, unexpected = enc.load_state_dict(enc_sd, strict=False)
    print(f"  missing: {len(missing)} unexpected: {len(unexpected)}")
    if missing[:3]: print('   missing[:3]:', missing[:3])
    if unexpected[:3]: print('   unexpected[:3]:', unexpected[:3])

    class HFEncCLS(nn.Module):
        def __init__(self, enc): super().__init__(); self.enc = enc
        def forward(self, x):
            out = self.enc(x, interpolate_pos_encoding=True)
            return out.last_hidden_state[:, 0]
    return HFEncCLS(enc).to(DEVICE)


def main():
    print(f"Device: {DEVICE}")
    pixels = sample_pixels()
    print(f"Sampled pixels: {pixels.shape}, dtype={pixels.dtype}, range[{pixels.min():.3f}, {pixels.max():.3f}]")

    results = {}

    # ---- (A) Random-init ViT-Tiny ----
    print(f"\n=== (A) Random-init ViT-Tiny → CLS ===")
    torch.manual_seed(SEED)
    m_rand = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=0).to(DEVICE)
    results['random_vit_tiny'] = jac_spectrum(m_rand, pixels, 'RAND')
    del m_rand; torch.cuda.empty_cache()

    # ---- (B) ImageNet-pretrained ViT-Tiny ----
    print(f"\n=== (B) ImageNet-pretrained ViT-Tiny → CLS ===")
    try:
        m_in = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0).to(DEVICE)
        results['imagenet_vit_tiny'] = jac_spectrum(m_in, pixels, 'INET')
        del m_in; torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ImageNet load failed: {e}")
        results['imagenet_vit_tiny'] = {'error': str(e)}

    # ---- (C) LeWM-trained encoder ----
    print(f"\n=== (C) LeWM JEPA-trained encoder → CLS ===")
    try:
        m_lewm = load_lewm_encoder()
        results['lewm_jepa_baseline_seed0'] = jac_spectrum(m_lewm, pixels, 'JEPA')
    except Exception as e:
        print(f"  LeWM load failed: {e}")
        import traceback; traceback.print_exc()
        results['lewm_jepa_baseline_seed0'] = {'error': str(e)}

    out = Path('/workspace/le-wm/refine-logs/c6_vit_jacobian_control.json')
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")
    print("\n=== SUMMARY ===")
    for k, v in results.items():
        if 'error' in v:
            print(f"  {k}: ERROR {v['error']}")
        else:
            print(f"  {k}: σ₁={v['mean_spectral_norm']:.3f}  SR={v['mean_stable_rank']:.3f}  "
                  f"ER={v['mean_effective_rank']:.3f}  F²={v['mean_frobenius_sq']:.1f}")


if __name__ == '__main__':
    main()
