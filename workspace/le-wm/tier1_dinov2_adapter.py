"""B10: DINOv2 cross-pretraining encoder swap for cross-encoder C_dyn check.

Replaces v3's LeWM-trained ViT-Tiny encoder + projector with:
  DINOv2 ViT-S/14 (384-dim, frozen) → fixed linear adapter (384→192, frozen) → predictor

Then trains the predictor (3 variants × 8K) and computes the same metric panel as B1/B4.

Usage:
  # First: build & save the adapter
  python tier1_dinov2_adapter.py --build-adapter \
        --out /workspace/stablewm_home/v5_dino_adapter.pt \
        --init random --seed 42

  # Then: train one variant
  python tier1_dinov2_adapter.py --train \
        --adapter /workspace/stablewm_home/v5_dino_adapter.pt \
        --name v5_dino_baseline --steps 8000 --batch 16 --seed 0

  python tier1_dinov2_adapter.py --train --adapter ... \
        --name v5_dino_uvlowr +predictor.ffn_rank=4

  python tier1_dinov2_adapter.py --train --adapter ... \
        --name v5_dino_randdiff +loss.rand_diff.weight=0.01 +loss.rand_diff.eps=0.05 +loss.rand_diff.num_dirs=2
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import hdf5plugin  # noqa
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/workspace/le-wm")
from module import ARPredictor, Embedder

H5 = "/workspace/stablewm_home/pusht_expert_train.h5"
HOME = Path("/workspace/stablewm_home")

# DINOv2 expects ImageNet normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────
#  Adapter construction
# ─────────────────────────────────────────────────────────────────────

def build_adapter(dim_in=384, dim_out=192, init="random", seed=42, n_pca_frames=1000):
    """Build a fixed 384→192 projection. Returns torch.Tensor."""
    g = torch.Generator().manual_seed(seed)
    if init == "random":
        W = torch.randn(dim_in, dim_out, generator=g)
        W = W / W.norm(dim=0, keepdim=True)
        return W
    elif init == "pca":
        # fit PCA on n_pca_frames DINOv2 patch-mean features
        print(f"[adapter] fitting PCA on {n_pca_frames} val frames")
        dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                                pretrained=True, source="github").to(DEV).eval()
        with h5py.File(H5, "r") as f:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(f["pixels"]), n_pca_frames, replace=False)
            idx = sorted(int(i) for i in idx)
            pix = f["pixels"][idx]
        pix = torch.from_numpy(pix).float().permute(0, 3, 1, 2) / 255.0
        pix = (pix - torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)) \
              / torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        feats = []
        with torch.no_grad():
            for i in range(0, len(pix), 32):
                batch = pix[i:i+32].to(DEV)
                out = dinov2.forward_features(batch)
                pooled = out["x_norm_patchtokens"].mean(dim=1)
                feats.append(pooled.cpu())
        feats = torch.cat(feats, dim=0).numpy()      # (N, 384)
        # PCA
        feats_c = feats - feats.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(feats_c, full_matrices=False)
        W = torch.from_numpy(Vt[:dim_out].T).float()  # (384, 192)
        return W
    else:
        raise ValueError(f"unknown init: {init}")


# ─────────────────────────────────────────────────────────────────────
#  DINOv2-encoder JEPA wrapper
# ─────────────────────────────────────────────────────────────────────

class DINOv2Encoder(nn.Module):
    """Frozen DINOv2 ViT-S/14 + frozen 384→192 adapter."""
    def __init__(self, W_adapter):
        super().__init__()
        self.dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                                     pretrained=True, source="github")
        self.dinov2.eval()
        for p in self.dinov2.parameters():
            p.requires_grad_(False)
        self.register_buffer("W_adapter", W_adapter)
        self.register_buffer("imagenet_mean",
                             torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std",
                             torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, pixels):
        """pixels: (B, 3, 224, 224) in [0, 1].  Returns (B, 192)."""
        x = (pixels - self.imagenet_mean) / self.imagenet_std
        out = self.dinov2.forward_features(x)
        patch_tokens = out["x_norm_patchtokens"]      # (B, 256, 384)
        pooled = patch_tokens.mean(dim=1)             # (B, 384)
        return pooled @ self.W_adapter                # (B, 192)


class DINOv2JEPA(nn.Module):
    """LeWM-shaped JEPA but with DINOv2-adapter encoder + skipped projector."""
    def __init__(self, encoder, predictor, action_encoder):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        # projector / pred_proj effectively identity since adapter already lands in 192-d

    def encode(self, info):
        pixels = info["pixels"].float()
        B = pixels.size(0)
        flat = pixels.reshape(B * pixels.size(1), *pixels.shape[2:])
        emb = self.encoder(flat)                      # (B*T, 192)
        info["emb"] = emb.reshape(B, -1, 192)
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def predict(self, emb, act_emb):
        preds = self.predictor(emb, act_emb)          # (B, T, 192)
        return preds  # NO pred_proj — already in matched space


# ─────────────────────────────────────────────────────────────────────
#  Data loader (subset of train.py logic)
# ─────────────────────────────────────────────────────────────────────

def make_dataloader(batch_size, seed):
    """Hand-rolled Push-T loader (avoid spt.data complications)."""
    f = h5py.File(H5, "r")
    ep_len = f["ep_len"][:]
    ep_off = f["ep_offset"][:]
    H_HIST = 3
    NUM_PREDS = 1
    NUM_STEPS = H_HIST + NUM_PREDS  # 4
    FRAMESKIP = 5

    rng = np.random.default_rng(seed)
    # collect valid start indices (start within episode, with enough length)
    starts = []
    for ep_i in range(len(ep_len)):
        L = int(ep_len[ep_i]); off = int(ep_off[ep_i])
        n_starts = max(0, L - NUM_STEPS * FRAMESKIP + 1)
        starts.extend([(off + s) for s in range(0, n_starts, FRAMESKIP)])
    starts = np.array(starts)
    # 90/10 train/val
    perm = rng.permutation(len(starts))
    n_train = int(len(starts) * 0.9)
    train_starts = starts[perm[:n_train]]

    def sample_batch():
        idx = rng.choice(train_starts, batch_size, replace=False)
        pix_arr, act_arr = [], []
        for s in idx:
            span = NUM_STEPS * FRAMESKIP
            pix = f["pixels"][s : s + span : FRAMESKIP][:NUM_STEPS]
            act = f["action"][s : s + span].reshape(NUM_STEPS, -1)
            pix_arr.append(pix); act_arr.append(act)
        pix = np.stack(pix_arr).astype(np.float32) / 255.0     # (B, 4, 224, 224, 3)
        pix = pix.transpose(0, 1, 4, 2, 3)                      # (B, 4, 3, 224, 224)
        act = np.stack(act_arr).astype(np.float32)              # (B, 4, 10)
        return (torch.from_numpy(pix).to(DEV),
                torch.from_numpy(act).to(DEV))

    return sample_batch, f


# ─────────────────────────────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────────────────────────────

def train_dinov2_jepa(args, extra_overrides):
    """Train one variant of the DINOv2 JEPA."""
    print(f"[B10] === {args.name}  steps={args.steps}  seed={args.seed} ===")
    out_dir = HOME / args.name
    out_dir.mkdir(exist_ok=True)
    if (out_dir / f"{args.name}_epoch_1_object.ckpt").exists():
        print(f"[B10] SKIP {args.name} (ckpt exists)")
        return

    # parse extra overrides for ffn_rank, rand_diff
    ffn_rank = None
    rd_w, rd_eps, rd_dirs = 0.0, 0.05, 2
    for o in extra_overrides:
        if "ffn_rank" in o:
            ffn_rank = int(o.split("=")[1])
        elif "rand_diff.weight" in o:
            rd_w = float(o.split("=")[1])
        elif "rand_diff.eps" in o:
            rd_eps = float(o.split("=")[1])
        elif "rand_diff.num_dirs" in o:
            rd_dirs = int(o.split("=")[1])

    # Build model
    W = torch.load(args.adapter, weights_only=True)
    encoder = DINOv2Encoder(W).to(DEV)
    predictor = ARPredictor(
        num_frames=3, input_dim=192, hidden_dim=192, output_dim=192,
        depth=6, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1,
        emb_dropout=0.0, ffn_rank=ffn_rank,
    ).to(DEV)
    effective_act_dim = 5 * 2
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=192).to(DEV)
    wm = DINOv2JEPA(encoder, predictor, action_encoder)
    wm.train()

    # Optimizer (only predictor + action_encoder)
    opt_params = list(predictor.parameters()) + list(action_encoder.parameters())
    optimizer = torch.optim.AdamW(opt_params, lr=5e-5, weight_decay=1e-3, betas=(0.9, 0.95))

    # Data
    sample_batch, hf = make_dataloader(args.batch, args.seed)

    # Train loop
    history = {"step": [], "pred_loss": [], "rand_diff_loss": []}
    t0 = time.time()
    for step in range(args.steps):
        pixels, action = sample_batch()
        action = torch.nan_to_num(action, 0.0)
        info = wm.encode({"pixels": pixels, "action": action})
        emb = info["emb"]; act_emb = info["act_emb"]
        ctx_emb = emb[:, :3]; tgt_emb = emb[:, 1:]
        ctx_act = act_emb[:, :3]
        pred_emb = wm.predict(ctx_emb, ctx_act)
        pred_loss = F.mse_loss(pred_emb, tgt_emb)
        loss = pred_loss

        if rd_w > 0.0:
            ctx_d = ctx_emb.detach()
            base = wm.predict(ctx_d, ctx_act)
            rd_total = 0.0
            for _ in range(rd_dirs):
                v = torch.randn_like(ctx_d)
                n = v.flatten(1).norm(dim=-1).clamp_min(1e-8).view(-1, 1, 1)
                v = v / n
                pert = wm.predict(ctx_d + rd_eps * v, ctx_act)
                rd_total = rd_total + (pert - base).flatten(1).norm(dim=-1).mean() / rd_eps
            rd_total = rd_total / rd_dirs
            history["rand_diff_loss"].append(float(rd_total.item()))
            loss = loss + rd_w * rd_total
        else:
            history["rand_diff_loss"].append(0.0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
        optimizer.step()

        history["step"].append(step)
        history["pred_loss"].append(float(pred_loss.item()))

        if (step + 1) % 1000 == 0:
            print(f"[B10] step={step+1}  pred_loss={pred_loss.item():.4f}  "
                  f"rd_loss={history['rand_diff_loss'][-1]:.4f}  "
                  f"elapsed={time.time()-t0:.1f}s")

    # Compute final-step Jacobian (small sample)
    print("[B10] computing predictor Jacobian (final step)")
    wm.eval()
    sigma_metrics = compute_predictor_jacobian(wm, sample_batch, n_samples=4)

    # Save
    state = {
        "encoder_W_adapter": W.cpu(),
        "predictor_state_dict": predictor.state_dict(),
        "action_encoder_state_dict": action_encoder.state_dict(),
        "ffn_rank": ffn_rank,
        "rd_w": rd_w, "rd_eps": rd_eps, "rd_dirs": rd_dirs,
        "history": history,
        "sigma_metrics": sigma_metrics,
        "args": vars(args),
    }
    torch.save(state, out_dir / f"{args.name}_epoch_1_object.ckpt")
    json.dump({
        "predictor_params": sum(p.numel() for p in predictor.parameters()),
        "encoder_params": sum(p.numel() for p in encoder.dinov2.parameters()),
        "total_params": sum(p.numel() for p in wm.parameters()),
        "wall_clock_s": time.time() - t0,
        "global_step": args.steps,
        "final_pred_loss": history["pred_loss"][-1],
        "final_rand_diff_loss": history["rand_diff_loss"][-1] if rd_w > 0 else None,
        **sigma_metrics,
    }, open(out_dir / "engineering_metrics.json", "w"), indent=2)
    print(f"[B10] saved {out_dir / args.name}_epoch_1_object.ckpt")
    print(f"[B10] σ_metrics: {sigma_metrics}")


@torch.no_grad()
def compute_predictor_jacobian(wm, sample_batch, n_samples=4):
    """Approximate predictor Jacobian via small finite differences."""
    from torch.func import jacrev
    wm.eval()
    sigmas = []; F_sqs = []; top_svs_list = []
    for _ in range(n_samples):
        pixels, action = sample_batch()
        action = torch.nan_to_num(action, 0.0)
        info = wm.encode({"pixels": pixels[:1], "action": action[:1]})
        emb = info["emb"][:1, :3]; act_emb = info["act_emb"][:1, :3]
        # Jacobian of predict() wrt emb, flatten input/output
        def f(z): return wm.predict(z.reshape(1, 3, 192), act_emb).reshape(-1)
        z_in = emb.reshape(-1)
        J = jacrev(f)(z_in)                            # (out_dim, in_dim)
        # SVD
        U, S, V = torch.linalg.svd(J, full_matrices=False)
        sigmas.append(float(S[0].item()))
        F_sqs.append(float((S ** 2).sum().item()))
        top_svs_list.append(S[:32].cpu().tolist())
    sig = float(np.mean(sigmas))
    Fsq = float(np.mean(F_sqs))
    SR = Fsq / max(sig ** 2, 1e-8)
    return {
        "mean_spectral_norm": sig,
        "mean_frobenius_norm_sq": Fsq,
        "mean_stable_rank": SR,
        "top_svs_avg": np.mean(top_svs_list, axis=0).tolist(),
        "n_samples": n_samples,
    }


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build-adapter", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--out", type=str, help="output path for adapter")
    p.add_argument("--adapter", type=str, help="path to saved adapter W")
    p.add_argument("--init", type=str, default="random", choices=["random", "pca"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", type=str)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=16)
    args, extra = p.parse_known_args()

    if args.build_adapter:
        print(f"[B10] building adapter (init={args.init}, seed={args.seed})")
        W = build_adapter(384, 192, init=args.init, seed=args.seed)
        out = args.out or "/workspace/stablewm_home/v5_dino_adapter.pt"
        torch.save(W, out)
        print(f"[B10] adapter saved -> {out}  shape={tuple(W.shape)}")
        # also save a small sanity report
        norm = (W ** 2).sum(0).sqrt().mean().item()
        print(f"[B10] adapter column-norm mean = {norm:.4f}")
        return

    if args.train:
        assert args.adapter and Path(args.adapter).exists(), \
            f"adapter not found: {args.adapter}"
        assert args.name, "must pass --name"
        train_dinov2_jepa(args, extra)
        return

    print("Usage: --build-adapter ... | --train ...")


if __name__ == "__main__":
    main()
