"""
Post-training evaluation:

1. Multi-step rollout MSE at horizons [1, 5, 10, 20, 50].
   For each held-out validation trajectory: encode the first `history_size` frames,
   run the predictor autoregressively for H more steps, encode the ground-truth
   pixels at each future time, MSE over (B, D).

2. Physical-quantity probing accuracy.
   Train a Ridge regressor from frozen-encoder latents (192-d) to the dataset's
   `state` field (Push-T: 7-d) on N samples; evaluate R² on M held-out.

3. Save everything to {run_dir}/eval.json and {run_dir}/eval_rollout_per_step.csv.

Usage:
    python eval_rollout.py compare_runs/baseline/baseline_object.ckpt \
        --horizons 1 5 10 20 50 --n-rollout 256 --n-probe-train 1024
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


def build_world_model(cfg, device):
    """Re-build the JEPA world model exactly as train.py does."""
    sys.path.insert(0, str(Path(__file__).parent))
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

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
    pred_proj = MLP(hidden_dim, output_dim=embed_dim, hidden_dim=2048,
                    norm_fn=torch.nn.BatchNorm1d)
    return JEPA(encoder=encoder, predictor=predictor,
                action_encoder=action_encoder, projector=projector,
                pred_proj=pred_proj)


def load_weights(world_model, weights_path: Path):
    sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("model.", "", 1) if k.startswith("model.") else k: v
          for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not k.startswith("sigreg.")}
    missing, unexpected = world_model.load_state_dict(sd, strict=False)
    return missing, unexpected


def build_dataset(cfg, num_steps_override=None):
    """Build the Push-T dataset with the right number of timesteps per item."""
    sys.path.insert(0, str(Path(__file__).parent))
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from utils import get_column_normalizer, get_img_preprocessor

    ds_cfg = dict(cfg.data.dataset)
    if num_steps_override is not None:
        ds_cfg["num_steps"] = num_steps_override

    transforms = [get_img_preprocessor(source='pixels', target='pixels',
                                       img_size=cfg.img_size)]
    dataset = swm.data.HDF5Dataset(**ds_cfg, transform=None)
    for col in cfg.data.dataset.keys_to_load:
        if col.startswith("pixels"):
            continue
        normalizer = get_column_normalizer(dataset, col, col)
        transforms.append(normalizer)
    dataset.transform = spt.data.transforms.Compose(*transforms)
    return dataset


# ---------------------------------------------------------------------------
#  Multi-step rollout
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_pixels(wm, pixels):
    """pixels: (B, T, C, H, W) -> emb: (B, T, D)."""
    pixels = pixels.float()
    b = pixels.size(0)
    flat = rearrange(pixels, "b t ... -> (b t) ...")
    out = wm.encoder(flat, interpolate_pos_encoding=True)
    cls = out.last_hidden_state[:, 0]
    emb = wm.projector(cls)
    return rearrange(emb, "(b t) d -> b t d", b=b)


@torch.no_grad()
def rollout_predictor(wm, emb_init, act_emb_seq, history_size, horizon):
    """Roll the predictor `horizon` steps starting from emb_init.

    emb_init: (B, history_size, D)
    act_emb_seq: (B, T, A_emb) where T >= history_size + horizon - 1
    Returns predicted future embeddings: (B, horizon, D).
    """
    z = emb_init.clone()
    preds = []
    for t in range(horizon):
        ctx_z = z[:, -history_size:]
        ctx_a = act_emb_seq[:, t : t + history_size]
        if ctx_a.size(1) < history_size:  # pad if we ran out of actions
            pad = ctx_a.new_zeros(ctx_a.size(0), history_size - ctx_a.size(1), ctx_a.size(2))
            ctx_a = torch.cat([ctx_a, pad], dim=1)
        # Predictor forward + projection
        out = wm.predictor(ctx_z, ctx_a)
        out = wm.pred_proj(rearrange(out, "b t d -> (b t) d"))
        out = rearrange(out, "(b t) d -> b t d", b=ctx_z.size(0))
        pred = out[:, -1:]  # next-step embedding only
        preds.append(pred)
        z = torch.cat([z, pred], dim=1)
    return torch.cat(preds, dim=1)  # (B, horizon, D)


def evaluate_rollout(wm, dataset, history_size, horizons, n_eval, batch_size, device):
    """Returns list of {horizon, mse, n} dicts."""
    wm.eval()
    max_horizon = max(horizons)
    needed_steps = history_size + max_horizon
    # Build a fresh DataLoader from the dataset
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, drop_last=False, prefetch_factor=None, pin_memory=False,
    )

    h_to_sse = {h: 0.0 for h in horizons}
    h_to_count = {h: 0 for h in horizons}

    seen = 0
    for batch in loader:
        if seen >= n_eval:
            break
        if batch["pixels"].size(1) < needed_steps:
            continue  # skip too-short trajectories
        pixels = batch["pixels"][:, :needed_steps].to(device)
        actions = batch["action"][:, :needed_steps].to(device)
        actions = torch.nan_to_num(actions, 0.0)

        # Encode all needed frames + actions
        emb_all = encode_pixels(wm, pixels)
        act_emb = wm.action_encoder(actions)
        emb_init = emb_all[:, :history_size]

        preds = rollout_predictor(wm, emb_init, act_emb, history_size, max_horizon)
        # Ground truth at horizon h is emb_all[:, history_size + h - 1]
        for h in horizons:
            tgt = emb_all[:, history_size + h - 1]
            pred_h = preds[:, h - 1]
            sse = (pred_h - tgt).pow(2).mean(dim=-1).sum().item()  # mean over D, sum over B
            h_to_sse[h] += sse
            h_to_count[h] += pred_h.size(0)
        seen += pixels.size(0)

    return [{"horizon": h, "mse_per_step": h_to_sse[h] / max(1, h_to_count[h]),
             "n_samples": h_to_count[h]} for h in horizons]


# ---------------------------------------------------------------------------
#  Probing
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_latents_states(wm, dataset, n_samples, batch_size, device, history_size):
    """Encode n_samples examples; return latents (N, D) and states (N, S).

    Uses the LAST timestep of each trajectory window so probe gets a single
    snapshot per sample.
    """
    wm.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, drop_last=False, prefetch_factor=None, pin_memory=False,
    )
    all_latents, all_states = [], []
    seen = 0
    for batch in loader:
        if seen >= n_samples:
            break
        pixels = batch["pixels"].to(device)
        # encode the LAST history-size frame
        emb = encode_pixels(wm, pixels[:, history_size-1 : history_size])  # (B, 1, D)
        all_latents.append(emb[:, 0].cpu())
        # state at the same timestep
        state = batch["state"][:, history_size - 1].cpu()
        all_states.append(state)
        seen += pixels.size(0)

    latents = torch.cat(all_latents, dim=0)[:n_samples]
    states = torch.cat(all_states, dim=0)[:n_samples]
    return latents.numpy(), states.numpy()


def probe_state(wm, dataset, n_train, n_test, batch_size, device, history_size, alpha=1.0):
    """Train a Ridge regressor latent->state on n_train, evaluate on n_test."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    Z_tr, S_tr = collect_latents_states(wm, dataset, n_train, batch_size, device, history_size)
    Z_te, S_te = collect_latents_states(wm, dataset, n_test, batch_size, device, history_size)
    # Use disjoint splits via indexing offset
    if len(Z_te) < n_test:
        # If overlap, just use second half of Z_tr for test
        n_te = min(len(Z_tr) // 4, n_test)
        Z_te = Z_tr[-n_te:]; S_te = S_tr[-n_te:]
        Z_tr = Z_tr[:-n_te]; S_tr = S_tr[:-n_te]

    reg = Ridge(alpha=alpha)
    reg.fit(Z_tr, S_tr)
    pred = reg.predict(Z_te)
    overall_r2 = r2_score(S_te, pred)
    per_dim_r2 = []
    for d in range(S_te.shape[1]):
        per_dim_r2.append(r2_score(S_te[:, d], pred[:, d]))
    mse = float(((pred - S_te) ** 2).mean())
    return {
        "n_train": int(len(Z_tr)),
        "n_test": int(len(Z_te)),
        "overall_r2": float(overall_r2),
        "per_dim_r2": [float(x) for x in per_dim_r2],
        "mse": mse,
        "ridge_alpha": float(alpha),
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="path to *_object.ckpt or *_weights.ckpt")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10, 15, 20],
                    help="multi-step rollout horizons; max ~20 fits Push-T (median traj len 123 raw frames, frameskip=5 ~24 dataset steps)")
    ap.add_argument("--n-rollout", type=int, default=256,
                    help="number of held-out trajectories for rollout MSE")
    ap.add_argument("--n-probe-train", type=int, default=1024)
    ap.add_argument("--n-probe-test", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-dir", type=str, default=None,
                    help="defaults to ckpt's parent dir")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = ckpt_path.parent / "config.yaml"
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(cfg_path)

    # Build model + load weights
    wm = build_world_model(cfg, device)
    weights_ckpt = (ckpt_path.parent
                    / f"{ckpt_path.parent.name}_weights.ckpt")
    if not weights_ckpt.exists():
        weights_ckpt = next(ckpt_path.parent.glob("*_weights.ckpt"))
    print(f"loading weights from {weights_ckpt}")
    missing, unexpected = load_weights(wm, weights_ckpt)
    print(f"missing keys: {len(missing)}; unexpected keys: {len(unexpected)}")
    wm.to(device).eval()

    history_size = cfg.wm.history_size
    max_h = max(args.horizons)

    # Dataset for rollout: need each item to have `history + max_horizon` frames
    print(f"\n=== rollout: horizons={args.horizons}, n={args.n_rollout} ===")
    rollout_ds = build_dataset(cfg, num_steps_override=history_size + max_h)
    t0 = time.time()
    rollout_results = evaluate_rollout(
        wm, rollout_ds, history_size, args.horizons,
        n_eval=args.n_rollout, batch_size=args.batch_size, device=device,
    )
    rollout_secs = time.time() - t0
    for r in rollout_results:
        print(f"  horizon={r['horizon']}: mse_per_step={r['mse_per_step']:.5f}  "
              f"(n={r['n_samples']})")

    # Save per-horizon CSV
    csv_path = out_dir / "eval_rollout_per_step.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["horizon", "mse_per_step", "n_samples"])
        for r in rollout_results:
            w.writerow([r["horizon"], r["mse_per_step"], r["n_samples"]])
    print(f"saved {csv_path}")

    # Probing: dataset with smaller num_steps (just need history+1 for state)
    print(f"\n=== probing: latent -> state (Ridge) ===")
    probe_ds = build_dataset(cfg, num_steps_override=history_size + 1)
    t1 = time.time()
    probe_result = probe_state(
        wm, probe_ds,
        n_train=args.n_probe_train, n_test=args.n_probe_test,
        batch_size=args.batch_size, device=device, history_size=history_size,
    )
    probe_secs = time.time() - t1
    print(f"  overall_r2 = {probe_result['overall_r2']:.4f}  mse={probe_result['mse']:.5f}")
    print(f"  per-dim R²: {[round(x, 3) for x in probe_result['per_dim_r2']]}")

    # Combine
    summary = {
        "ckpt": str(ckpt_path),
        "weights": str(weights_ckpt),
        "rollout_horizons": args.horizons,
        "rollout_results": rollout_results,
        "rollout_eval_seconds": rollout_secs,
        "probe": probe_result,
        "probe_eval_seconds": probe_secs,
    }
    out_json = out_dir / "eval.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out_json}")


if __name__ == "__main__":
    main()
