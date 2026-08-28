"""Tier-1: dump predicted + encoded latent trajectories for all variants.

For each variant ckpt:
  - Load full LeWM (encoder + projector + pred_proj + predictor + action_encoder)
  - Encode first H_HIST=3 frames of N validation trajectories
  - Roll out predictor for HORIZON=20 steps -> predicted_z (N, HORIZON, 192)
  - Encode ground-truth future frames -> encoded_z (N, HORIZON+1, 192)  [includes t=0]
  - Save Push-T state at corresponding timesteps -> state_gt (N, HORIZON+1, 7)
  - Save action sequence too -> actions (N, HORIZON+H_HIST, 2)

Output: /workspace/le-wm/refine-logs/tier1_traj/{variant}_traj.npz
"""
import os
import sys
import time
import json
from pathlib import Path

import hdf5plugin  # noqa
import h5py
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/workspace/le-wm")
# We don't import jepa.py directly because building the full LeWM needs hydra; we
# instead load the *_epoch_1_object.ckpt which is the pickled spt.Module that has
# the entire world_model + sigreg + optimizers + callbacks.

H5     = "/workspace/stablewm_home/pusht_expert_train.h5"
H_HIST = 3                       # history_size (matches v3 config)
HORIZON = 20                     # rollout length
N_TRAJ = 80                      # validation trajectories to dump
FRAMESKIP = 5                    # matches Push-T train config
SEED = 0                         # which val trajectories to draw

VARIANTS = ["tier1_baseline", "tier1_uvlowr", "tier1_randdiff"]
HOME = Path("/workspace/stablewm_home")
OUT  = Path("/workspace/le-wm/refine-logs/tier1_traj")
OUT.mkdir(parents=True, exist_ok=True)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_world_model(name):
    """Load the spt.Module pickle and extract the world_model."""
    ckpt_path = HOME / name / f"{name}_epoch_1_object.ckpt"
    assert ckpt_path.exists(), f"missing ckpt: {ckpt_path}"
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # spt.Module subclasses pl.LightningModule; the world_model is at obj.model
    if hasattr(obj, "model"):
        wm = obj.model
    else:
        # fallback: assume obj itself is the world model (defensive)
        wm = obj
    wm.eval()
    wm.to(DEV)
    for p in wm.parameters():
        p.requires_grad_(False)
    return wm


@torch.no_grad()
def encode_pixels(wm, pixels):
    """pixels: (B, T, 3, 224, 224). returns (B, T, D) post-projector."""
    B, T = pixels.shape[:2]
    pix = pixels.reshape(B * T, *pixels.shape[2:])
    o = wm.encoder(pix, interpolate_pos_encoding=True)
    cls = o.last_hidden_state[:, 0]                 # (B*T, 192)
    emb = wm.projector(cls)                         # (B*T, 192)
    return emb.reshape(B, T, -1)


@torch.no_grad()
def rollout(wm, init_emb, action_seq):
    """init_emb: (B, H_HIST, 192).  action_seq: (B, H_HIST + HORIZON, A_dim).

    Returns z_pred: (B, HORIZON, 192) — the predictor's autoregressive outputs.
    """
    z = init_emb.clone()                            # (B, H_HIST, D)
    act_emb = wm.action_encoder(action_seq)         # (B, H_HIST + HORIZON, A_emb)
    preds = []
    for t in range(HORIZON):
        ctx_z = z[:, -H_HIST:]                      # (B, H_HIST, D)
        ctx_a = act_emb[:, t : t + H_HIST]          # (B, H_HIST, A_emb)
        pred = wm.predict(ctx_z, ctx_a)             # (B, H_HIST, D)
        next_z = pred[:, -1:]                       # (B, 1, D)
        z = torch.cat([z, next_z], dim=1)
        preds.append(next_z)
    z_pred = torch.cat(preds, dim=1)                # (B, HORIZON, D)
    return z_pred


def pick_validation_trajectories(rng, ep_len, ep_off, n_traj, min_len):
    """Pick N trajectories of sufficient length, last 10% of episodes (val-ish)."""
    n_eps = len(ep_len)
    val_start = int(n_eps * 0.9)
    candidates = [i for i in range(val_start, n_eps) if ep_len[i] >= min_len]
    chosen = rng.choice(candidates, size=min(n_traj, len(candidates)), replace=False)
    return sorted([int(x) for x in chosen])


def gather_trajectory_data(ep_indices, f, T_needed):
    """For each chosen episode, take the first T_needed*FRAMESKIP raw frames.

    Pixels + states are subsampled by FRAMESKIP (one per latent timestep).
    Actions are NOT subsampled — they get reshaped into FRAMESKIP-grouped
    "smoothed" actions of shape (T_needed, FRAMESKIP * action_dim_raw)
    to match the model's Embedder input_dim = frameskip * action_dim.

    Returns:
      pixels:  (N, T_needed, 3, 224, 224) float32 / 255
      actions: (N, T_needed, FRAMESKIP*A_raw=10) float32
      states:  (N, T_needed, 7) float32
    """
    ep_len = f["ep_len"][:]; ep_off = f["ep_offset"][:]
    span = T_needed * FRAMESKIP
    pix_arr, act_arr, state_arr = [], [], []
    for ep in ep_indices:
        o = int(ep_off[ep]); L = int(ep_len[ep])
        if span > L:
            continue
        pix = f["pixels"][o : o + span : FRAMESKIP][:T_needed]
        st  = f["state"][o : o + span : FRAMESKIP][:T_needed]
        act_raw = f["action"][o : o + span]                 # (span, A_raw=2)
        if pix.shape[0] < T_needed or act_raw.shape[0] < span:
            continue
        # reshape: (T_needed * FRAMESKIP, 2) -> (T_needed, FRAMESKIP*2 = 10)
        act = act_raw.reshape(T_needed, -1)
        pix_arr.append(pix); state_arr.append(st); act_arr.append(act)
    pix_arr  = np.stack(pix_arr,  axis=0).astype(np.float32) / 255.0
    pix_arr  = pix_arr.transpose(0, 1, 4, 2, 3)             # (N, T, C, H, W)
    act_arr  = np.stack(act_arr,  axis=0).astype(np.float32)
    state_arr = np.stack(state_arr, axis=0).astype(np.float32)
    return pix_arr, act_arr, state_arr


def main():
    rng = np.random.default_rng(SEED)
    T_needed = H_HIST + HORIZON                                 # 3 + 20 = 23 timesteps

    with h5py.File(H5, "r") as f:
        ep_len = f["ep_len"][:]
        ep_off = f["ep_offset"][:]
        min_subsampled_len = T_needed * FRAMESKIP
        idx = pick_validation_trajectories(rng, ep_len, ep_off,
                                           n_traj=N_TRAJ, min_len=min_subsampled_len)
        print(f"[dump] picked {len(idx)} validation trajectories "
              f"(min subsampled len {min_subsampled_len}, FRAMESKIP={FRAMESKIP})")
        pixels, actions, states = gather_trajectory_data(idx, f, T_needed)
        print(f"[dump] pixels {pixels.shape}  actions {actions.shape}  states {states.shape}")

    pixels_t = torch.from_numpy(pixels).to(DEV)
    actions_t = torch.from_numpy(actions).to(DEV)

    # We share the encoded GT across variants because encoder is frozen + identical.
    # But predicted differs per variant.
    for variant in VARIANTS:
        out_path = OUT / f"{variant}_traj.npz"
        if out_path.exists() and os.environ.get("FORCE") != "1":
            print(f"[dump] SKIP {variant} (exists). Set FORCE=1 to overwrite.")
            continue
        t0 = time.time()
        print(f"[dump] === {variant} ===")
        wm = load_world_model(variant)

        # process in batches to keep memory low
        B = 8
        z_pred_chunks = []
        z_enc_chunks  = []
        for s in range(0, pixels_t.size(0), B):
            e = min(s + B, pixels_t.size(0))
            # encoded GT for all frames t = 0..H_HIST + HORIZON - 1
            z_enc = encode_pixels(wm, pixels_t[s:e])      # (B, T_needed, 192)
            init_emb = z_enc[:, :H_HIST]                  # (B, H_HIST, 192)
            z_pred = rollout(wm, init_emb, actions_t[s:e])  # (B, HORIZON, 192)
            z_pred_chunks.append(z_pred.cpu().numpy())
            z_enc_chunks.append(z_enc.cpu().numpy())
        predicted_z = np.concatenate(z_pred_chunks, axis=0)        # (N, HORIZON, 192)
        encoded_z   = np.concatenate(z_enc_chunks,  axis=0)        # (N, T_needed, 192)
        dt = time.time() - t0
        print(f"[dump] {variant}: predicted_z {predicted_z.shape}  encoded_z {encoded_z.shape}  "
              f"in {dt:.1f}s")

        # For metrics that need predicted vs GT at same timesteps:
        #   predicted_z[:, 0..H-1]  corresponds to encoded_z[:, H_HIST..H_HIST+H-1]
        # So we save encoded_z aligned as:
        #   encoded_aligned = encoded_z[:, H_HIST-1 : H_HIST + HORIZON]
        #   shape (N, HORIZON+1, 192)  — includes the last "init" frame as t=0 anchor
        encoded_aligned = encoded_z[:, H_HIST - 1 : H_HIST + HORIZON]
        # state aligned similarly (last init frame + HORIZON future)
        state_aligned = states[:, H_HIST - 1 : H_HIST + HORIZON]
        # For predicted, prepend the same t=0 anchor (last init latent) so it has shape (N, H+1, D)
        anchor = encoded_z[:, H_HIST - 1 : H_HIST]                # (N, 1, 192)
        predicted_aligned = np.concatenate([anchor, predicted_z], axis=1)  # (N, H+1, 192)

        np.savez_compressed(
            out_path,
            predicted_z=predicted_aligned,
            encoded_z=encoded_aligned,
            state_gt=state_aligned,
            actions=actions[:, H_HIST - 1 : H_HIST + HORIZON],
            ep_idx=np.array(idx),
            metadata=json.dumps({
                "variant": variant,
                "n_traj": int(predicted_aligned.shape[0]),
                "horizon": HORIZON,
                "h_hist": H_HIST,
                "frameskip": FRAMESKIP,
                "latent_dim": int(predicted_aligned.shape[2]),
                "state_dim": int(state_aligned.shape[2]),
            }),
        )
        print(f"[dump] saved -> {out_path}")
        del wm
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
