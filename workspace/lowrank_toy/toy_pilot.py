"""
Toy pilot: low-rank predictor regularization on a synthetic low-rank dynamical system.

Compares 3 predictors on the SAME ARPredictor architecture (copied from le-wm/module.py):

  baseline   - standard ARPredictor (full rank everywhere)
  uv_lowr    - FFN linears replaced with explicit Linear(in,r) o Linear(r,out)
               + standard weight decay (variational nuclear norm via UV factorization)
  rand_diff  - full-rank predictor + randomized one-sided finite-difference
               nuclear norm penalty E_{v on sphere} ||p(z+eps v, c) - p(z, c)||_2 / eps
               (Scarvelis & Solomon NeurIPS 2024 style, scaled by lambda)

Synthetic data: latent dim d=64. True dynamics
  z_{t+1} = z_t + alpha * tanh(A z_t + B a_t)        with A = U V^T, rank(U)=rank(V)=4
This gives "intrinsic" dynamics dimension 4, embedded in d=64 - the same setup
TD-JEPA / Koopman intuition predicts low-rank predictors should suffice.

We compare:
  - training pred MSE
  - 1, 4, 8-step rollout MSE on held-out trajectories
  - peak GPU memory during training
  - wall-clock per step
  - (uv only) effective parameter count

Run: python toy_pilot.py
"""

import argparse
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ============================================================================
#  ARPredictor architecture (copied verbatim from le-wm/module.py, then patched)
# ============================================================================

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    """Standard FFN: dim -> hidden_dim -> dim."""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class LowRankFeedForward(nn.Module):
    """FFN with both linears factored as low-rank: Linear(in,r) o Linear(r,out).

    Variational nuclear-norm form: ||W||_* = min_{W=UV^T} (||U||_F^2 + ||V||_F^2) / 2.
    With ordinary weight decay on U,V (PyTorch-default L2 = ||U||_F^2 + ||V||_F^2),
    SGD/AdamW solves the variational problem -> low-rank weight, no explicit nuclear
    norm computation needed.
    """

    def __init__(self, dim, hidden_dim, rank, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.rank = rank
        # W1: dim -> hidden_dim factored as dim -> rank -> hidden_dim
        # W2: hidden_dim -> dim factored as hidden_dim -> rank -> dim
        self.norm = nn.LayerNorm(dim)
        self.w1_a = nn.Linear(dim, rank, bias=False)
        self.w1_b = nn.Linear(rank, hidden_dim, bias=True)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.w2_a = nn.Linear(hidden_dim, rank, bias=False)
        self.w2_b = nn.Linear(rank, dim, bias=True)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)
        x = self.w1_b(self.w1_a(x))
        x = self.act(x)
        x = self.drop1(x)
        x = self.w2_b(self.w2_a(x))
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, causal=True):
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Same as le-wm with optional low-rank FFN."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0,
                 ffn_rank=None):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        if ffn_rank is None:
            self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        else:
            self.mlp = LowRankFeedForward(dim, mlp_dim, ffn_rank, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class ARPredictor(nn.Module):
    def __init__(self, *, num_frames, depth, heads, mlp_dim, dim, dim_head=64,
                 dropout=0.0, ffn_rank=None):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim) * 0.02)
        self.layers = nn.ModuleList([
            ConditionalBlock(dim, heads, dim_head, mlp_dim, dropout, ffn_rank=ffn_rank)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, c):
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        for blk in self.layers:
            x = blk(x, c)
        return self.norm(x)


# ============================================================================
#  Synthetic low-rank dynamics dataset
# ============================================================================

def make_true_dynamics(d, rank, action_dim, device, seed=0):
    """A residual tanh dynamical system whose 'force' lives in a rank-r subspace."""
    g = torch.Generator(device=device).manual_seed(seed)
    # rank-r factors for the state-side matrix
    U = torch.randn(d, rank, generator=g, device=device) / math.sqrt(d)
    V = torch.randn(d, rank, generator=g, device=device) / math.sqrt(rank)
    # action coupling
    B = torch.randn(d, action_dim, generator=g, device=device) / math.sqrt(action_dim)
    return U, V, B


def rollout_true(U, V, B, z0, actions, alpha=0.3):
    """z0: (N, d), actions: (N, T, action_dim) -> (N, T+1, d)."""
    Z = [z0]
    z = z0
    for t in range(actions.size(1)):
        force = torch.tanh(z @ V @ U.t() + actions[:, t] @ B.t())
        z = z + alpha * force
        Z.append(z)
    return torch.stack(Z, dim=1)


def make_dataset(N, T, d, action_dim, U, V, B, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    z0 = torch.randn(N, d, generator=g, device=device) * 0.5
    actions = torch.randn(N, T, action_dim, generator=g, device=device) * 0.5
    Z = rollout_true(U, V, B, z0, actions)
    return Z, actions  # Z: (N, T+1, d); actions: (N, T, ad)


# ============================================================================
#  Predictor wrapper: history -> next-step (matches LeWM lejepa_forward)
# ============================================================================

class WorldModelToy(nn.Module):
    """Predictor that consumes (B, T_ctx, d) latents + actions and predicts (B, T_ctx, d)."""

    def __init__(self, d, action_dim, num_frames, depth=3, heads=4, dim_head=32,
                 mlp_dim=256, dropout=0.0, ffn_rank=None):
        super().__init__()
        self.action_enc = nn.Sequential(
            nn.Linear(action_dim, d), nn.SiLU(), nn.Linear(d, d),
        )
        self.predictor = ARPredictor(
            num_frames=num_frames, depth=depth, heads=heads, mlp_dim=mlp_dim,
            dim=d, dim_head=dim_head, dropout=dropout, ffn_rank=ffn_rank,
        )

    def predict_one_step(self, ctx_z, ctx_a):
        """ctx_z: (B, T, d); ctx_a: (B, T, ad). Returns (B, T, d) — predicted next."""
        c = self.action_enc(ctx_a)
        return self.predictor(ctx_z, c)

    def rollout(self, init_z, actions, history_size):
        """
        init_z: (B, history_size, d), actions: (B, T_total, ad). Returns (B, T_total, d) preds.
        """
        B = init_z.size(0)
        T_total = actions.size(1)
        z = init_z.clone()
        preds = []
        # We have history_size tokens already; predict next (T_total - history_size + 1) steps
        # Start from t = history_size - 1 (predict z[history_size]) and roll.
        for t in range(T_total - history_size + 1):
            ctx_z = z[:, -history_size:]
            ctx_a = actions[:, t : t + history_size]
            pred = self.predict_one_step(ctx_z, ctx_a)[:, -1:]  # (B, 1, d)
            preds.append(pred)
            z = torch.cat([z, pred], dim=1)
        return torch.cat(preds, dim=1)  # (B, T_total - history_size + 1, d)


# ============================================================================
#  Regularizers
# ============================================================================

def randomized_diff_nuclear(model, ctx_z, ctx_a, eps=0.05, num_dirs=4):
    """Nuclear-norm-of-Jacobian surrogate via randomized one-sided finite difference.

    For predictor p: z -> p(z, c), the Jacobian nuclear norm satisfies
        ||J||_* = max over orth. sets of directions sum ||J v_i||_2.
    A simple one-sample stochastic surrogate is
        E_{v on sphere} ||p(z + eps v, c) - p(z, c)||_2 / eps,
    which we average over `num_dirs` independent v.
    Returns a scalar penalty (positive, lower = more low-rank).
    """
    base = model.predict_one_step(ctx_z, ctx_a)
    B = ctx_z.size(0)
    total = 0.0
    for _ in range(num_dirs):
        v = torch.randn_like(ctx_z)                                   # (B, T, d)
        norm = v.flatten(1).norm(dim=-1).clamp_min(1e-8).view(B, 1, 1)
        v = v / norm                                                  # unit Frobenius norm per batch
        perturbed = model.predict_one_step(ctx_z + eps * v, ctx_a)
        diff = (perturbed - base).flatten(1)                          # (B, T*d)
        total = total + diff.norm(dim=-1).mean() / eps                # scalar
    return total / num_dirs


# ============================================================================
#  Training one variant
# ============================================================================

@contextmanager
def cuda_mem_tracker(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    yield
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def train_variant(name, model, train_data, val_data, history_size, n_preds,
                  device, steps=1500, batch_size=128, lr=3e-4, wd=1e-3,
                  reg_weight=0.0, reg_kind="none", log_every=200):
    """Returns dict of metrics."""
    Z_tr, A_tr = train_data
    Z_va, A_va = val_data

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    train_losses = []
    reg_values = []
    step_times = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    model.train()
    N_tr = Z_tr.size(0)
    g = torch.Generator(device="cpu").manual_seed(123)

    t_total_start = time.time()
    for step in range(steps):
        idx = torch.randint(0, N_tr, (batch_size,), generator=g)
        Z_b = Z_tr[idx]
        A_b = A_tr[idx]
        ctx_z = Z_b[:, :history_size]
        ctx_a = A_b[:, :history_size]
        tgt_z = Z_b[:, n_preds : n_preds + history_size]

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        pred_z = model.predict_one_step(ctx_z, ctx_a)
        pred_loss = (pred_z - tgt_z).pow(2).mean()

        if reg_kind == "rand_diff" and reg_weight > 0:
            reg = randomized_diff_nuclear(model, ctx_z, ctx_a)
        else:
            reg = torch.tensor(0.0, device=device)

        loss = pred_loss + reg_weight * reg

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.time() - t0)

        train_losses.append(pred_loss.item())
        reg_values.append(reg.item() if torch.is_tensor(reg) else float(reg))

        if (step + 1) % log_every == 0 or step == 0:
            recent = sum(train_losses[-log_every:]) / max(1, len(train_losses[-log_every:]))
            print(f"  [{name}] step {step+1:5d}/{steps} | pred_mse={recent:.5f} "
                  f"| reg={reg_values[-1]:.4f} | step_ms={1000*sum(step_times[-log_every:])/max(1,len(step_times[-log_every:])):.1f}")

    t_total = time.time() - t_total_start

    # Rollout eval
    model.eval()
    with torch.no_grad():
        # k-step rollout over the entire val trajectory
        init_z = Z_va[:, :history_size]
        actions = A_va  # (N, T_total, ad); T_total is what we generated minus 1
        T_total = A_va.size(1)
        # The rollout helper returns predictions for steps history_size..T_total
        rollout_preds = model.rollout(init_z, actions, history_size)
        # Compare to ground truth Z_va[:, history_size:]
        true_future = Z_va[:, history_size:]
        # If shapes differ by 1 (off-by-one), align
        m = min(rollout_preds.size(1), true_future.size(1))
        rollout_preds = rollout_preds[:, :m]
        true_future = true_future[:, :m]
        # per-step MSE
        per_step_mse = (rollout_preds - true_future).pow(2).mean(dim=(0, 2)).cpu().tolist()

    peak_mb = (torch.cuda.max_memory_allocated(device) / 1024 / 1024
               if device.type == "cuda" else 0.0)

    return {
        "name": name,
        "params": count_params(model),
        "final_train_pred_mse": sum(train_losses[-50:]) / 50,
        "final_reg_value": sum(reg_values[-50:]) / 50,
        "per_step_rollout_mse": per_step_mse,
        "rollout_1step": per_step_mse[0] if len(per_step_mse) >= 1 else None,
        "rollout_4step": per_step_mse[3] if len(per_step_mse) >= 4 else None,
        "rollout_last": per_step_mse[-1] if per_step_mse else None,
        "peak_mem_mb": peak_mb,
        "wall_time_s": t_total,
        "mean_step_ms": 1000 * sum(step_times) / len(step_times),
    }


# ============================================================================
#  Pilot driver
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="results.json")
    parser.add_argument("--reg-weight", type=float, default=0.05,
                        help="weight for randomized-diff penalty in variant B")
    parser.add_argument("--ffn-rank", type=int, default=4,
                        help="rank for variant A FFN factorization")
    parser.add_argument("--quick", action="store_true",
                        help="very short run for smoke test")
    args = parser.parse_args()

    if args.quick:
        args.steps = 200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # ---- problem setup ----
    d = 64
    action_dim = 4
    history_size = 3
    n_preds = 1
    rollout_T = 8     # number of future steps generated per trajectory
    N_train = 4096
    N_val = 512

    print(f"\n=== Toy pilot: d={d}, true_rank=4, history={history_size}, "
          f"rollout_T={rollout_T}, device={device} ===\n")

    U, V, B = make_true_dynamics(d, rank=4, action_dim=action_dim, device=device,
                                 seed=args.seed)
    Z_tr, A_tr = make_dataset(N_train, rollout_T, d, action_dim, U, V, B,
                              device=device, seed=args.seed + 1)
    Z_va, A_va = make_dataset(N_val, rollout_T, d, action_dim, U, V, B,
                              device=device, seed=args.seed + 2)

    # When training one-step, ARPredictor consumes history_size frames and predicts
    # all `history_size` next-frame outputs (autoregressive). We use the standard LeWM
    # objective: predict z_{t+1} from z_t for each t in the context.
    # That means we want Z_b[:, :history_size] -> Z_b[:, 1:history_size+1].
    # rollout_T must be >= history_size + 1.

    # ---- common predictor config ----
    base_kwargs = dict(d=d, action_dim=action_dim, num_frames=history_size,
                       depth=3, heads=4, dim_head=32, mlp_dim=256, dropout=0.0)
    train_kwargs = dict(history_size=history_size, n_preds=n_preds,
                        device=device, steps=args.steps, batch_size=args.batch,
                        lr=3e-4, wd=1e-3)
    train_data = (Z_tr, A_tr)
    val_data = (Z_va, A_va)

    results = []

    # --- Baseline ---
    print(">>> Variant baseline (full-rank predictor, no extra reg)")
    torch.manual_seed(args.seed)
    m = WorldModelToy(**base_kwargs).to(device)
    print(f"    params: {count_params(m):,}")
    r = train_variant("baseline", m, train_data, val_data,
                      reg_weight=0.0, reg_kind="none", **train_kwargs)
    results.append(r); del m; torch.cuda.empty_cache()

    # --- Variant A: UV explicit low-rank ---
    print(f"\n>>> Variant uv_lowr (FFN factored at rank={args.ffn_rank})")
    torch.manual_seed(args.seed)
    m = WorldModelToy(**base_kwargs, ffn_rank=args.ffn_rank).to(device)
    print(f"    params: {count_params(m):,}")
    r = train_variant("uv_lowr", m, train_data, val_data,
                      reg_weight=0.0, reg_kind="none", **train_kwargs)
    results.append(r); del m; torch.cuda.empty_cache()

    # --- Variant B: randomized differential nuclear-norm penalty ---
    print(f"\n>>> Variant rand_diff (full-rank + nuc-norm penalty, lambda={args.reg_weight})")
    torch.manual_seed(args.seed)
    m = WorldModelToy(**base_kwargs).to(device)
    print(f"    params: {count_params(m):,}")
    r = train_variant("rand_diff", m, train_data, val_data,
                      reg_weight=args.reg_weight, reg_kind="rand_diff", **train_kwargs)
    results.append(r); del m; torch.cuda.empty_cache()

    # ---- Summary ----
    print("\n=== Summary ===\n")
    cols = ["name", "params", "final_train_pred_mse", "rollout_1step",
            "rollout_4step", "rollout_last", "peak_mem_mb", "mean_step_ms"]
    header = "  ".join(f"{c:>20}" for c in cols)
    print(header)
    for r in results:
        row = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                if c == "params" or c == "peak_mem_mb":
                    row.append(f"{v:>20.1f}")
                else:
                    row.append(f"{v:>20.5f}")
            elif isinstance(v, int):
                row.append(f"{v:>20d}")
            else:
                row.append(f"{str(v):>20s}")
        print("  ".join(row))

    # config to dump
    out = {
        "config": {
            "d": d, "action_dim": action_dim, "history_size": history_size,
            "n_preds": n_preds, "rollout_T": rollout_T,
            "N_train": N_train, "N_val": N_val,
            "steps": args.steps, "batch": args.batch, "seed": args.seed,
            "reg_weight": args.reg_weight, "ffn_rank": args.ffn_rank,
            "predictor_kwargs": {k: v for k, v in base_kwargs.items() if k != "ffn_rank"},
            "device": str(device),
        },
        "results": results,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
