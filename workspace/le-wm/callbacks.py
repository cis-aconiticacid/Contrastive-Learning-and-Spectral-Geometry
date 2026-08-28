"""Custom Lightning callbacks for the low-rank predictor experiments.

- JacobianProbeCallback: every N steps (and step 0) compute predictor Jacobian SVD
- ProbingCallback: every N steps (and step 0) train Ridge probe latent->state, log R²
- LatentCovCallback: every N steps (and step 0) compute encoder latent covariance eigvals
- EngineeringMetricsCallback: peak GPU mem, wall-clock, param count (once)
- PreClipGradNormCallback: total grad norm BEFORE clipping (uses on_after_backward)
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

import torch
import lightning as pl
from einops import rearrange


# ---------------------------------------------------------------------------
#  Shared helpers
# ---------------------------------------------------------------------------

def _analyze_spectrum(sv: torch.Tensor):
    """Given singular values, compute spectral_norm / stable_rank / effective_rank."""
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
        "top_svs": sv[:32].tolist(),
    }


def _build_fixed_inputs(trainer, n_samples=8):
    """Pick first n_samples deterministically from val loader."""
    val_loader = trainer.val_dataloaders
    if isinstance(val_loader, list):
        val_loader = val_loader[0]
    items = []
    for b in val_loader:
        for i in range(b["pixels"].size(0)):
            items.append({k: v[i] for k, v in b.items() if torch.is_tensor(v)})
            if len(items) >= n_samples:
                break
        if len(items) >= n_samples:
            break
    batch = {k: torch.stack([s[k] for s in items]) for k in items[0]}
    device = trainer.strategy.root_device
    return {k: v.to(device) for k, v in batch.items()}


# ---------------------------------------------------------------------------
#  Jacobian probe (predictor)
# ---------------------------------------------------------------------------

def _predictor_fn(predictor, pred_proj, emb_in, act_in):
    emb = emb_in.unsqueeze(0)
    act = act_in.unsqueeze(0)
    out = predictor(emb, act)
    out = pred_proj(rearrange(out, "b t d -> (b t) d"))
    return out.flatten()


def _compute_predictor_jacobian(predictor, pred_proj, emb_in, act_in):
    from torch.func import jacrev
    fn = lambda e: _predictor_fn(predictor, pred_proj, e, act_in)
    return jacrev(fn)(emb_in).reshape(-1, emb_in.numel())


class JacobianProbeCallback(pl.Callback):
    """Probes predictor Jacobian SVD on a fixed val batch every N steps + step 0."""

    def __init__(self, output_path, every_n_steps=500, n_samples=6):
        super().__init__()
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.every_n_steps = every_n_steps
        self.n_samples = n_samples
        self._fixed_emb = None
        self._fixed_act = None
        if self.output_path.exists():
            self.output_path.unlink()

    def _ensure_fixed(self, trainer, pl_module):
        if self._fixed_emb is not None:
            return
        wm = pl_module.model
        was_training = wm.training; wm.eval()
        try:
            batch = _build_fixed_inputs(trainer, self.n_samples)
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)
            with torch.no_grad():
                info = wm.encode(batch)
            ctx_len = 3
            self._fixed_emb = info["emb"][:, :ctx_len].detach()
            self._fixed_act = info["act_emb"][:, :ctx_len].detach()
        finally:
            if was_training: wm.train()

    def _probe(self, trainer, pl_module, step):
        wm = pl_module.model
        was_training = wm.training; wm.eval()
        try:
            self._ensure_fixed(trainer, pl_module)
            samples = []
            for i in range(self.n_samples):
                J = _compute_predictor_jacobian(wm.predictor, wm.pred_proj,
                                                self._fixed_emb[i], self._fixed_act[i])
                stats = _analyze_spectrum(torch.linalg.svdvals(J))
                stats["sample"] = i
                stats["step"] = step
                samples.append(stats)
            summary = {
                "step": step,
                "mean_spectral_norm": sum(s["spectral_norm"] for s in samples) / len(samples),
                "mean_stable_rank": sum(s["stable_rank"] for s in samples) / len(samples),
                "mean_effective_rank": sum(s["effective_rank"] for s in samples) / len(samples),
                "mean_n_svs_above_1pct": sum(s["n_svs_above_1pct_max"] for s in samples) / len(samples),
                "samples": samples,
            }
            with self.output_path.open("a") as f:
                f.write(json.dumps(summary) + "\n")
            print(f"[jac@{step:>5}] spec={summary['mean_spectral_norm']:.3f} "
                  f"sr={summary['mean_stable_rank']:.1f} "
                  f"er={summary['mean_effective_rank']:.1f}", flush=True)
        finally:
            if was_training: wm.train()

    def on_train_start(self, trainer, pl_module):
        if trainer.is_global_zero: self._probe(trainer, pl_module, 0)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero: return
        gs = trainer.global_step
        if gs > 0 and gs % self.every_n_steps == 0:
            self._probe(trainer, pl_module, gs)

    def on_train_end(self, trainer, pl_module):
        if not trainer.is_global_zero: return
        gs = int(trainer.global_step)
        if gs > 0 and gs % self.every_n_steps == 0: return
        self._probe(trainer, pl_module, gs)


# ---------------------------------------------------------------------------
#  Probing callback (Ridge probe of state from encoder latent)
# ---------------------------------------------------------------------------

class ProbingCallback(pl.Callback):
    """Trains a cheap Ridge probe latent → 7-d state on a fixed val batch.

    Uses the LAST history-size frame's encoder output (single 192-d latent per sample).
    Cheap version: 256 train + 64 test (held-out, both fixed), Ridge alpha=1.0.
    """

    def __init__(self, output_path, every_n_steps=500,
                 n_train=256, n_test=64, ridge_alpha=1.0,
                 history_size=3):
        super().__init__()
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.every_n_steps = every_n_steps
        self.n_train = n_train
        self.n_test = n_test
        self.ridge_alpha = ridge_alpha
        self.history_size = history_size
        self._fixed_pixels_train = None
        self._fixed_state_train = None
        self._fixed_pixels_test = None
        self._fixed_state_test = None
        if self.output_path.exists():
            self.output_path.unlink()

    def _ensure_fixed(self, trainer, pl_module):
        if self._fixed_pixels_train is not None:
            return
        # Build a deterministic train+test split from val loader
        val_loader = trainer.val_dataloaders
        if isinstance(val_loader, list):
            val_loader = val_loader[0]
        N = self.n_train + self.n_test
        items_p = []
        items_s = []
        for b in val_loader:
            B = b["pixels"].size(0)
            for i in range(B):
                items_p.append(b["pixels"][i, self.history_size - 1])  # last frame, (C, H, W)
                items_s.append(b["state"][i, self.history_size - 1])
                if len(items_p) >= N:
                    break
            if len(items_p) >= N:
                break
        device = trainer.strategy.root_device
        pix = torch.stack(items_p).to(device)         # (N, C, H, W)
        st = torch.stack(items_s).to(device).float()  # (N, S)
        self._fixed_pixels_train = pix[:self.n_train]
        self._fixed_state_train = st[:self.n_train]
        self._fixed_pixels_test = pix[self.n_train:self.n_train + self.n_test]
        self._fixed_state_test = st[self.n_train:self.n_train + self.n_test]

    @torch.no_grad()
    def _encode_pixels(self, wm, pixels):
        # pixels (B, C, H, W) -> latent (B, D)
        out = wm.encoder(pixels.float(), interpolate_pos_encoding=True)
        cls = out.last_hidden_state[:, 0]
        return wm.projector(cls)

    def _probe(self, trainer, pl_module, step):
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score
        wm = pl_module.model
        was_training = wm.training; wm.eval()
        try:
            self._ensure_fixed(trainer, pl_module)
            Z_tr = self._encode_pixels(wm, self._fixed_pixels_train).cpu().numpy()
            Z_te = self._encode_pixels(wm, self._fixed_pixels_test).cpu().numpy()
            S_tr = self._fixed_state_train.cpu().numpy()
            S_te = self._fixed_state_test.cpu().numpy()
            reg = Ridge(alpha=self.ridge_alpha)
            reg.fit(Z_tr, S_tr)
            pred = reg.predict(Z_te)
            r2 = float(r2_score(S_te, pred))
            r2_per_dim = [float(r2_score(S_te[:, d], pred[:, d])) for d in range(S_te.shape[1])]
            mse = float(((pred - S_te) ** 2).mean())
            entry = {
                "step": step,
                "overall_r2": r2,
                "per_dim_r2": r2_per_dim,
                "mse": mse,
                "n_train": int(self.n_train),
                "n_test": int(self.n_test),
                "ridge_alpha": float(self.ridge_alpha),
            }
            with self.output_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
            print(f"[probe@{step:>5}] R²={r2:.4f}  per-dim=[" +
                  ",".join(f"{x:.2f}" for x in r2_per_dim) + "]", flush=True)
        finally:
            if was_training: wm.train()

    def on_train_start(self, trainer, pl_module):
        if trainer.is_global_zero: self._probe(trainer, pl_module, 0)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero: return
        gs = trainer.global_step
        if gs > 0 and gs % self.every_n_steps == 0:
            self._probe(trainer, pl_module, gs)

    def on_train_end(self, trainer, pl_module):
        if not trainer.is_global_zero: return
        gs = int(trainer.global_step)
        if gs > 0 and gs % self.every_n_steps == 0: return
        self._probe(trainer, pl_module, gs)


# ---------------------------------------------------------------------------
#  Latent covariance eigenvalue spectrum
# ---------------------------------------------------------------------------

class LatentCovCallback(pl.Callback):
    """Computes (D, D) covariance matrix of encoder latents on a fixed val batch.

    Reports eigenvalue spectrum statistics: top-K, stable rank, effective rank,
    nuclear-norm proxy. Useful diagnostic for whether encoder latent space is
    actually using its full dimensionality.
    """

    def __init__(self, output_path, every_n_steps=500, n_samples=256, history_size=3):
        super().__init__()
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.every_n_steps = every_n_steps
        self.n_samples = n_samples
        self.history_size = history_size
        self._fixed_pixels = None
        if self.output_path.exists():
            self.output_path.unlink()

    def _ensure_fixed(self, trainer):
        if self._fixed_pixels is not None: return
        val_loader = trainer.val_dataloaders
        if isinstance(val_loader, list):
            val_loader = val_loader[0]
        items = []
        for b in val_loader:
            for i in range(b["pixels"].size(0)):
                items.append(b["pixels"][i, self.history_size - 1])
                if len(items) >= self.n_samples:
                    break
            if len(items) >= self.n_samples:
                break
        device = trainer.strategy.root_device
        self._fixed_pixels = torch.stack(items).to(device)

    @torch.no_grad()
    def _probe(self, trainer, pl_module, step):
        wm = pl_module.model
        was_training = wm.training; wm.eval()
        try:
            self._ensure_fixed(trainer)
            out = wm.encoder(self._fixed_pixels.float(), interpolate_pos_encoding=True)
            cls = out.last_hidden_state[:, 0]
            Z = wm.projector(cls)                          # (N, D)
            Zc = Z - Z.mean(0, keepdim=True)
            cov = (Zc.t() @ Zc) / (Z.size(0) - 1)          # (D, D)
            eigvals = torch.linalg.eigvalsh(cov).clamp_min(0).flip(0)  # descending
            stats = _analyze_spectrum(eigvals.sqrt())  # spectrum stats expects "sv-like"
            entry = {
                "step": step,
                "trace": cov.diagonal().sum().item(),
                "spectral_eigval": eigvals[0].item(),
                "stable_rank": stats["stable_rank"],
                "effective_rank": stats["effective_rank"],
                "n_eigvals_above_1pct": stats["n_svs_above_1pct_max"],
                "top_eigvals": eigvals[:32].cpu().tolist(),
                "n_samples": int(Z.size(0)),
                "latent_dim": int(Z.size(1)),
            }
            with self.output_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
            print(f"[lcov@{step:>5}] trace={entry['trace']:.3f} "
                  f"sr={entry['stable_rank']:.2f} "
                  f"er={entry['effective_rank']:.2f}", flush=True)
        finally:
            if was_training: wm.train()

    def on_train_start(self, trainer, pl_module):
        if trainer.is_global_zero: self._probe(trainer, pl_module, 0)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero: return
        gs = trainer.global_step
        if gs > 0 and gs % self.every_n_steps == 0:
            self._probe(trainer, pl_module, gs)

    def on_train_end(self, trainer, pl_module):
        if not trainer.is_global_zero: return
        gs = int(trainer.global_step)
        if gs > 0 and gs % self.every_n_steps == 0: return
        self._probe(trainer, pl_module, gs)


# ---------------------------------------------------------------------------
#  Engineering metrics
# ---------------------------------------------------------------------------

class EngineeringMetricsCallback(pl.Callback):
    """Saves a single JSON with predictor params, peak GPU memory, wall-clock."""

    def __init__(self, output_path):
        super().__init__()
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = None

    def on_train_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            torch.cuda.reset_peak_memory_stats()
            self.t0 = time.time()

    def on_train_end(self, trainer, pl_module):
        if not trainer.is_global_zero: return
        wm = pl_module.model
        info = {
            "predictor_params": sum(p.numel() for p in wm.predictor.parameters()),
            "encoder_params": sum(p.numel() for p in wm.encoder.parameters()),
            "total_params": sum(p.numel() for p in wm.parameters()),
            "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
            "wall_clock_s": time.time() - self.t0,
            "global_step": int(trainer.global_step),
        }
        self.output_path.write_text(json.dumps(info, indent=2))


# ---------------------------------------------------------------------------
#  Pre-clip grad norm via on_after_backward
# ---------------------------------------------------------------------------

class PreClipGradNormCallback(pl.Callback):
    """Logs total grad-norm BEFORE clipping. Hooked to on_after_backward, which
    fires immediately after backward() and BEFORE Lightning's gradient clipping.
    Earlier we used on_before_optimizer_step which on Lightning 2.6 may run AFTER
    `configure_gradient_clipping` in some code paths — moving it to
    on_after_backward is the safe choice.
    """

    def on_after_backward(self, trainer, pl_module):
        if not trainer.is_global_zero: return
        total = 0.0
        n = 0
        for p in pl_module.parameters():
            if p.grad is not None:
                total = total + p.grad.detach().pow(2).sum().item()
                n += 1
        if n > 0:
            pl_module.log("fit/grad_norm_pre_clip", float(total ** 0.5),
                          on_step=True, on_epoch=False)


# Backward-compat alias
GradNormLoggerCallback = PreClipGradNormCallback


# ---------------------------------------------------------------------------
#  KeepFrozenInEvalCallback (used in frozen-encoder ablation, v3)
# ---------------------------------------------------------------------------

class KeepFrozenInEvalCallback(pl.Callback):
    """Ensures the listed submodules of `pl_module.model` stay in eval() mode
    across Lightning's train()/eval() toggles. This is the picklable
    replacement for monkey-patching `.train()` on the submodule itself
    (which would break torch.save(model) for *_object.ckpt).

    Use case: when encoder + projector + pred_proj are frozen for the
    Stage-2 ablation, we don't want their BatchNorm running stats to drift
    when Lightning flips the whole module to train mode.

    Also asserts (once, on fit_start) that the frozen submodules have
    requires_grad=False on every parameter — catches silent freeze bugs.
    """

    def __init__(self, module_names=("encoder", "projector", "pred_proj")):
        super().__init__()
        self.module_names = tuple(module_names)

    def _frozen_modules(self, pl_module):
        for nm in self.module_names:
            m = getattr(pl_module.model, nm, None)
            if m is not None:
                yield nm, m

    def on_fit_start(self, trainer, pl_module):
        for name, m in self._frozen_modules(pl_module):
            bad = [n for n, p in m.named_parameters() if p.requires_grad]
            assert not bad, (
                f"[KeepFrozenInEval] FATAL: {name} has trainable params: {bad[:5]}"
            )
            m.eval()
            print(f"[KeepFrozenInEval] {name}: all params requires_grad=False, in eval mode")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Re-assert eval mode every batch (Lightning may have flipped it back)
        for _, m in self._frozen_modules(pl_module):
            if m.training:
                m.eval()

    def on_train_epoch_start(self, trainer, pl_module):
        for _, m in self._frozen_modules(pl_module):
            m.eval()
