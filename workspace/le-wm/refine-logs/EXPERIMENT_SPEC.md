# Experiment Spec v5 — Detailed Technical Specification

Companion to `EXPERIMENT_PLAN.md`. Specifies exact model architecture, data pipeline, regularizer math, metric definitions, probing protocol, DMD method, per-block measurements, and decision-gate inequalities for v5.

---

## §1 Model architecture (LeWM JEPA on Push-T)

All four sub-modules live inside `JEPA` (defined in `/workspace/le-wm/jepa.py`):

```
JEPA
├── encoder: ViTModel (HuggingFace, ViT-Tiny, no pooling layer)
├── projector: MLP (post-encoder feature transformer)
├── predictor: ARPredictor (autoregressive transformer over (z, action) tokens)
├── pred_proj: MLP (aligns predictor output to encoder-output space)
└── action_encoder: Embedder (turns 10-d "smoothed" action into 192-d emb)
```

### §1.1 Encoder — `transformers.ViTModel`

```python
# built via spt.backbone.utils.vit_hf("tiny", patch_size=14, image_size=224, pretrained=False)
ViTConfig(
    image_size = 224,
    patch_size = 14,                 # → 16×16 = 256 patches + 1 CLS = 257 tokens
    num_channels = 3,
    hidden_size = 192,
    num_hidden_layers = 12,
    num_attention_heads = 3,
    intermediate_size = 768,         # ViT-Tiny FFN expansion
    qkv_bias = True,
    add_pooling_layer = False,
)
# Pixel input: (B, 3, 224, 224) → output.last_hidden_state[:, 0]: (B, 192) CLS token
# Encoder param count: 5,501,376 (from v3 engineering_metrics.json)
```

### §1.2 Projector — `module.MLP`

```
MLP(input_dim=192, hidden_dim=2048, output_dim=192, norm_fn=BatchNorm1d):
  Linear(192 → 2048) → BatchNorm1d(2048) → GELU → Linear(2048 → 192)

# Inputs: (B, 192) CLS post-ViT
# Outputs: (B, 192) "post-projector latent" z_t — this is the JEPA representation
# Parameter count: ~0.79 M
```

### §1.3 Predictor — `module.ARPredictor`

```
ARPredictor(
    num_frames = 3,                  # = wm.history_size
    input_dim = 192,                 # = embed_dim
    hidden_dim = 192,                # = encoder.config.hidden_size
    output_dim = 192,
    depth = 6,                       # transformer blocks
    heads = 16,
    mlp_dim = 2048,                  # FFN expansion (4× hidden_dim)
    dim_head = 64,                   # → inner_dim = heads × dim_head = 1024
    dropout = 0.1,
    emb_dropout = 0.0,
    ffn_rank = None,                 # if int r > 0: replaces FeedForward with LowRankFeedForward
)

# Each Block:
#   Attention(192, heads=16, dim_head=64) → causal self-attn over T=3 latent tokens
#   FeedForward(192, mlp_dim=2048) OR LowRankFeedForward(192, mlp_dim=2048, rank=r)
#   Residual + LayerNorm
#
# Action conditioning: act_emb concatenated/added (depends on Block type) into hidden dim
#
# Parameter count:
#   baseline (no ffn_rank):   10,791,360
#   uvlowr_r4 (ffn_rank=4):   6,126,528  (−43.2%)
```

#### §1.3a `LowRankFeedForward(dim=192, hidden_dim=2048, rank=r, dropout=0.1)`

Replaces each FFN's two Linear layers with rank-r UV factorizations:

```
W1 → w1_a · w1_b:   Linear(dim → r, bias=False) ∘ Linear(r → hidden_dim, bias=True)
GELU, Dropout
W2 → w2_a · w2_b:   Linear(hidden_dim → r, bias=False) ∘ Linear(r → dim, bias=True)

# Implicit nuclear-norm regularization via weight decay:
#   ||W||_* = min_{W=UV^T} (||U||_F² + ||V||_F²) / 2
# AdamW with wd=1e-3 implements this directly.
```

Per FFN parameter cost (vs full-rank):
- Full FFN: `dim·hidden + hidden·dim = 192·2048·2 = 786 432` params (× 6 blocks ≈ 4.7M)
- Rank-4 FFN: `(dim·4 + 4·hidden) + (hidden·4 + 4·dim) = 4·(192+2048+2048+192) + biases ≈ 18 K` per FFN × 6 blocks ≈ 110K
- Rank reduction from r=∞ to r=4 saves ~4.6M of predictor params

### §1.4 pred_proj — `module.MLP`

Identical architecture to projector: `MLP(192, 2048, 192, BatchNorm1d)`. Used to align predictor output space to encoder output space during JEPA training (so that `pred_loss = (predict(z_<3) - z_t)²` operates on aligned spaces).

### §1.5 action_encoder — `module.Embedder`

```
Embedder(
    input_dim = 10,                  # = frameskip(5) × action_dim(2)
    smoothed_dim = 10,
    emb_dim = 192,
    mlp_scale = 4,
):
    patch_embed = Conv1d(10 → 10, kernel_size=1)
    embed = Linear(10 → 4·192=768) → SiLU → Linear(768 → 192)

# Action smoothing: the 5 raw 2-d actions within one latent-step are
# concatenated into a 10-d vector, then Conv1d 1×1 + 2-layer MLP → 192-d emb.
```

### §1.6 Total parameter count

| Variant | encoder | projector | predictor | pred_proj | action_enc | total |
|---|---:|---:|---:|---:|---:|---:|
| baseline   | 5.50M | 0.79M | 10.79M | 0.79M | 0.16M | **18.03M** |
| uvlowr_r4  | 5.50M | 0.79M | 6.13M  | 0.79M | 0.16M | **13.37M** |
| randdiff   | 5.50M | 0.79M | 10.79M | 0.79M | 0.16M | **18.03M** |

(numbers from `engineering_metrics.json` in tier-1 local repro and v3 runs)

---

## §2 Data pipeline (Push-T expert)

### §2.1 Source HDF5

```
/workspace/stablewm_home/pusht_expert_train.h5  (44 GB, blosc-compressed)

Schema:
  pixels   (2 336 736, 224, 224, 3)  uint8       — frame-by-frame
  action   (2 336 736, 2)             float32     — raw 2-d actions
  proprio  (2 336 736, 4)             float32     — agent (x, y, vx, vy)
  state    (2 336 736, 7)             float32     — agent_xyvxvy + block xy + block_angle
  ep_offset (18 685,)                 int64       — start index per episode
  ep_len    (18 685,)                 int64       — length per episode
```

### §2.2 `swm.data.HDF5Dataset` transformation

For each dataset item (one (history_size + num_preds) chunk):

```
num_steps = num_preds + history_size = 1 + 3 = 4
frameskip = 5
span      = num_steps × frameskip = 20 raw frames

# Sampling: random start o, take 20 consecutive raw frames
pixels = h5['pixels'][o : o+20 : 5]   # → (4, 224, 224, 3)  ← subsampled by 5
state  = h5['state'][o : o+20 : 5]    # → (4, 7)
proprio= h5['proprio'][o : o+20 : 5]  # → (4, 4)
action = h5['action'][o : o+20]       # → (20, 2)  ← NOT subsampled
action = action.reshape(4, 10)        # → (4, 10)  ← 5 frames concatenated
```

### §2.3 Normalizers

`utils.get_column_normalizer(dataset, col, col)` builds a per-column normalizer fitted on the full dataset:
- proprio, state, action: standard z-score per dimension
- pixels: 0..255 → 0..1 (handled by `get_img_preprocessor`)

### §2.4 Train/val split

```python
rnd_gen = torch.Generator().manual_seed(cfg.seed)  # cfg.seed varies per run
train_set, val_set = spt.data.random_split(
    dataset, lengths=[0.9, 0.1], generator=rnd_gen
)
```

Different seeds → different splits. v3 used seeds 0, 1 for the two-seed condition.

### §2.5 Validation trajectories for trajectory-level metrics (B5, tier-1)

For DMD / TLPS / curvature / TwoNN we pick 80 trajectories from the last 10% of episodes (val-like region), each providing 23 subsampled latent steps (3 history + 20 horizon).

```python
candidates = [i for i in range(int(0.9 * n_eps), n_eps) if ep_len[i] >= 115]
chosen = rng.choice(candidates, size=80, replace=False)
```

---

## §3 Training stack

### §3.1 Lightning + spt.Module

```python
trainer = pl.Trainer(
    max_epochs = 1,
    devices = "auto",
    accelerator = "gpu",
    precision = "bf16",              # mixed precision
    gradient_clip_val = 1.0,
    callbacks = [
        ModelObjectCallBack,           # saves *_epoch_1_object.ckpt
        JacobianProbeCallback,         # writes jacobian_probe.jsonl
        ProbingCallback,               # writes probing.jsonl
        LatentCovCallback,             # writes latent_cov.jsonl
        EngineeringMetricsCallback,    # writes engineering_metrics.json
        PreClipGradNormCallback,       # logs fit/grad_norm_pre_clip
        # v3+:
        KeepFrozenInEvalCallback,      # re-asserts eval-mode on frozen modules per batch
    ],
    limit_train_batches = 8000,       # for Stage-2 (v5 frozen runs)
    limit_val_batches = 20,
)
```

### §3.2 Optimizer

```yaml
optimizer:
  type: AdamW
  lr: 5e-5                # default for v3 frozen runs
  weight_decay: 1e-3      # implements implicit nuclear-norm reg via UV factorization
betas: (0.9, 0.95)        # set inside spt.Module default
scheduler: LinearWarmupCosineAnnealingLR  # warmup 0, cosine decay over max_epochs
gradient_clip_val: 1.0
```

In frozen-encoder mode, optimizer only updates predictor + action_encoder parameters (encoder + projector + pred_proj have `requires_grad=False` and are kept in `.eval()` by KeepFrozenInEvalCallback).

### §3.3 Batch / loader

```yaml
loader:
  batch_size: 32                     # AutoDL 4090; reduce to 8 for local 4060
  num_workers: 4                     # AutoDL; 0 required on WSL2 (shm = 64 MB)
  persistent_workers: True
  pin_memory: True
  prefetch_factor: 2
shuffle: True
drop_last: True
```

### §3.4 Forward pass (`lejepa_forward` in train.py)

```
1. emb = encode(batch)            # (B, T=4, 192)  post-projector
2. ctx_emb = emb[:, :3]           # history
3. tgt_emb = emb[:, 1:]           # targets (next 3 latents)
4. pred_emb = predict(ctx_emb, ctx_act_emb)   # (B, 3, 192) post-pred_proj
5. pred_loss = ((pred_emb - tgt_emb)²).mean()
6. if not frozen:
       sigreg_loss = SIGReg(emb.transpose(0,1))
       total = pred_loss + 0.09 · sigreg_loss
   else:
       total = pred_loss
7. if randdiff enabled (any mode):
       rd_loss = 0
       for d in range(num_dirs):
           v = randn_like(ctx_emb_d); v /= ||v||_flat
           pert = predict(ctx_emb_d + ε·v, ctx_act_emb)
           base = predict(ctx_emb_d, ctx_act_emb)
           rd_loss += ||pert - base||_flat / ε
       rd_loss /= num_dirs
       total += λ · rd_loss      # λ = loss.rand_diff.weight (typically 0.01)
```

---

## §4 Regularizers — exact math

### §4.1 SIGReg (Stage-1 only; dropped in frozen Stage-2)

`L_SIGReg(emb)` for `emb: (T, B, D)`:

1. Sample `A ∈ R^{D × 1024}` random unit-norm vectors per call: `A = randn / ||randn||_col`
2. Project: `x = emb · A → (T, B, 1024)`
3. Build Epps-Pulley statistic on each random 1-d projection:

```
t   = [0, 0.1875, ..., 3.0]        # 17 knots over [0, 3]
φ(t) = exp(-t²/2)                  # standard normal characteristic function
weights = [Δt/2, Δt, Δt, ..., Δt/2] · φ(t)   # composite Simpson-style weights, gaussian-windowed

For each projected dimension and each batch slice:
    char_emp_cos(t) = (1/B) Σ_b cos(t · x_b)
    char_emp_sin(t) = (1/B) Σ_b sin(t · x_b)
    err(t)          = (char_emp_cos(t) − φ(t))² + char_emp_sin(t)²
    stat = (err · weights) · B

L_SIGReg = mean over projections and time of `stat`
```

Pushes the latent cloud toward an isotropic Gaussian by minimizing distance between empirical and standard-normal characteristic functions on random 1-d projections.

Weight in total loss: `cfg.loss.sigreg.weight = 0.09`.

**In v5 frozen runs (B1-B4): SIGReg is dropped** (`+freeze.skip_sigreg=true`). Rationale: with encoder+projector frozen, SIGReg gradient has no consumer → wastes compute. Already confirmed in v3 ablation.

### §4.2 uvlowr — UV factorization with implicit nuclear norm

Replace each FFN's two `Linear(in, out)` layers with `Linear(in, r) ∘ Linear(r, out)`. The nuclear norm of the composite W = U·V satisfies:

```
||W||_* = min_{W = U V^T}  (||U||_F² + ||V||_F²) / 2
```

So the variational form of nuclear-norm regularization on W is equivalent to **L₂ regularization on U and V separately**. AdamW's `weight_decay = 1e-3` implements exactly this implicit penalty on each factor.

This is **purely architectural**: zero extra training loss; works on the predictor weights themselves, not the Jacobian.

### §4.3 randdiff (Scarvelis & Solomon NeurIPS 2024, arXiv:2405.14544)

Randomized one-sided finite-difference estimator of the predictor's Jacobian nuclear norm:

```
For each batch:
  z_d  = z.detach()                        # so we don't backprop through encoder
  base = predict(z_d, a)                   # forward pass
  total = 0
  for j in range(num_dirs):                 # default num_dirs = 2
      v_j ~ N(0, I)                         # in flat (B·T·D)-dim
      v_j /= ||v_j||_flat                   # unit norm per sample
      pert = predict(z_d + ε · v_j, a)
      diff = (pert - base).flatten(1)
      total += diff.norm(dim=-1).mean()  / ε    # per-sample FD norm, averaged
  L_randdiff = total / num_dirs
```

This estimates `E_v [||∂predict/∂z|_z · v||]` which is an upper bound on a Hutchinson-flavored estimate of `||J||_*`. In v3/v4/v5 defaults: `ε=0.05, num_dirs=2, λ=0.01` (multiplier on `L_randdiff` in the total loss).

### §4.4 SIGReg vs uvlowr vs randdiff — design matrix

| Regularizer | Acts on | Direct loss term? | Frozen-Stage-2 active? |
|---|---|---|---|
| SIGReg | encoder output (latent distribution) | YES (in Stage-1) | NO (dropped) |
| uvlowr | predictor FFN weights (architectural) | NO (implicit via wd) | YES (architectural) |
| randdiff | predictor Jacobian (samples) | YES | YES |

---

## §5 Instrumentation callbacks (write `*.jsonl` / `.json` to run dir)

Run dir = `${STABLEWM_HOME}/${name}/`. All callbacks fire every `probe_every` steps (default 500; in v3/v5 → 500).

### §5.1 `JacobianProbeCallback` → `jacobian_probe.jsonl`

Computes predictor Jacobian on `n_samples=6` randomly drawn validation samples per probe step.

For each sample (B=1 context):
- Builds `f(z) = predict(z, a)` evaluated at the real context's `z`
- Uses `torch.func.jacrev` to get full Jacobian J ∈ R^{D × D} (D = 192 × 3 history → flattened)
- Computes:
  - `spectral_norm = σ₁(J)`
  - `frobenius_norm_sq = ||J||_F²`
  - `top_svs = top 32 singular values of J` (descending)
  - `stable_rank = ||J||_F² / σ₁²`
  - `effective_rank = exp(-Σ pᵢ log pᵢ)` where `pᵢ = σᵢ²/Σσⱼ²`
  - `n_svs_above_1pct_max = #{σᵢ > 0.01 σ₁}`

Per-step output averaged over the 6 samples for `mean_*` fields; per-sample `top_svs` and stats also stored.

### §5.2 `ProbingCallback` → `probing.jsonl`

Trains an inline ridge regression at each probe step:
- `n_train=256, n_test=64, alpha=1.0` (default for v3+v5)
- target = 7-d state (agent_xy + agent_vxvy + block_xy + block_angle)
- features = post-projector latents at t=0 of each (history, prediction) chunk
- Output: `overall_r2`, `per_dim_r2 [7]`, `mse`, `n_train`, `n_test`

### §5.3 `LatentCovCallback` → `latent_cov.jsonl`

Computes empirical covariance of post-projector latent z over `n_samples=256` validation samples (default for v3+v5).

```
Z ∈ R^{256 × 192}
C = (Z - mean) · (Z - mean)^T / (256 - 1)     # 192 × 192
eigvals = eigvalsh(C)  sorted desc
trace = sum(eigvals)                          # always returns full trace
top_eigvals = eigvals[:32]                    # top 32 stored
spectral_eigval = eigvals[0]
stable_rank = (trace)² / Σ eigvals²
effective_rank = exp(-Σ pᵢ log pᵢ)            # entropy-based
n_eigvals_above_1pct = #{λᵢ > 0.01 · eigvals[0]}
```

### §5.4 `EngineeringMetricsCallback` → `engineering_metrics.json`

```
predictor_params, encoder_params, total_params:  int
peak_gpu_memory_mb: float
wall_clock_s: float
global_step: int
```

### §5.5 `PreClipGradNormCallback` → logs `fit/grad_norm_pre_clip` to CSV

Records pre-clip gradient norm of the entire model at each step. Useful for detecting training instability.

---

## §6 Per-block exact measurements + decision criteria

### §6.0 v3 reference numbers (reused as control points for v5 sweeps)

From `/workspace/lewm_autodl_results_v3/`:

| Run | Metric | v3 value | v5 reuse role |
|---|---|---|---|
| R002 frozen_baseline_seed0 | σ₁ | 2.05 | B1 ε baseline; B4 λ=0 |
| R002 | F² | 87.5 ± 1.7 | B4 baseline F² |
| R002 | SR | 20.8 ± 0.6 | B4 baseline SR |
| R002 | rollout MSE h=10 | 0.267 | B4 baseline rollout |
| R002 | tier-1 \|λ_max\| | 1.095 | B4 baseline DMD |
| R002 | tier-1 n_unstable | 5 | B4 baseline DMD |
| R004 frozen_uvlowr_r4_seed0 | σ₁ | 2.06 | B3 r=4 point |
| R004 | rollout h=10 | 0.318 | B3 r=4 point |
| R004 | tier-1 \|λ_max\| | 1.073 | — |
| R006 frozen_randdiff_seed0 | σ₁ | 1.61 | B1 ε=0.05; B4 λ=0.01 |
| R006 | F² | 59.2 ± 0.9 | B4 λ=0.01 |
| R006 | SR | 22.7 ± 0.4 | B4 λ=0.01 |
| R006 | rollout h=10 | 0.961 | B4 λ=0.01 |
| R006 | tier-1 \|λ_max\| | 0.996 | B4 λ=0.01 |
| R006 | tier-1 n_unstable | 0 | B4 λ=0.01 |

These are the **anchor points** v5 sweeps will surround. Reusing them saves 4 GPU-h.

### §6.1 B1 ε sweep (frozen randdiff, on v3 baseline Stage-1)

**4 runs**: ε ∈ {0.0125, 0.05 (reuse R006), 0.2, 0.8}, seed=0, 8K steps, λ=0.01, num_dirs=2.

**Measure (per run)**:
- From `jacobian_probe.jsonl` final step (step=8000):
  - σ₁(predictor) — `mean_spectral_norm`
  - F²(predictor) — average of `samples[*].frobenius_norm_sq`
  - SR(predictor) — `mean_stable_rank`
  - σ₁/σ₁₀ — computed post-hoc from `samples[*].top_svs`
  - β — power-law lstsq on log(σᵢ) ~ log(i), i ∈ [1..32]
- From `eval.json`:
  - rollout MSE at h ∈ {1, 3, 5, 10, 15, 20}, n_test=256
  - probing R² (n_train=2048, n_test=256, ridge α=1.0)
- From `eval_rollout_per_step.csv`:
  - per-step seed-level MSE (for std bars at each h)
- Post-hoc DMD (B5):
  - |λ_max|, n_unstable, top-5 |λ| from rank-32 DMD on 80 val trajectories

**Decision (P-C2 evaluation)**:

```
P-C2 PASSES iff:
   #{ε ∈ {0.0125, 0.05, 0.2, 0.8} : σ₁(randdiff, ε) < σ₁(baseline) − 0.10} ≥ 3
   (baseline σ₁ = 2.07 ± 0.06 from R002+R003 v3 mean)
```

The −0.10 buffer is larger than v3 baseline std (0.06) and accounts for the v3 reversal magnitude (~−21% = −0.44).

**Failure interpretation tree**:
- All 4 ε produce σ₁ ≥ baseline → v3 reversal was ε-specific → C2 reframed to "regime-dependent"
- 0 of 4 ε produce v2-style σ₁ > baseline +30% → no ε reproduces v2 inflation → C2 strong
- ε = 0.0125 (¼v3) shrinks more than ε = 0.05 → confirms "smaller ε = more linear regime" → σ₁ measures local Jacobian → spectrum-concentration is encoder-driven not Jacobian-driven

### §6.2 B2 uvlowr-Stage-1 + freeze multi-predictor

**1 Stage-1 run + 6 Stage-2 frozen runs**.

**Stage-1 run (R220)**:
- `+predictor.ffn_rank=4`, SIGReg ACTIVE (λ_sigreg = 0.09), 20K steps, batch 32, seed 0
- Stop conditions: `fit/pred_loss` plateaued (last-2K-step slope < 0.001) OR step = 25K
- Saves ckpt: `${STABLEWM_HOME}/v5_stage1_uvlowr/v5_stage1_uvlowr_weights.ckpt`

**Stage-2 frozen runs (R230-R235)**: 3 variants × 2 seeds × 8K, each freezing encoder+projector+pred_proj from R220's ckpt.

**Measure (per run)**:
- Same metrics as B1, plus:
- Cross-Stage-1 comparison table:

| | on v3 baseline-Stage-1 (v3 numbers) | on v5 uvlowr-Stage-1 (B2 numbers) | direction match? |
|---|---|---|---|
| baseline σ₁ | 2.07 ± 0.06 | TBD | — |
| uvlowr σ₁ | 2.08 ± 0.04 | TBD | — |
| randdiff σ₁ | 1.63 ± 0.03 | TBD | (sign of randdiff vs baseline within each Stage-1) |
| baseline F² | 87.5 ± 1.7 | TBD | — |
| randdiff F² | 59.2 ± 0.9 | TBD | F²(randdiff) < F²(baseline) by ≥20% on both? |
| rollout h=10 ordering | uvlowr < baseline < randdiff | TBD | preserved? |

**Decision**:
```
B2 confirms C2 generalization iff:
  (i)  σ₁(randdiff, uvlowr-S1) < σ₁(baseline, uvlowr-S1) − 0.10     # sign-flip direction holds
  (ii) F²(randdiff, uvlowr-S1) / F²(baseline, uvlowr-S1) < 0.80     # ≥20% shrinkage

Both must hold across both seeds (seed 0 and seed 1), with mean satisfying the inequality.
```

### §6.3 B3 uvlowr rank sweep

**4 runs** (r=4 reuses R004): r ∈ {1, 2, 4 (reuse), 8, 16}, seed 0, 8K steps, on v3 baseline Stage-1.

**Measure**:
- rollout MSE at h ∈ {1, 5, 10, 15, 20} per r → primary curve
- predictor params per r → param-budget calibration
- predictor σ₁/SR/F² per r → Jacobian-side correlation
- probing R² overall per r → "is the rank constraint hurting probing too?"

**Decision (P-C13 evaluation)**:

```
P-C13 PASSES iff:
  argmin_{r ∈ {1, 2, 4, 8, 16}} rollout MSE(h=10, r) ∈ {2, 4}
  AND rollout MSE(h=10, r=16) ≥ rollout MSE(h=10, r=4) − 0.02

(i.e., r=4 or r=2 is the best; r=16 doesn't beat r=4 by more than noise.)
```

Visual: rollout-vs-r plot should look like a U (or hockey-stick with plateau), not monotone.

### §6.4 B4 randdiff λ dose-response ⭐

**4 runs** (λ=0 and λ=0.01 reused): λ ∈ {0 (reuse R002), 0.001, 0.003, 0.01 (reuse R006), 0.03, 0.1}, seed 0, 8K, ε=0.05, num_dirs=2, on v3 baseline Stage-1.

**Measure (per λ)**:

| Metric | Source | Computation |
|---|---|---|
| σ₁(predictor) | jacobian_probe.jsonl final | `mean_spectral_norm` |
| F²(predictor) | jacobian_probe.jsonl | `mean(samples[*].frobenius_norm_sq)` |
| SR(predictor) | jacobian_probe.jsonl | `mean_stable_rank` |
| rand_diff_loss final | csv/metrics.csv | last `fit/rand_diff_loss` |
| rollout MSE | eval.json | per-h |
| **\|λ_max\|** | DMD on rollout | top eigval magnitude (see §6.6) |
| **n_unstable** | DMD | #{i: \|λᵢ\| > 1} |
| **spectral abscissa** | DMD | max log\|λᵢ\| |
| predicted TLPS | tier-1 metric | mean cos(v_t, v_{t+1}) over 80 traj × 20 h |
| predicted curvature | tier-1 | ‖a‖ / ‖v‖² mean |

**Decision (P-Cdyn evaluation)**:

```
P-Cdyn PASSES iff:
  |λ_max|(λ=0.1) ≤ |λ_max|(λ=0) − 0.05
  AND |λ_max|(λ) is monotone non-increasing across λ ∈ {0, 0.001, 0.003, 0.01, 0.03, 0.1}
      (allowing ±0.02 noise tolerance between adjacent λ)
```

**Expected dose-response plot** (Fig 4):
- x-axis: log(λ) from 1e-3 to 1e-1
- left y-axis: |λ_max| — should monotone-decrease from ~1.10 (λ=0) to ~0.7 (λ=0.1)
- right y-axis: rollout MSE @ h=10 — should show U-curve or monotone increase (predictor over-stabilizes → bad)

If P-Cdyn holds AND the U-curve appears AND there's a "sweet spot" λ where rollout improves over baseline → that's a publishable mechanism story.

### §6.5 B5 post-hoc DMD on full-budget ckpts

**No new training**. Just runs `tier1_dump_trajectories.py` (extended to take any ckpt path) + `tier1_metrics.py` over all v5 frozen ckpts produced by B1, B2, B3, B4.

**Total ckpts to process**:
- B1: 3 new + 1 reused (v3 R006) = 4
- B2: 6 new
- B3: 4 new + 1 reused (v3 R004) = 5
- B4: 4 new + 2 reused (v3 R002, R006) = 6
- v3 reference: 3 (R002, R004, R006)
- Total: ~20 ckpts × 5 min each ≈ 1.5 GPU-h

**Decision (tier-1 replication)**:

```
DMD replication at full 8K is confirmed iff:
  |λ_max|(B4 λ=0.01, 8K v5) ≤ |λ_max|(tier-1 3K randdiff) + 0.05
  AND |λ_max|(B4 λ=0.01, 8K) < |λ_max|(B4 λ=0, 8K) − 0.05

(numerically: tier-1 randdiff = 0.996; v5 should be in [0.95, 1.05] roughly)
```

If at 8K all variants have |λ_max| = 1 → tier-1 finding was a 3K-step artifact → reframe C_dyn as "more contractive than baseline" only.

### §6.6 DMD methodology + nonlinear contraction ρ (B5 augmentation, v5.1)

For each ckpt, compute trajectories then DMD spectrum:

```python
# Inputs:
#   Predicted trajectory z ∈ R^{N × (H+1) × D}, N=80, H=20, D=192
#   Pool snapshots across all N trajectories:
X = z[:, :H].reshape(-1, D).T              # (D=192, N·H=1600)
Y = z[:, 1:].reshape(-1, D).T              # same shape

# Truncated SVD of X
U, S, Vt = svd(X, full_matrices=False)
r = min(32, S.shape[0])                    # rank cap = 32
U_r, S_r, Vt_r = U[:, :r], S[:r], Vt[:r, :]

# Reduced linear operator
A_tilde = U_r.T @ Y @ Vt_r.T @ diag(1/S_r)   # (r, r)

# Eigendecomposition
eigs = eigvals(A_tilde)                    # complex r-vector

# Reporting
|λ_max|       = max |eigs|
n_unstable    = #{|eigs[i]| > 1.0}
n_marginal    = #{0.99 ≤ |eigs[i]| ≤ 1.01}
spectral_abscissa = max log|eigs|
```

Same as in `tier1_metrics.py` (already written). For B5, we apply this to each v5 ckpt's predicted trajectory.

#### §6.6.b Nonlinear contraction ratio ρ (NEW v5.1)

DMD's |λ_max| is a property of the **linear** approximation of the predictor's local dynamics. |λ_max| < 1 does NOT strictly imply the **nonlinear** predictor is a contraction map — the linear spectrum could be dominated by stable modes while the nonlinear behavior is expansive in some directions. Reviewer attack #G.

Direct fix: compute a nonlinear Lipschitz-like ratio on validation latents:

```
ρ = E_{z_1, z_2 ∼ Z_val}  ||f(z_1) − f(z_2)|| / ||z_1 − z_2||

where:
  f = wm.predict (full nonlinear forward pass over the predictor)
  Z_val = pool of post-projector latents from 80 val trajectories, 21 timesteps each
        ≈ 1680 candidate latents
  N = 500 random pairs (z_1, z_2), z_1 ≠ z_2
  action a_1 fixed per pair (use the action from z_1's trajectory)

Report (per ckpt):
  ρ_mean = E[ratio]           ← average local Lipschitz on val latents
  ρ_p95  = 95th percentile     ← tail of expansion
  ρ_max  = max ratio
```

**Interpretation**:
- `ρ_mean < 1` strictly → predictor is a global contraction on val support (nonlinear confirmation of |λ_max|<1)
- `ρ_mean ≈ 1` but `|λ_max| < 1` → spectrum dominated by stable modes but nonlinear behavior still expansive in some directions → C_dyn narrowed to "spectral contraction only"
- `ρ_p95 > 1` while `ρ_mean < 1` → contractive on average but occasional expansion → reframe as "predominantly contractive"

**Cost**: essentially free; piggy-backs on B5 DMD trajectory dumps. ~5 sec extra per ckpt.

**Cross-validation check** (sanity): if our DMD and ρ are measuring the same phenomenon, expect Pearson r(|λ_max|, ρ_mean) > 0.7 across the λ-sweep ckpts (B4). If r is weak, the two metrics tell different stories — that itself is the finding to report.

### §6.7 TLPS, curvature, TwoNN ID (tier-1 carried into v5 — already implemented)

Per `tier1_metrics.py`:

- **TLPS** = mean_t cos(v_t, v_{t+1}) where v_t = z_{t+1} − z_t
- **κ** (curvature) = mean_t ||v_{t+1} − v_t|| / (||v_t||² + ε)
- **TwoNN ID**: fit log(1 − F̂(μ)) = −d log μ via OLS on sorted μ = r₂/r₁ ratios; report slope = d

These are SECONDARY in v5 (per novelty check). Reported in appendix figure only.

---

## §7 Probing protocol (final-step ridge)

Used in `eval_rollout.py`:

```python
# Inputs: predictor's output latents at t=H, and ground-truth state at t=H
Z_train, Z_test = encode(pixels)[:, -1], encode(pixels)[:, -1]    # post-projector
S_train, S_test = state[:, -1], state[:, -1]                       # ground truth 7-d
# n_train = 1024, n_test = 256, ridge α = 1.0

W = solve(Z_train^T Z_train + α I, Z_train^T S_train)
pred = Z_test @ W
R² = 1 - sum((S_test - pred)²) / sum((S_test - mean(S_test))²)
```

The 7 state dimensions are: agent_x, agent_y, agent_vx, agent_vy, block_x, block_y, block_angle.

For h-decaying probe (B4 secondary), we compute R²_h at each rollout step h ∈ {0, 1, 3, 5, 10, 15, 20} using the predicted latent at horizon h vs ground-truth state at horizon h.

---

## §8 Pre-registered prediction summary (P-* gates, v5.1 with σ_noise)

Updated to use **empirical σ_noise** from B0' 5-seed baseline (R002, R003, R202, R203, R204):

```
P-C2:    #{ε ∈ {0.0125, 0.05, 0.2, 0.8} : σ₁(randdiff, ε) ≤ baseline_mean − 2·σ_noise[σ₁]} ≥ 3

P-Cdyn:  |λ_max|(λ=0.1) ≤ |λ_max|(λ=0) − 2·σ_noise[|λ_max|]
         AND  |λ_max| monotone non-increasing across λ ∈ {0, 0.001, 0.003, 0.01, 0.03, 0.1}
              (with ±σ_noise tolerance between adjacent λ)
         AND  ρ_mean(randdiff at λ=0.01) < 0.95     ← nonlinear contraction confirmation

P-C13:   argmin_{r ∈ {1, 2, 4, 8, 16}} rollout-MSE(h=10, r) ∈ {2, 4}
         AND  rollout-MSE(h=10, r=16) ≥ rollout-MSE(h=10, r=4) − 2·σ_noise[rollout_h10]

P-Anti-I (NEW, B4'): σ₁(randdiff @ 40K) ≤ baseline_mean − 2·σ_noise[σ₁]
                     (40K Stage-2 doesn't flip signature back to v2)

P-Anti-H (NEW, B5'): all 4 (ε, λ) corners satisfy σ₁ ≤ baseline_mean + 2·σ_noise[σ₁]
                     (no corner reproduces v2 inflation)

Abort gate M1→M2:
         If ANY ε in B1 produces σ₁(randdiff,ε) ≥ baseline_mean + 0.20
         → STOP. Do not run M2. Reframe paper.
```

**Overall gate**: if ≥ 2/3 main P-* AND P-Anti-I AND P-Anti-H all hold → proceed to draft.

**σ_noise table** (placeholders, fill from B0' results):

| Metric | σ_noise (n=5 seeds, B0' R002+R003+R202+R203+R204) | 2σ threshold |
|---|---|---|
| σ₁ (predictor) | ___ | ___ |
| F² (predictor) | ___ | ___ |
| SR (predictor) | ___ | ___ |
| rollout MSE h=10 (baseline) | ___ | ___ |
| \|λ_max\| (DMD tier-1 protocol) | ___ | ___ |
| ρ_mean (nonlinear contraction) | ___ | ___ |

---

## §9 File outputs schema

Run-level outputs (per training run, in `${STABLEWM_HOME}/<run_name>/`):
```
config.yaml                         hydra-merged config
csv/version_0/metrics.csv           CSVLogger output (per-step metrics)
engineering_metrics.json            {predictor_params, peak_gpu_mb, wall_clock_s}
jacobian_probe.jsonl                per-probe-step predictor Jacobian
probing.jsonl                       per-probe-step inline ridge R²
latent_cov.jsonl                    per-probe-step latent covariance
eval.json                           final rollout MSE per h + final probe R²
eval_rollout_per_step.csv           per-step rollout MSE breakdown
final_analyses.json                 weight SVD per layer + encoder Jacobian + MLP probe
<name>_weights.ckpt                 state_dict only (216 MB, reusable as Stage-1)
<name>_epoch_1_object.ckpt          full pickled spt.Module (70 MB, for eval)
```

v5 aggregate-level outputs (in `lewm_autodl_results_v5/`):
```
ANALYSIS.md                         v5 results writeup with all P-* verdicts
aggregated_v5/eps_sweep/            CSV: variant × ε × metric
aggregated_v5/rank_sweep/           CSV: variant × r × metric
aggregated_v5/lam_sweep/            CSV: variant × λ × metric
dmd_full_budget/<run>.json          per-ckpt DMD: |λ_max|, eigs, TLPS, curvature
v5_cross_condition.csv              master table: (stage1_source, variant, ε, λ, r, h, metrics)
figures/                            Fig 2 (ε sweep), Fig 3 (rank), Fig 4 (λ dose-response)
```

---

## §10 Code modules — what does what

| File | Role | Modified for v5? |
|---|---|---|
| `train.py` | main hydra entry; forward-pass logic; freeze logic | NO (uses v3 patches) |
| `jepa.py` | JEPA module (encode + predict + rollout) | NO |
| `module.py` | ARPredictor, LowRankFeedForward, MLP, SIGReg, Embedder, Attention, FeedForward, Block | NO |
| `callbacks.py` | JacobianProbe, Probing, LatentCov, Engineering, PreClipGradNorm, **KeepFrozenInEval** | NO (uses v3 patches) |
| `eval_rollout.py` | post-training rollout MSE eval | NO |
| `final_analyses.py` | post-training weight SVD + encoder Jac + MLP probe | NO |
| `aggregate.py` | per-variant CSV aggregation | minor extensions for v5 |
| **`run_pipeline_v5.sh`** | NEW orchestrator: M0/B1/B2/B3/B4/B5/B6 | NEW |
| **`tier1_dump_and_metrics.py`** | NEW unified script: takes ckpt path, dumps + computes DMD + tier-1 metrics | NEW |
| `tier1_dump_trajectories.py` | tier-1 trajectory dumper (3 hardcoded variants) | extended to CLI in v5 |
| `tier1_metrics.py` | tier-1 metric computation | reused |
| `tier0_analysis.py` | PR / gap / β / SV decay | reused |
| `tier0_correlation.py` | cross-metric correlation matrix | reused |
| `v4_eps_bracket_probe.py` | R100 latent-norm probe → ε bracket | reused |

---

## §11 Operational lessons inherited (v2 → v3 → v4 → v5)

- AutoDL disk: `expand_system_disk_by_gb=80` from instance creation (Push-T h5 decompress needs ~27 GB+ headroom)
- AutoDL network: speedtest < 2 MB/s → release + recreate
- `~/.pip/pip.conf` to tsinghua/aliyun mirror BEFORE `pip install torch`
- `flock -n /workspace/_setup_v5.lock` to prevent parallel-setup clobbering
- Never `pkill -f "pip install"` from a shell whose cmdline contains "pip install" (kills self)
- SSH cmds via paramiko default to `/root`; always prefix `cd /workspace/le-wm &&`
- Picklable freeze: use `KeepFrozenInEvalCallback` (NOT `module.train = MethodType(...)`)
- AutoDL: ¥3/hr while running; release immediately after final scp pull
- WSL2 local reproduction: `num_workers=0` (shm = 64 MB) is mandatory

---

## §11b NEW v5.1 blocks: B0', B4', B5' specs

### §11b.1 B0' — cross-seed noise floor

| Field | Value |
|---|---|
| Variant | `frozen_baseline` (no regularizer) |
| Stage-1 | v3 `stage1_baseline_seed0_weights.ckpt` |
| Seeds | {0 (reuse R002), 1 (reuse R003), **2, 3, 4 (new)**} |
| Steps | 8000 |
| Metrics computed per seed | σ₁, F², SR, σ₁/σ₁₀, β, |λ_max|, ρ_mean |
| Aggregate over 5 seeds | mean ± std → σ_noise table → feeds P-* thresholds |
| Run IDs | R202, R203, R204 (R002, R003 reused) |
| Cost | 3 × 0.8h = 2.4 GPU-h |

### §11b.2 B4' — 40K Stage-2 extension (under-training discharge)

| Field | Value |
|---|---|
| Variant | `frozen_randdiff` (λ=0.01, ε=0.05, num_dirs=2) |
| Stage-1 | v3 baseline |
| Seed | 0 |
| Steps | **40000** (5× v3's 8K) |
| Metrics tracked per checkpoint | σ₁, F², SR, |λ_max|, rollout MSE at {8K, 16K, 24K, 32K, 40K} |
| Decision | σ₁ at 40K must remain ≤ baseline_mean − 2·σ_noise[σ₁] (v3 direction) |
| Run ID | R256 |
| Cost | ~3 GPU-h (5× longer than typical 8K run) |

### §11b.3 B5' — (ε, λ) 4-corner cross-product

| Run ID | ε | λ | Other |
|---|---|---|---|
| R261 (low-low) | 0.0125 | 0.001 | num_dirs=2, on v3 baseline Stage-1, 8K, seed=0 |
| R262 (low-high) | 0.0125 | 0.1 | same |
| R263 (high-low) | 0.8 | 0.001 | same |
| R264 (high-high) | 0.8 | 0.1 | same |

Decision: P-Anti-H requires ALL 4 corners to satisfy σ₁ ≤ baseline_mean + 2·σ_noise[σ₁].
Cost: 4 × 0.8h = 3.2 GPU-h.

### §11b.4 ⭐ B10 NICE — DINOv2 cross-pretraining (anti-claim J)

**Tests**: C_dyn (over-contraction) generalizes to a non-LeWM encoder.

**Architecture swap**:

```python
# 1. Load DINOv2 ViT-S/14 (no training, frozen)
import torch
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
dinov2.eval()
for p in dinov2.parameters(): p.requires_grad_(False)

# Native DINOv2 ViT-S/14:
#   image_size = 518 (default) or 224 (works with interpolation)
#   patch_size = 14 → 224/14 = 16 patches per side → 256 patch tokens + 1 CLS
#   hidden_size = 384

# 2. Adapter: 384 → 192 fixed linear projection
def build_adapter(dim_in=384, dim_out=192, seed=42, init='random'):
    """Fixed (frozen) projection from DINOv2 patch-mean → LeWM embed_dim.

    init='random': Gaussian rows L2-unit-normalized
    init='pca':    fit PCA on 1000 val-frame DINOv2 patch-mean features
    """
    g = torch.Generator().manual_seed(seed)
    if init == 'random':
        W = torch.randn(dim_in, dim_out, generator=g)
        W = W / W.norm(dim=0, keepdim=True)   # column-unit-norm
    elif init == 'pca':
        # ... fit PCA on a held-out batch of DINOv2 features
        W = pca_components(n=dim_out, fit_data=...)
    return W                                  # (384, 192), frozen

# 3. Forward path replacing LeWM encoder+projector:
@torch.no_grad()
def encode_dinov2(pixels):
    """pixels: (B, T, 3, 224, 224), normalized with ImageNet mean/std."""
    B, T = pixels.shape[:2]
    flat = pixels.reshape(B*T, 3, 224, 224)
    out = dinov2.forward_features(flat)        # {'x_norm_patchtokens': (B*T, 256, 384), 'x_norm_clstoken': (B*T, 384), ...}
    patch_tokens = out['x_norm_patchtokens']    # (B*T, 256, 384)
    pooled = patch_tokens.mean(dim=1)           # (B*T, 384)  — drop CLS, mean-pool patches
    z = pooled @ W_adapter                      # (B*T, 192)
    return z.reshape(B, T, 192)
```

**Data normalization change**:
```python
# Replace LeWM's default Push-T normalization with ImageNet stats:
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)
# applied AFTER the existing pixels-to-[0,1] step
```

**Train-time loop changes**:
- Skip `projector` and `pred_proj` (predictor input is already in 192-d adapter space; output stays in 192-d)
- Add `+encoder=dinov2` Hydra flag; pipeline detects it and swaps encode function + skips projector/pred_proj
- Optimizer trains: only `predictor` + `action_encoder` (adapter and DINOv2 frozen)

**Code module**: new `tier1_dinov2_adapter.py` — exposes `build_dinov2_world_model(rank=None, regularizer=None)` that returns a `JEPA`-shaped object using `encode_dinov2` + LeWM `predictor` + LeWM `action_encoder`. Tracker run IDs:

| Run ID | Variant | Override on top of base |
|---|---|---|
| R300 | adapter build (no training) | analysis: fit/save W_adapter; sanity-check encoded latent norm |
| R301 | dino_baseline | none |
| R302 | dino_uvlowr_r4 | `+predictor.ffn_rank=4` |
| R303 | dino_randdiff | `+loss.rand_diff.weight=0.01 +loss.rand_diff.eps=0.05 +loss.rand_diff.num_dirs=2` |

**Decision criteria**:
- **PASS (C_dyn generalizes)**: σ₁(dino_randdiff) < σ₁(dino_baseline) AND (|λ_max|(dino_randdiff) < |λ_max|(dino_baseline) OR ρ_mean(dino_randdiff) < ρ_mean(dino_baseline))
- **FAIL (LeWM-specific)**: σ₁(dino_randdiff) ≥ σ₁(dino_baseline) → C_dyn weakened to "LeWM-encoder-only"
- **CAVEAT for paper**: report both directions and magnitudes honestly; the cross-encoder match is the strongest evidence for the general claim

**Total cost**: 3 runs × 0.8h + 0.5h adapter setup = **3.4 GPU-h**. **Priority: NICE**.

---

## §12 Expected effect sizes vs noise (so we know if a result is real)

From v3 final-step measurements (n=2 seeds for frozen variants, except R002 which has 2 seeds via R002+R003):

| Quantity | Std across seeds (v3) | Expected v5 effect | Detectable? |
|---|---|---|---|
| σ₁(predictor) | ~0.06 | randdiff Δ = −0.44 (v3); λ-dose Δ should span ~0.3 across λ | YES (effect ≫ noise) |
| F²(predictor) | ~1.7 | randdiff Δ = −28 (v3); λ-dose Δ should span ~40 | YES |
| SR(predictor) | ~0.6 | randdiff Δ = +1.6 (v3); modest signal | marginal — need clean trends |
| rollout MSE h=10 | ~0.006 (baseline), ~0.16 (uvlowr ←HUGE) | randdiff Δ = +0.7 | YES for randdiff; **CAREFUL for uvlowr** (seed dependence) |
| \|λ_max\| (tier-1 DMD) | ~0.005 within-variant (tier-1 n=80 trajectories) | randdiff Δ = −0.1 (vs baseline 1.10) | YES (effect 20× noise) |
| Probing R² | ~0.022 (inline n=64); ~0.001 (eval n=256, frozen) | identical across frozen variants | mechanically constrained, no signal |
| latent PR | ~0.12 (across frozen seeds) | identical across frozen variants | mechanically constrained |

**Reviewer-safe effect sizes** (Δ > 4σ for ≥2-seed runs):
- σ₁ shift of ≥0.25
- F² shift of ≥7
- rollout MSE h=10 shift of ≥0.05 (baseline), ≥0.4 (uvlowr/randdiff)
- |λ_max| shift of ≥0.02 (tier-1 noise tiny → 0.05 threshold in P-Cdyn is generous)

The B4 λ-dose-response is the cleanest signal because it sweeps a single hyperparameter and looks for monotone migration of a robust effect-size-≫-noise quantity (|λ_max|). This is the most reviewer-defensible design choice in v5.
