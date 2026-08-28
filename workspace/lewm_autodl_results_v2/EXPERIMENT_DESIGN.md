# Experiment Design — LeWM Low-Rank Predictor (40 K-step run)

**Status**: Retrospective. Written after the run completed. Documents what was
*actually* designed (implicitly), maps each hypothesis to the metric meant to test
it, calls out design gaps revealed post-hoc.

---

## Research question

> Given that the JEPA-style LeWM predictor's effective Jacobian rank during
> training is much lower than its nominal latent dimension (Maes et al. 2026
> setup: 192-d × history 3 = 576-d Jacobian), can we impose a low-rank prior on
> the predictor without hurting downstream quality? Two routes:
>
> 1. **Architectural (`uvlowr_r4`)**: factor each FFN linear in the predictor
>    as `Linear(in, r) ∘ Linear(r, out)` with `r=4` (LoRA-style).
> 2. **Regularizer (`randdiff λ=0.01`)**: add the Scarvelis & Solomon (2024)
>    one-sided finite-difference penalty on `‖J_predictor‖_*` to the loss.
>
> Compare against unmodified LeWM (`baseline`).

The implicit headline test: **Does adding the constraint cost prediction
quality, and does it actually make the Jacobian low-rank in the way intended?**

---

## Hypotheses

Each hypothesis has: a one-line statement, the primary metric, a falsification
criterion (the observation that would refute it), and the *predicted outcome*
before the run.

### H1. uvlowr efficiency

**Hypothesis**: `uvlowr_r4` matches baseline on validation prediction loss
with substantially fewer predictor parameters.

| field | value |
|-------|-------|
| primary metric | `final val pred_loss`, predictor param count |
| falsification | `val_pred(uvlowr) > val_pred(baseline) + 2σ_cross_seed` |
| predicted outcome | TIE (within 1 σ) on val_pred, ~−40 % params |

### H2. uvlowr rollout stability

**Hypothesis**: `uvlowr_r4` matches baseline on multi-step rollout MSE at all
tested horizons; if anything, it might be better at long horizons (the
original "low-rank dynamics" pitch).

| field | value |
|-------|-------|
| primary metric | rollout MSE at `h ∈ {1,3,5,10,15,20}` |
| falsification | uvlowr MSE > baseline MSE + 2 σ at any horizon |
| predicted outcome | TIE at all horizons |

### H3. uvlowr Jacobian rank propagation

**Hypothesis**: constraining the FFN to rank 4 propagates to a low-rank
predictor Jacobian.

| field | value |
|-------|-------|
| primary metric | predictor Jacobian stable rank at step 40 000 |
| falsification | `sr(uvlowr) ≥ sr(baseline) − 2 σ` |
| predicted outcome | Pre-run we expected uvlowr_sr ≪ baseline_sr. Pre-experiment 5 K run had already **failed** this (sr both ≈ 50), so the design's true purpose here was to confirm the 5 K finding at scale. |

### H4. randdiff Jacobian compression

**Hypothesis**: `randdiff` actively drives the predictor Jacobian rank down
relative to baseline.

| field | value |
|-------|-------|
| primary metric | predictor Jacobian stable rank trajectory + final value |
| falsification | `sr(randdiff) ≥ sr(baseline) − 2 σ` at step 20 000+ |
| predicted outcome | randdiff sr significantly lower than baseline (penalty design |
|                   |                                                  successfully driving rank down) |

### H5. randdiff preserves prediction quality

**Hypothesis**: `randdiff` at λ=0.01 imposes the Jacobian-rank constraint
*without* sacrificing prediction quality.

| field | value |
|-------|-------|
| primary metric | val pred_loss, rollout MSE @ h=10 |
| falsification | randdiff val_pred > baseline + 2σ **or** rollout @h=10 > baseline + 2σ |
| predicted outcome | TIE on pred_loss; possibly BETTER on rollout at long horizons (if low-rank Jacobian helps compounding) |

### H6. randdiff improves probing (representation quality)

**Hypothesis**: by constraining the predictor's Jacobian, `randdiff` forces the
encoder to produce a more "physically structured" latent, improving probe R².

| field | value |
|-------|-------|
| primary metric | Ridge probe R² overall + per-dim (especially velocity dims) |
| falsification | randdiff Ridge R² ≤ baseline Ridge R² (within noise) |
| predicted outcome | randdiff > baseline on probe, especially velocity dims (carrying forward the 5 K-step observation that randdiff vy probe was best). |

### H7. Jacobian rank ↔ rollout link

**Hypothesis**: lower predictor Jacobian rank → better long-horizon rollout
(the original "low-rank dynamics stabilize rollout" intuition).

| field | value |
|-------|-------|
| primary metric | monotonicity check across 3 variants: `(sr, MSE@h=20)` |
| falsification | rank ordering of sr is reverse of rank ordering of MSE |
| predicted outcome | monotonic positive: lower sr → lower MSE |

---

## Design choices

### Independent variables (the things we vary)

| variable | levels | rationale |
|----------|--------|-----------|
| **predictor variant** | `{baseline, uvlowr_r4 (r=4), randdiff (λ=0.01, ε=0.05, num_dirs=2)}` | Two intervention types + control |
| **seed** | `{0, 1, 2}` | Cross-seed std for noise floor |

### Fixed parameters (held identical across all 9 runs)

| parameter | value | reason |
|-----------|-------|--------|
| dataset | Push-T expert (`pusht_expert_train.h5`, 1.98 M frames) | only LeWM dataset readily available at small enough size |
| LeWM model | default config: ViT-Tiny encoder (192-d), embed_dim=192, predictor depth=6/heads=16/dim_head=64/mlp_dim=2048, history_size=3, frameskip=5 | match paper |
| optimizer | AdamW lr=5e-5, wd=1e-3, grad clip val=1.0, linear-warmup-cosine sched | from `config/train/lewm.yaml` |
| batch | 32 (paper default 128 — reduced for instance VRAM headroom) | constraint, not choice |
| training length | 40 000 steps (~ 0.26 epoch on 1.98 M frames at batch=32) | budget vs paper's 100 epochs (~12 M steps) |
| precision | bf16 mixed | default |
| num_workers | 4 with `persistent_workers=True`, `pin_memory=True` | reproducibility, throughput |

### Dependent variables (what we measure)

| metric | source | freq | purpose |
|--------|--------|------|---------|
| total / pred / SIGReg loss | training step | every step | training dynamics |
| pre-clip grad norm | `on_after_backward` | every step | optimization stability |
| predictor Jacobian SR / ER / spectral norm | `JacobianProbeCallback` (6 samples) | every 500 steps + step 0 | **primary signal** for H3, H4, H7 |
| Ridge probe R² (n_train=256, n_test=64) inline | `ProbingCallback` | every 500 steps + step 0 | **trajectory of** representation quality (noisy version) |
| encoder latent covariance eigenvalues (n=256) | `LatentCovCallback` | every 500 steps + step 0 | encoder bottleneck dynamics |
| Multi-step rollout MSE at h={1,3,5,10,15,20} | `eval_rollout.py` on 256 held-out trajectories | once at end | **primary signal** for H2, H5, H7 |
| Ridge probe R² (n_train=1024, n_test=256) at end | `eval_rollout.py` | once at end | **final** representation quality |
| Per-weight SVD on all predictor Linears | `final_analyses.py` | once at end | mechanistic: where rank lives |
| Encoder Jacobian SVD (input → 192-d, 4 samples) | `final_analyses.py` | once at end | encoder compression |
| MLP probe R² (2 hidden layers) vs Ridge | `final_analyses.py` | once at end | linear vs nonlinear info |
| peak GPU memory, wall-clock, param count | `EngineeringMetricsCallback` | once at end | engineering cost |

### Controls

- **Untrained baseline (step 0)** for every per-step metric (Jacobian, probe, latent cov).
  Step 0 is a control for "random encoder + random predictor → how informative
  is the latent inherently?"
- **Same data ordering** across variants (Lightning's `random_split` with same
  generator seed `cfg.seed`).
- **Same SIGReg / loss weighting** across all variants — only the
  predictor-internal architecture (uvlowr) or extra loss term (randdiff) differ.
- **Same prober** (Ridge α=1.0 / MLP hidden=256, AdamW lr=1e-3, 200 epochs) for
  the post-hoc probing — eliminates "different probe" as a confounder.

### Confounders identified

| confounder | mitigation | residual risk |
|-----------|------------|---------------|
| AutoDL instance throughput drift over 14 h run | metrics are step-indexed not time-indexed; we compare same-step quantities | negligible for scientific claims; only wall-clock comparisons affected |
| randdiff has 2 extra forward passes per step | comparing at **same step count** means randdiff sees fewer effective optimizer updates per wall-second, but our comparison fixes step count, so the model has seen the same # of training examples | randdiff costs 23 % more wall-clock; this is a real cost, not a confounder |
| BatchNorm in projector zero-inits → low latent norm at step 0 | step-0 control already reflects this | step 0 cov measurements are degenerate but H1–H7 don't depend on step 0 latent geometry |
| stale partial randdiff_seed0 data (during the disk-conflict incident, see ANALYSIS §3 caveat below) | randdiff_seed0 was re-run from scratch, replacing all data; no carry-over | none |
| only 1 rank value (uvlowr r=4), 1 λ value (randdiff λ=0.01) | acknowledged limitation, not in scope | results are conditional on these choices being reasonable |

---

## Verdict mapping (predictions vs results)

| H  | Predicted | Observed | Verdict | Notes |
|----|-----------|----------|---------|-------|
| H1 | TIE val_pred + −40 % params | val_pred 0.017 vs 0.015 (within 1 σ), params −43 % | **CONFIRMED** | within margin |
| H2 | TIE rollout at all h | h=1: uvlowr WORSE; h=5/10: uvlowr BETTER; h≥15: indistinguishable | **MIXED** | h=10 is cleanest: uvlowr beats baseline ~30 %, both std ~30 % of mean. Statistically *not* a tie, but high variance |
| H3 | uvlowr sr ≪ baseline sr | 28.6 ± 2.9 vs 32.3 ± 3.2 — within 2 σ | **FALSIFIED** | confirms 5 K finding: FFN constraint does NOT propagate to full-predictor Jacobian |
| H4 | randdiff sr ≪ baseline sr | 18.4 ± 3.8 vs 32.3 ± 3.2 — ~4 σ apart | **CONFIRMED** | strong compression effect, especially after step 5 K |
| H5 | TIE on quality | val_pred 43 % worse; h=10 rollout 3× worse | **FALSIFIED** | randdiff at λ=0.01 is a clear handicap |
| H6 | randdiff probe > baseline | 0.871 vs 0.898 (within 1 σ, but randdiff trending lower) | **FALSIFIED** | reverses the 5 K observation, which was apparently transient |
| H7 | lower sr → lower MSE | sr: rand 18 < uvlowr 29 < base 32 ; MSE@h=10: uvlowr 0.16 < base 0.21 < rand 0.61 | **FALSIFIED** | rank ordering of sr does NOT predict rank ordering of MSE; randdiff has lowest sr and worst MSE |

### Reading the verdicts

- **uvlowr_r4 passes H1 and is mixed on H2**: a near-free −43 % parameter
  reduction. The "low-rank dynamics" mechanism (H3) it was sold as does not
  exist, but the lean-predictor outcome (H1) does. This re-frames uvlowr as a
  *parameter-efficiency* result, not a *dynamics* result.

- **randdiff fails H5, H6, H7** despite passing H4: it successfully imposes the
  intended rank constraint, but the constraint is costly without producing the
  hoped-for benefits. At λ=0.01 it's strictly a handicap.

- **H7 is the critical conceptual falsification**: lower predictor Jacobian
  rank does NOT correlate with better rollout. This invalidates the original
  pitch ("predictor Jacobian low rank → stable long-horizon rollout") that
  motivated investigating low-rank constraints in the first place.

---

## Design limitations / what we should have done

These are gaps **identified retrospectively**. None of them invalidate the
above; they constrain the conclusions.

### A. Only one rank value for uvlowr

Tested `r=4` only. Cannot say:
- whether r=4 is optimal (maybe r=8 strictly dominates baseline?)
- whether there's a rank-vs-perf knee
- whether r=1 still works (and what the lower limit is)

**Should have done**: r ∈ {1, 2, 4, 8, 16, 32} sweep with 1-2 seeds each.
Approximate cost: 6 × ~62 min ≈ 6.2 h, ~¥20 on AutoDL.

### B. Only one λ value for randdiff

Tested λ=0.01 only. The toy pilot's λ sweep showed `0.001 ≈ 0` and `λ ≥ 0.01`
hurts; but real LeWM might have a different operating point.

**Should have done**: λ ∈ {0.001, 0.003, 0.005, 0.01, 0.03} sweep.
Approximate cost: 5 × ~70 min ≈ 5.8 h, ~¥18.

### C. No power analysis

Seed std at h ≥ 15 ≈ mean. With 3 seeds we cannot resolve differences <2×.
Differences at h ≤ 10 *are* resolvable.

**Should have done**: pilot 1-2 seeds at h=20 to estimate σ, then commit to
appropriate seed count for full sweep. Or evaluate on more rollout
trajectories per seed (currently 256; bumping to 1024 reduces sampling
variance by 2×).

### D. Single dataset

Only Push-T. Cannot generalize the rank-cost trade-offs to other dynamics
(TwoRoom is grid-world-like; OGBench-Cube is 3-D manipulation).

**Should have done**: at minimum repeat on TwoRoom (small, same-format dataset). Cost: another ~14 h on AutoDL, ~¥40.

### E. No planning-success metric (the real downstream metric)

Rollout MSE is a *proxy* for "good world model". The LeWM paper actually
evaluates on Push-T planning success rate over 100 episodes. We never measured
this — it requires the `swm.World` env stack and a CEM/MPC wrapper.

**Should have done**: implement planning-success eval on the released
checkpoints. The Push-T env is now set up (we did this for installation).
Mostly a wrapper-and-CEM-integration task, not a re-train.

### F. No hyperparameter audit

We took LeWM's defaults (lr=5e-5, wd=1e-3, sigreg weight=0.09) and assumed
they're optimal for all three variants. uvlowr (smaller predictor) might want
higher lr; randdiff (extra penalty) might want different sigreg balance. Not
investigated.

### G. The randdiff_seed0 disk-conflict incident

During the 40 K run, a watcher-relaunch bug caused randdiff_seed0 to be killed
and restarted (see incident in conversation history). It was re-trained from
scratch with a fresh seed → no partial-data contamination. Final files are
clean. But this is a **process risk** we should plan around for future runs
(make pipelines truly idempotent w.r.t. relaunch).

---

## What "good" experiment design would have looked like

If I were doing this again, the design doc would be written *before* the run,
with:

1. **Pre-registered hypotheses**, each with a single primary falsification
   criterion (not "we'll see what the data says").
2. **Power calculation**: pilot 1 seed to estimate σ, then choose
   `n_seeds` to resolve the predicted effect size.
3. **Ablation matrix in advance**: rank sweep × λ sweep × at least 2 datasets,
   with clearly delineated "main result" vs "ablation".
4. **Downstream metric (planning success)** integrated into the pipeline.
5. **Pre-specified analysis**: write the analysis code (and even the figure
   layout) *before* the data lands, so the analyst can't pattern-match to
   noise.

For the next round (uvlowr rank sweep) I'll do this properly.
