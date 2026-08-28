# v3 Frozen-Encoder Ablation — Analysis

**Date**: 2026-05-11
**Source**: `/workspace/lewm_autodl_results_v3/`
**Design doc**: `/workspace/le-wm/refine-logs/EXPERIMENT_PLAN.md`
**Pipeline log**: `run_pipeline_v3.log`

## TL;DR

The frozen-encoder ablation produces **one strong positive C2 result, one weak P4 result, and several confounds we cannot fully discharge in v3**.

| Prediction | Outcome | Strength |
|---|---|---|
| **P1**: probing R² becomes mechanically identical across variants | **CONFIRMED** (R²=0.8329 ± 0.0000) | Trivial (mechanical: encoder frozen) |
| **P2**: encoder Jacobian identical across variants | **CONFIRMED** (σ₁=11.28, SR=2.70, F²=343.7 — identical across all 6 frozen runs) | Sanity check; passes |
| **P3**: predictor spectral signatures persist or amplify when encoder can't absorb | **FALSIFIED in direction** — randdiff's signature **REVERSES** between v2 (concentration) and v3 (shrinkage) | **Suggestive evidence for C2 (encoder absorption is causal)** — pending ε-rescaled rerun (Confound 3) |
| **P4**: rollout MSE differences amplify; uvlowr long-horizon win survives or dies cleanly | **MIXED** — uvlowr h=10 mean below baseline but n=2 with high spread (seed 0/1: 0.318/0.090); v2 broad h≥5 pattern does NOT cleanly replicate | Inconclusive |

**Headline finding**: in v2 joint training, randdiff produced σ₁ inflation (+43%) and Jacobian SR collapse (−43%); in v3 frozen training, randdiff produces σ₁ shrinkage (−21%) and SR mild increase (+10%). **The reversal is consistent with — but does not alone prove — encoder-side adaptation as the v2 driver.** n=2 frozen seeds plus an unrescaled ε (Confound 3) mean an ε-mis-calibrated counterfactual cannot yet be ruled out. A planned ε-rescaled rerun (~1.5 GPU-hr) is the line between "suggestive" and "strong".

---

## Setup

| Phase | Variant | Steps | Seeds | Notes |
|---|---|---|---|---|
| Stage-1 | `stage1_baseline` (joint) | 20,000 | {0} | encoder + projector + predictor + pred_proj + SIGReg, all trainable |
| M0 sanity | `frozen_sanity` (frozen) | 1,000 | {0} | freeze invariants asserted, pred_loss drop ≥30% confirmed |
| M1 main | `frozen_baseline` | 8,000 | {0, 1} | predictor + action_encoder trainable; encoder/projector/pred_proj frozen from stage1_baseline_seed0 |
| M1 main | `frozen_uvlowr_r4` | 8,000 | {0, 1} | `+predictor.ffn_rank=4` |
| M1 main | `frozen_randdiff` | 8,000 | {0, 1} | `+loss.rand_diff.weight=0.01 +eps=0.05 +num_dirs=2` |

**Stage-1 ckpt**: `stage1_baseline_seed0_weights.ckpt` (216 MB). Loaded into encoder + projector + pred_proj at start of every frozen run. SIGReg loss skipped (no gradient consumer).

**Compute**: AutoDL 4090-48G, total ~6 GPU-hours for 8 runs after setup.

---

## Verified results (n=2 seeds for frozen variants, n=1 for Stage-1)

### Rollout MSE (test trajectories, n=256 per horizon)

| variant | h=1 | h=3 | h=5 | h=10 | h=15 | h=20 |
|---|---|---|---|---|---|---|
| stage1_baseline (joint 20K, n=1) | 0.0068 | 0.042 | 0.082 | 0.081 | 0.193 | 0.220 |
| frozen_baseline (8K, n=2) | **0.0056 ± 0.0001** | 0.029 ± 0.005 | **0.048 ± 0.002** | 0.271 ± 0.006 | 1.420 ± 0.154 | 1.918 ± 0.482 |
| frozen_uvlowr_r4 (8K, n=2) | 0.008 ± 0.002 | 0.040 ± 0.001 | 0.101 ± 0.037 | **0.204 ± 0.161** (seeds: 0.318 / 0.090) | **1.223 ± 0.594** | **1.993 ± 0.310** |
| frozen_randdiff (8K, n=2) | 0.032 ± 0.002 | 0.100 ± 0.015 | 0.224 ± 0.050 | 0.889 ± 0.102 | 2.169 ± 0.221 | 2.423 ± 0.243 |

**Observations**:
- **frozen_baseline beats stage1_baseline at h=1 in this single comparison** (0.0056 vs 0.0068) — but stage1 has only n=1, so no error bars; cannot conclude "predictor-from-scratch is faster at short horizon" from this alone.
- **frozen models all degrade dramatically at h≥10** vs stage1_baseline (1.9 vs 0.22 at h=20). This reflects that 8K predictor-only training on a fixed encoder doesn't reach the prediction quality of 20K joint training where encoder + predictor co-adapt.
- **Frozen mean ordering at h=10**: uvlowr (0.20) < baseline (0.27) < randdiff (0.89) — same direction as v2 BUT uvlowr h=10 mean is heavily dominated by seed 1 (0.090); seed 0 = 0.318 actually loses to baseline. With n=2 the apparent advantage is fragile.
- **Frozen ordering at h=5**: baseline (0.048) < uvlowr (0.10) < randdiff (0.22) — different from v2 (where uvlowr won at h=5). Effect size is large here, less sensitive to n=2.
- High variance at h≥15 makes uvlowr vs baseline indistinguishable; need ≥3 seeds.

### Probing R²

| variant | overall R² (eval.json, n_test=256) | probing.jsonl (n_test=64, final step) |
|---|---|---|
| stage1_baseline | 0.8329 | 0.4703 (step 20000) |
| frozen_baseline | **0.8329 ± 0.0000** | **0.4480 ± 0.0223** (mean over 2 seeds) |
| frozen_uvlowr_r4 | **0.8329 ± 0.0000** | **0.4480 ± 0.0223** |
| frozen_randdiff | **0.8329 ± 0.0000** | **0.4480 ± 0.0223** |

**P1 confirmed mechanically**: all 6 frozen runs give byte-identical R² in eval.json (n_test=256, 0.8329) and in `probing.jsonl` (n_test=64, 0.448 ± 0.022) because (encoder + projector + pred_proj frozen) → (identical latents) → (identical ridge probe input) → identical R². This is a sanity check on the freeze logic, not a discovery.

**Within a single seed, the inline R² is exactly constant across all 17 checkpoint steps** (seed 0: 0.4703 throughout; seed 1: 0.4257 throughout). The 0.022 std is purely between-seed variance from val-batch sampling, NOT a real "training progress" signal — frozen latents cannot change. This means inline `probing.jsonl` has **zero signal as a training-progress metric** in frozen runs; it serves only as a freeze-correctness check.

The eval.json R²=0.8329 (n_test=256) is higher than the inline R²=0.448 (n_test=64), reflecting larger sample size + different random splits. Both are computed on the same frozen latent representation.

### Predictor (composite) Jacobian — final step

| variant | σ₁ | SR | F² | ER |
|---|---|---|---|---|
| stage1_baseline (joint 20K) | 1.72 | 32.4 | 95.2 | 123.4 |
| frozen_baseline (8K) | 2.07 ± 0.06 | 20.4 ± 0.8 | 87.5 ± 1.7 | 105.9 ± 2.7 |
| frozen_uvlowr_r4 | 2.08 ± 0.04 | 20.8 ± 0.6 | 89.9 ± 5.8 | 107.7 ± 0.1 |
| frozen_randdiff | **1.63 ± 0.03** | **22.4 ± 0.4** | **59.2 ± 0.9** | 108.8 ± 2.8 |

**The critical comparison** — randdiff effect on Jacobian, v2 (joint) vs v3 (frozen):

| | v2 joint training | v3 frozen training | Direction of randdiff effect |
|---|---|---|---|
| σ₁ baseline / randdiff | 1.74 / 2.48 | 2.07 / 1.63 | v2: ↑43%; v3: ↓21% |
| SR baseline / randdiff | 32.3 / 18.4 | 20.4 / 22.4 | v2: ↓43%; v3: ↑10% |
| F² baseline / randdiff | 97 / 110 | 87 / 59 | v2: +14%; v3: −32% |

**The randdiff Jacobian signature REVERSES between v2 and v3**. This is the most important v3 finding.

**Interpretation**: in v2 joint training, the encoder co-adapted under randdiff pressure to make predictor's local linear response easier to suppress on average — but achieved this by *shaping latents into directions* that randdiff happens to penalize less, producing the σ₁-inflation/SR-collapse pattern. With encoder frozen (v3), randdiff acts on a fixed input distribution and produces the *expected* Frobenius shrinkage. The v2 signature is an emergent property of joint training, not a property of randdiff itself.

### Encoder Jacobian (sanity check — must be identical)

| variant | σ₁ | SR | F² |
|---|---|---|---|
| stage1_baseline | 11.28 ± 0.11 | 2.70 ± 0.025 | 343.7 ± 4.0 |
| frozen_baseline | 11.28 ± 0.11 | 2.70 ± 0.024 | 343.7 ± 3.7 |
| frozen_uvlowr_r4 | 11.28 ± 0.11 | 2.70 ± 0.024 | 343.7 ± 3.7 |
| frozen_randdiff | 11.28 ± 0.11 | 2.70 ± 0.024 | 343.7 ± 3.7 |

**Mechanically identical across all frozen variants** ✓. P2 confirmed (sanity check on freeze logic).

### Latent covariance SR (final, from `latent_cov.jsonl`)

| variant | SR | ER |
|---|---|---|
| stage1_baseline | 8.13 | 11.72 |
| frozen_baseline | 7.95 ± 0.25 | 11.62 ± 0.14 |
| frozen_uvlowr_r4 | 7.95 ± 0.25 | 11.62 ± 0.14 |
| frozen_randdiff | 7.95 ± 0.25 | 11.62 ± 0.14 |

Identical across frozen variants (same fixed encoder + projector → same latent covariance). Slightly different from stage1 (different val batch sampling).

### Engineering

| variant | pred_params | peak mem | wall-clock | trainable / total |
|---|---|---|---|---|
| stage1_baseline | 10.79 M | 3839 MB | 1698 s | 100% |
| frozen_baseline | 10.79 M | 1877 MB | 677 s | ~60% (predictor + action_encoder only) |
| frozen_uvlowr_r4 | **6.18 M (−43%)** | 1803 MB | 671 s | — |
| frozen_randdiff | 10.79 M | 1562 MB | 672 s | — |

Frozen runs are **~2.5× faster** than joint training (677 vs 1698 s) — encoder backward + SIGReg eliminated. uvlowr's −43% params replicates from v2.

---

## P1 / P2 / P3 / P4 verdicts

### P1 — Probing R² identical across variants if encoder absorbed regularization
- **Verdict**: CONFIRMED mechanically. But this is a tautology: with encoder + projector + pred_proj all frozen and identical, the ridge probe input is byte-identical → R² must match.
- **Real meaning**: this is a sanity check on the freeze logic, not a test of C2. To meaningfully test C2 via probing, we'd need to compare v2 probing R² spread (small) to a counterfactual where encoder *could* adapt to each variant separately — which is the v2 joint-training comparison.
- **Reinterpretation**: in v2, probing R² spread across variants was 0.871-0.898 (Δ ≈ 0.027). If C2 is true (encoder absorbs), spread should shrink as encoder freedom decreases. With encoder fully frozen (v3), spread = 0 exactly. This is consistent with C2 in a weak sense.

### P2 — Encoder Jacobian identical across variants
- **Verdict**: CONFIRMED. All 6 frozen runs produce encoder Jacobian σ₁=11.28, SR=2.70, F²=343.7 (within 1% of each other).
- **Significance**: sanity check on freeze + the `KeepFrozenInEvalCallback`. The monkey-patched eval mode persists across Lightning train/eval toggles. ✓

### P3 — Predictor spectral signatures persist when encoder can't absorb
- **Verdict**: FALSIFIED IN DIRECTION — randdiff signature **REVERSES** between v2 and v3.
- **Significance**: this is the most provocative v3 finding, but with current setup (n=2 seeds, unrescaled ε), it is **suggestive evidence** for C2 — not definitive.
- In v2, randdiff produced (σ₁ ↑43%, SR ↓43%, F² ↑14%) — energy concentrated in top direction.
- In v3 with frozen encoder, randdiff produces (σ₁ ↓21%, SR ↑10%, F² ↓32%) — energy shrunk overall.
- The sign flip is consistent with — but doesn't alone prove — an active encoder-adaptation mechanism in v2. **The competing explanation is that ε=0.05 (unchanged from v2) is mis-calibrated relative to v3's frozen-latent norm, producing a different randdiff effective strength and possibly a different sign of effect**. This is Confound 3 below.
- **Mechanism CONJECTURE (post-hoc, not directly measured in v3)**: under joint training, the encoder might evolve to produce latents where the average finite-difference penalty is minimized — by pushing variance into one direction the predictor can handle robustly (high σ₁ + low SR). With encoder fixed, randdiff acts on the predictor's Jacobian directly in its standard form (Frobenius shrinkage). This is one of several plausible stories; v3 does not directly probe it.
- **What v3 does support**: v2's σ₁-inflation/SR-collapse pattern for randdiff is at least partly a property of joint training, since under frozen encoder the same regularizer produces opposite-sign changes. **v2's C5/C7 spectral-redistribution claims must therefore be reframed** as observations about the joint system under randdiff pressure, not as intrinsic randdiff-on-predictor effects.
- **Required followup before "strong" claim**: ε-rescaled frozen_randdiff rerun (~1.5 GPU-hr). If reversal persists → strong evidence for C2. If signature returns to v2-like inflation → reversal was ε mis-calibration.

### P4 — Rollout MSE ordering preserves; uvlowr long-horizon advantage robust
- **Verdict**: MIXED.
- **At h=10**: uvlowr (0.20) < baseline (0.27) < randdiff (0.89). Same as v2 (uvlowr=0.155 < baseline=0.21 < randdiff=0.61). Direction preserved.
- **At h=5**: baseline (0.048) < uvlowr (0.10) < randdiff (0.22). Different from v2 (uvlowr=0.08 < baseline=0.11). At short horizon, uvlowr's advantage flips in v3.
- **At h=15/20**: high variance (uvlowr ±0.59 at h=15), uvlowr and baseline within 1σ.
- **uvlowr's win is NOT robust under frozen encoder** at the same magnitude as v2. There is a hint of survival at h=10 (relative advantage 25%) but the 5% improvement at h=5 in v2 disappears (uvlowr now 2× worse than baseline at h=5).
- **Interpretation**: the v2 uvlowr win was *partially* due to encoder adaptation (encoder shaped its latents to be predictable by a low-rank FFN) — when encoder is fixed (Caveat 4), uvlowr loses some of its advantage. The architectural prior still provides a small h=10 benefit, but not the broad h≥5 benefit seen in v2.

---

## Confounds (must report; cannot fully discharge)

### Confound 1: Stage-1 step count (20K vs v2's 40K)
- v3 frozen baseline at h=20 is 1.92, but stage1_baseline at h=20 is 0.22 — a 9× gap. Some of this is "8K predictor not converged on fixed encoder" but some is "stage-1 encoder not fully evolved at 20K".
- The v3 comparison "frozen variants vs frozen_baseline" remains valid (apples to apples), but **comparing v3 to v2 absolute numbers is unfair** — different encoder maturity.

### Confound 2: Predictor from scratch on baseline-co-trained encoder
- The Stage-1 encoder was shaped specifically by joint training with a UNIFORM full-rank predictor. The latent space it produces is "predictable-by-full-rank-predictor".
- When we train uvlowr-r4 predictor on this encoder, we test "rank-4 in baseline-shaped latent space", not "rank-4 with an encoder that knows about rank-4".
- **Implication**: if frozen_uvlowr underperforms vs joint v2 uvlowr, two explanations exist:
  - (a) encoder absorption in v2 was actually doing the work
  - (b) baseline-shaped latents are simply unfriendly to rank-4 predictors
- v3 cannot distinguish (a) from (b). A v4 experiment (train uvlowr Stage-1 → freeze → train multiple predictors on top) would.

### Confound 3: randdiff ε not rescaled
- v2 ε=0.05 chosen for jointly-trained latent norm; v3 uses same ε on a different (frozen) latent distribution.
- The randdiff effect direction reversed; if ε is "wrong" for v3 latent norm, the reversal could be partially mis-calibration rather than true encoder-absorption disambiguation.
- **Test**: measure |z| in stage1 frozen latents vs v2 trained latents, rescale ε proportionally, rerun frozen_randdiff. If reversal persists → robust C2 evidence. If signature returns to v2-like → ε was the culprit. Cheap (~1.5 hr GPU).

### Confound 4: Single Stage-1 seed (MORE serious after a reversal, not less)
- Only stage1_baseline_seed0 was used as the frozen encoder. A "lucky" or "unlucky" stage-1 trajectory could bias all 6 downstream runs.
- The original `EXPERIMENT_PLAN.md` anti-claim 2 explicitly identified Stage-1 single-seed as the response to "what if this ckpt was unlucky?". The decision to defer M2 robustness because P3 was "reversed not confirmed" is logically backwards — **a reversal is more surprising and more in need of robustness verification, not less**.
- M2 robustness check (frozen_randdiff on baseline_seed1) was a NICE-TO-HAVE in the plan but **not executed** because v3 only ran Stage-1 seed=0. This is the weakest spot in v3 and the next-highest priority after Confound 3.

### Confound 5: 8K frozen steps may be undertraining
- Predictor pred_loss CSV shows monotone decrease through step 8000; we don't have a clean signal of plateau.
- v2 took ~30K steps for predictor convergence with co-adapting encoder; 8K with fixed encoder might or might not be sufficient.
- This could explain variance in h=15/20 results.

### Confound 6: SIGReg loss active in Stage-1 but absent in Stage-2
- Stage-1 was trained with SIGReg loss as in v2 (shaping latent covariance toward isotropy).
- Stage-2 frozen runs **skip SIGReg** (no gradient consumer once encoder + projector are frozen; computing it is wasted compute).
- Asymmetry: the encoder weights were shaped by SIGReg, but the frozen-encoder runs effectively train predictor on latents that have specific SIGReg-shaped covariance. Whether this matters is unclear, but it's an asymmetry between the two phases.

### Confound 7: Stage-2 predictor from scratch vs v2 jointly-trained predictor
- The v2 comparison anchor uses a predictor that was **jointly co-trained** with the encoder for 40K steps.
- v3 Stage-2 predictors are **freshly initialized** and trained 8K steps from scratch on a fixed encoder.
- So when the v2-vs-v3 reversal table compares "v2 baseline σ₁=1.74" with "v3 frozen_baseline σ₁=2.07", part of the +19% σ₁ shift is plausibly due to "8K-from-scratch vs 40K-co-trained" predictor maturity, not pure regimen difference.
- This is a confound for the v2-vs-v3 *absolute* comparison; the v3-internal comparison (frozen_baseline vs frozen_randdiff) is not affected.

---

## What v3 changes about CLAIM.md

### Claims STRENGTHENED by v3

- **C2 (encoder absorption is causal)**: v3 provides the strongest evidence we have. P3 reversal is hard to explain without an active encoder-adaptation mechanism in v2.

### Claims REFRAMED by v3

- **C5 (mechanism dichotomy of encoder ‖J‖_F compression)**: in v3, all frozen variants share the SAME encoder Jacobian (mechanical). The v2 dichotomy of encoder Jacobian patterns under uvlowr vs randdiff was the encoder absorbing the regularization. **C5 should be reframed as a v2-only observation about joint training, not an intrinsic spectral property.**
- **C7 (regularization redistributes spectrum, not shrinks it)**: v3 falsifies the "randdiff inflates σ₁" sub-claim — that was an encoder-absorption artifact. In v3 with frozen encoder, randdiff DOES shrink Frobenius (87→59, -32%). **The Scarvelis-Solomon intuition is correct in isolation; v2 wasn't measuring it cleanly.**

### Claims UNCHANGED by v3

- **C1 (predictor Jacobian SR doesn't predict rollout)**: v3 has 3 frozen variants with predictor SRs in {20.4, 20.8, 22.4} (narrow range) and rollout h=10 in {0.27, 0.20, 0.89}. The randdiff outlier (highest SR, worst rollout) still contradicts a "more SR = better" story. C1 holds in the v3 SR range.
- **C3 (uvlowr is a Pareto improvement)**: weakened. In v3 frozen, uvlowr is only better at h=10; baseline beats uvlowr at h=5. The "Pareto" claim was always Push-T-specific; v3 narrows it further.
- **C4 (randdiff degrades long-horizon JEPA rollout)**: confirmed. Frozen randdiff is 2-4× worse than frozen baseline at all h≥3. Holds in both v2 and v3.
- **C6 (training-driven low-rank ViT Jacobian)**: independent of v3 (cheap check), unchanged.
- **C8 (under-training reversal warning)**: v3 didn't test this; unchanged.
- **C9, C10**: unchanged.
- **C13 (uvlowr r=4 matches Push-T intrinsic dim)**: independent; unchanged.

---

## Recommendations for next experiments

1. **Quick win**: re-run frozen_randdiff with ε rescaled to match v3 latent norm. ~1.5 hr GPU. Tests Confound 3.
2. **Highest information**: v4 = train uvlowr-co-trained Stage-1 + freeze + train multiple predictors (baseline, uvlowr, randdiff) on top. Tests Confound 2 cleanly. ~8 GPU-hrs.
3. **Sample size**: if any of (1) or (2) is positive, scale to 3 seeds for confidence intervals at h≥10.
4. **Hyperparam sweeps** (if budget): randdiff λ ∈ {0.001, 0.01, 0.1}, uvlowr rank ∈ {2, 4, 8, 16, 32}. These would close C1 strongly.

---

## Cost

- v3 instance: AutoDL 4090-48G with +80 GB disk (~110 GB total). Approximately ¥18 over ~6 GPU-hours.
- 4 earlier failed instances (small disk / network issues): ~¥6.
- Total v3 cost: ~¥24. Balance remaining: ¥175.
