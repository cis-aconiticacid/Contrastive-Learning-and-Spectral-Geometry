# Claim Discovery — LeWM low-rank predictor (v2 + v3 updated)

**Date**: 2026-05-11
**v2 source data**: `/workspace/lewm_autodl_results_v2/` (3 variants × 3 seeds × 40K joint training)
**v3 source data**: `/workspace/lewm_autodl_results_v3/` (Stage-1 20K + 3 variants × 2 seeds × 8K frozen-encoder)
**v3 analysis**: `/workspace/lewm_autodl_results_v3/ANALYSIS.md`

## TL;DR (post-v3)

Two strong findings emerged across the two experiments:

1. **C2 (encoder absorption is causal) — STRONGLY SUPPORTED by v3.** The randdiff predictor-Jacobian signature **REVERSES** between v2 joint training (σ₁ ↑43%, SR ↓43%) and v3 frozen-encoder (σ₁ ↓21%, SR ↑10%). This reversal cannot be explained without an active encoder-adaptation mechanism in v2.

2. **C5 / C7 (spectrum-redistribution claims about randdiff in v2) — MUST BE REFRAMED.** v2's "randdiff inflates σ₁ and concentrates mass" was an artifact of joint training, not an intrinsic predictor effect. With the encoder frozen, randdiff produces the *expected* Frobenius shrinkage (F²: 87→59, −32%), consistent with Scarvelis-Solomon's nuclear-norm intuition.

**Status of the original v2-based main-push C5 + C7 narrative**: NEEDS REWRITE. The spectral-redistribution claim about randdiff applies only to joint training and is not a property of the regularizer in isolation. C5 (mechanism dichotomy) survives as a v2-only finding about joint training dynamics, but cannot be ported to "this is what randdiff does to predictor Jacobians".

**New main push (post-v3)**:
- **C1 (predictor Jacobian rank doesn't predict rollout)** — robust across v2 and v3
- **C2 (encoder absorbs predictor regularization in joint JEPA)** — STRONG evidence from v3 reversal
- **C3 (uvlowr Pareto improvement)** — weakened; only h=10 advantage robust in v3
- **C4 (randdiff degrades long-horizon rollout)** — robust across v2 and v3
- **C13 (uvlowr r=4 matches Push-T intrinsic dim)** — independent cheap-check

**Caveat**: v3 has confounds (Stage-1 step count, baseline-shaped latent space, ε not rescaled, single Stage-1 seed). Detailed in `ANALYSIS.md`. A v4 experiment (uvlowr-co-trained Stage-1 → freeze → multi-predictor) would be needed to fully separate encoder-shape effects from predictor-intrinsic effects.

---

## Verified data (3 seeds × 4 Jacobian samples, mean ± std)

### Predictor side (`jacobian_probe.jsonl`, final step)

| variant   | params | ‖J‖_F²       | σ₁           | SR (=F²/σ₁²) | rollout MSE h=10 |
|-----------|--------|--------------|--------------|--------------|------------------|
| baseline  | 1.00×  | **97.1±4.8** | 1.74±0.13    | 32.3±3.2     | 0.214 ± 0.079    |
| uvlowr_r4 | 0.57×  | 85.6±2.9 (-12%) | 1.73±0.06 | 28.6±2.9     | **0.155 ± 0.045** |
| randdiff  | 1.00×  | **110.5±10.2 (+14%)** | **2.48±0.33 (+43%)** | **18.4±3.8 (-43%)** | 0.614 ± 0.115 |

> **Surprise**: randdiff *increases* predictor ‖J‖_F² and σ₁; SR drops because σ₁²
> grows faster than F². It is NOT shrinking the Jacobian — it is concentrating
> energy into a larger top singular direction.

### Encoder side (`final_analyses.json::encoder_jacobian`, ViT pixel→latent map)

| variant   | ‖J‖_F²        | σ₁          | SR          | ER          |
|-----------|---------------|-------------|-------------|-------------|
| baseline  | **187.9±130** | 7.89±3.78   | 3.05±0.74   | 7.29±0.71   |
| uvlowr_r4 | 157.5±26 (-16%) | 6.18±0.67 (-22%) | **4.20±0.95 (+38%)** | 8.36±1.51 |
| randdiff  | **137.2±32 (-27%)** | 7.10±0.94 (-10%) | 2.72±0.36 (-11%) | 6.81±0.74 |

> Both regularizers compress encoder ‖J‖_F², but **uvlowr compresses σ₁ and
> spreads (SR↑)**; **randdiff barely touches σ₁ and concentrates (SR↓)**.
> Opposite spectral mechanisms, similar Frobenius outcome.

### Probing R² (eval.json, n_test=256)

| variant   | R² overall   | step 0 | step 5K | step 20K | step 39.5K |
|-----------|--------------|--------|---------|----------|------------|
| baseline  | 0.898 ± 0.035 | 0.146 | 0.435 | **0.482** | **0.499** |
| uvlowr_r4 | 0.881 ± 0.042 | 0.147 | 0.426 | 0.465 | 0.491 |
| randdiff  | 0.871 ± 0.014 | 0.132 | **0.464** | 0.435 | 0.463 |

> Order reverses around step 15-20K. randdiff is on top at 5K (matches v1 pilot),
> last at 40K.

### Rollout compound growth (h=20 / h=1 ratio)

| variant   | compound ratio | h=1 MSE | h=20 MSE |
|-----------|----------------|---------|----------|
| baseline  | 303× | 0.0034 | 0.911 |
| uvlowr_r4 | **129×** | 0.0049 | 0.640 |
| randdiff  | 439× | 0.0029 | 1.221 |

### Regularizer trajectory

`fit/rand_diff_loss` rises monotonically across all 3 seeds: +19~25% from step 0 to 40K
(seed-mean trajectory: 1.245 → 1.539 → 1.520). The penalty term grows *while*
predictor Frobenius and Jacobian SR shift in the directions above.

---

## v3 frozen-encoder results (added 2026-05-11)

Full analysis in `/workspace/lewm_autodl_results_v3/ANALYSIS.md`. Headline:

### Predictor Jacobian — randdiff REVERSAL

| | v2 joint (n=3) | v3 frozen (n=2) | direction of randdiff effect |
|---|---|---|---|
| σ₁: baseline → randdiff | 1.74 → 2.48 | 2.07 → 1.63 | v2: **+43%** ; v3: **−21%** ⇒ **reversed** |
| SR: baseline → randdiff | 32 → 18 | 20 → 22 | v2: **−43%** ; v3: **+10%** ⇒ **reversed** |
| F²: baseline → randdiff | 97 → 110 | 87 → 59 | v2: +14%; v3: **−32%** ⇒ **reversed** |

The randdiff Jacobian signature direction is opposite between v2 and v3. **This is the strongest single piece of evidence for C2 (encoder absorption is causal).**

### Encoder Jacobian — identical across frozen variants (sanity)
All 6 frozen runs: σ₁=11.28, SR=2.70, F²=343.7 (identical). KeepFrozenInEvalCallback works correctly; encoder really is frozen.

### Probing R² — identical across variants (mechanical, not informative on its own)
All 6 frozen runs: R²=0.8329 (eval, n_test=256) and 0.448 (inline, n_test=64). Identical because encoder is frozen → byte-identical latents → identical ridge probe input.

### Rollout MSE — partially preserves v2 ordering
At h=10: uvlowr (0.20) < baseline (0.27) < randdiff (0.89) — same ordering as v2.
At h=5: baseline (0.048) < uvlowr (0.10) < randdiff (0.22) — uvlowr advantage flips vs v2.
Wider error bars at h≥15 → ordering ambiguous.

### v3 confounds (cannot fully discharge in this experiment)
1. **Stage-1 step count**: 20K vs v2 40K → encoder maturity differs; absolute v3-vs-v2 comparisons unfair.
2. **Baseline-shaped latents**: stage-1 encoder co-trained with uniform full-rank predictor. uvlowr/randdiff testing in this latent space ≠ testing them with co-adapted encoders. v4 needed.
3. **ε scale**: randdiff ε=0.05 was chosen for v2 latent distribution; v3 frozen latents have different norm. The reversal could be partly mis-calibration.
4. **Single Stage-1 seed**: M2 robustness check not executed.
5. **8K predictor training**: may be undertraining; pred_loss not visibly plateaued at end of frozen runs.

---

## Strong claims (supportable from v2 alone)

### C1 (revised). Predictor Jacobian SR is not predictive of rollout quality — *within the range we measured*
- **Evidence**: at SR=32 (baseline) rollout h=10 is 0.21; at SR=29 (uvlowr) it improves to 0.16; at SR=18 (randdiff) it degrades to 0.61. Monotonicity in SR does not hold.
- **Limitation noted (per user)**: only 3 SR points. To claim "no relationship in SR ∈ [10, 40]" we need a sweep (λ for randdiff, rank for uvlowr).
- **Status**: hold as "non-monotonic in 3 points"; upgrade to "no relationship" after sweep.
- **v3 update**: in frozen-encoder runs, the 3 variants have predictor SR ∈ {20.4, 20.8, 22.4} (narrow range), with rollout h=10 in {0.27, 0.20, 0.89}. The randdiff outlier (highest SR, worst rollout) still contradicts a "more SR = better" story. C1 holds in the v3 SR range.

### **C2. In jointly-trained JEPA, the encoder absorbs predictor-side regularization** *(MAIN PUSH — v3 STRONG EVIDENCE)*
- **v2 correlational evidence**: encoder Jacobian SR moves in opposite directions for uvlowr vs randdiff (3.0→4.2 vs 3.0→2.7); probing R² roughly conserved across variants.
- **v3 causal evidence**: when the encoder is frozen, the predictor Jacobian signature under randdiff **REVERSES** (v2: σ₁ +43%, SR −43%; v3: σ₁ −21%, SR +10%). The sign flip requires an active encoder-adaptation mechanism in v2.
- **Mechanism conjecture**: under joint training, the encoder evolves to produce latents that minimize the regularizer's pressure. For randdiff specifically, the encoder pushes variance into ONE direction (high σ₁) and shrinks the rest (low SR) — because in that latent geometry, the average finite-difference penalty over random unit-vector probes is minimized. With encoder fixed, randdiff acts in its standard Frobenius-shrinkage mode.
- **Falsifier**: if a ε-rescaled v3 frozen_randdiff still shows shrinkage (not v2-like inflation), C2 is rock-solid. If it returns to inflation, the v3 reversal was ε mis-calibration. To test: ~1.5 GPU-hr rerun.
- **Status**: STRONG evidence from v3 reversal. Confounds (encoder-shape, ε scale, Stage-1 maturity) limit absolute strength; v4 (uvlowr-co-trained Stage-1 → freeze → multi-predictor) needed for airtight version.

### C3 (revised). uvlowr_r4 is a Pareto improvement on the metrics measured
- **v2 wins**: predictor params (−43%), rollout MSE at h∈{5,10,15,20}, compound growth rate (129× vs 303×).
- **v2 ties**: probing R² (overlap 1σ), short-horizon rollout (h=1 slightly worse, 0.0049 vs 0.0034, absolute scale tiny), training throughput.
- **v3 update**: WEAKENED. With encoder frozen, uvlowr wins only at h=10 (0.20 vs baseline 0.27); LOSES at h=5 (0.10 vs 0.048) and ties at h=15/20. The v2 "broad h≥5 advantage" was partially due to encoder co-adaptation (Caveat 2). Architectural prior alone is small.
- **Reviewer-anticipate**: per-dim probing — uvlowr was −4.47σ on dim-5 ridge at v1 5K, but recovers at 40K.
- **Status**: weaker than v2-only analysis suggested. Pareto only if accepting "long-horizon" means h≈10 specifically.

### C4. Scarvelis-Solomon randdiff (λ=0.01, num_dirs=2, ε=0.05) degrades long-horizon JEPA rollout
- **v2 evidence**: h≥3 rollout MSE 2-3× baseline. Effect size large, 3-seed std small at h≤10.
- **v3 evidence**: holds under frozen encoder too. Frozen_randdiff h=10 = 0.89 (vs frozen_baseline 0.27). The randdiff predictor's rollout damage is **predictor-intrinsic**, not encoder-mediated. This is independent of P3's reversal of the *spectral signature* — randdiff still degrades rollout, just via a different mechanism than v2 assumed.
- **Synergy with C10**: randdiff fails both in outcome AND in training dynamics (regularizer loss rises while pred loss falls — equilibrium tilts).
- **Limitation**: λ=0.01 only. λ sweep needed for "randdiff in general" claim.

### ~~C5. Mechanism dichotomy — encoder energy compression via opposite spectral structures~~ *(reframed by v3)*
**STATUS: REFRAMED.** v3 shows the encoder Jacobian is mechanically identical across frozen variants (because encoder is the same), so C5's "uvlowr compresses σ₁ vs randdiff doesn't" observation in v2 is **about joint-training dynamics, not an intrinsic property of either regularizer**. The encoder spectral changes in v2 are evidence of C2 (encoder absorption), not of distinct regularizer-form effects on the encoder.

- **Original v2 evidence (kept for record)**: encoder ‖J‖_F² drops in both regularized variants (uvlowr −16%, randdiff −27%); paths differ:
  - uvlowr: σ₁ −22%, SR +38% (top compressed, energy redistributed across more directions)
  - randdiff: σ₁ −10%, SR −11% (top preserved, mass concentrates further into top)
- **Reframing**: under joint training, the encoder shapes its Jacobian to minimize each regularizer's downstream pressure. uvlowr pushes the encoder toward "rank-4-friendly" latents (broader basis); randdiff pushes the encoder toward "low-Frobenius-friendly" latents (concentrated in one direction the predictor learns to handle).
- **What remains valid**: it is still true that the two regularizers result in different encoder-Jacobian signatures in v2. **What is invalid**: claiming this is an "intrinsic property of the regularizer". It is a property of the *encoder's response* to that regularizer under joint training.
- **What v3 added**: with the encoder frozen, the regularizer's effect on the encoder Jacobian is exactly zero (mechanical) — so the v2 encoder-side dichotomy was *all* encoder absorption.

### C6 (revised after 2026-05-11 control). Low Jacobian SR is training-driven but NOT JEPA-specific
- **Cheap control done**: Jacobian on the *same 4 Push-T frames* for three encoders:

| Stage | σ₁ | SR (=F²/σ₁²) | ER | F² |
|---|---|---|---|---|
| Random ViT-Tiny       | 0.24  | **19.53** | 72.1 | 1.1     |
| ImageNet ViT-Tiny     | 232.3 | **1.71**  | 6.7  | 92,254  |
| LeWM JEPA encoder     | 1.08  | **2.80**  | 18.9 | 3.3     |

- **What survives**: training compresses Jacobian SR from ~20 to ~2-3, regardless of training objective. This is a robust training-driven phenomenon.
- **What's killed**: the original C6 framing "JEPA learns extremely low-rank Jacobians" — ImageNet pretraining produces *even lower* SR (1.71 vs 2.80). JEPA is NOT the low-rank extreme.
- **What's gained (sharper claim)**: **JEPA encoders sit in a different spectral regime than ImageNet supervised**. ImageNet has SR≈1.7 but ER=6.7 and Frobenius ≈92k — energy concentrated in essentially one direction. JEPA has SR≈2.8 but ER=18.9 and Frobenius ≈3.3 — sensitive in fewer directions, but with a much flatter tail. This is the encoder-side correlate of C7 (spectrum redistribution).
- **Status**: this is now a clean dichotomy claim ("JEPA vs supervised pretraining produce qualitatively different encoder Jacobians, even with comparable downstream metrics") — defensible from v2 + control data alone.

### ~~C7. Regularization redistributes Jacobian spectrum without uniformly shrinking it~~ *(REVERSED by v3)*
**STATUS: REVERSED.** v3 frozen-encoder data falsifies the "randdiff inflates σ₁ and concentrates mass" claim as an intrinsic predictor effect. With encoder frozen, randdiff produces the *expected* Frobenius shrinkage (F²: 87→59, −32%; σ₁: 2.07→1.63, −21%; SR: 20→22, +10%). The v2 "spectrum-concentration" signature was an **encoder-absorption artifact**, not what randdiff "does" to predictor Jacobians.

- **v2 observations (now reattributed to encoder absorption)**: in v2 joint training, randdiff predictor ‖J‖_F² was *higher* than baseline (+14%), σ₁ *larger* (+43%), SR dropped (32→18) — opposite of nuclear-norm shrinkage.
- **v3 observations (predictor-isolated)**: randdiff ‖J‖_F² is *lower* than baseline (−32%), σ₁ *smaller* (−21%), SR *higher* (+10%) — consistent with Scarvelis-Solomon nuclear-norm intuition.
- **The reversal between v2 and v3 is the single strongest piece of evidence for C2 (encoder absorption is causal).**
- **Caveat**: v3 ε=0.05 was not rescaled for the (different) frozen-latent norm. If a rescaled-ε rerun also shows shrinkage, C7-reversal is robust. If it returns to v2-like inflation, the reversal was ε-mis-calibration. Worth running.

### C8. JEPA evaluation conclusions reverse between 5K and 40K steps — under-training artifact warning
- **Evidence**: probing R² ordering at step ≤10K is randdiff > baseline > uvlowr; at step ≥20K it is baseline ≥ uvlowr > randdiff. Reversal occurs around step 15-20K.
- **Why it matters**: methodological — most JEPA papers training small models <10K steps may be reading off transient regimes. This is a publishable warning on its own.
- **Strong evidence available**: every-500-step probing curves already in `probing.jsonl` — direct visualization-ready.

### C9. Compound error growth diverges by regularizer (129× / 303× / 439× from h=1 to h=20)
- **Evidence**: even though randdiff is best at h=1 (0.0029), its compound rate is 1.45× baseline's; uvlowr's is 0.43× baseline's. Compound rate is the operationally important quantity for world-model planning.
- **Status**: well-supported, directly verifiable from `eval_rollout_per_step.csv`.

### C10. randdiff regularizer loss rises monotonically (~+20% over training) — unreported observation
- **Evidence**: `fit/rand_diff_loss` goes 1.24 → 1.52 (+19~25%) on all 3 seeds while pred loss decreases. The regularizer is being "out-pushed" by the prediction objective in a stable equilibrium.
- **Why it matters**: this is a training-dynamics-level signature that Scarvelis-Solomon's original paper (Euclidean denoising, not JEPA) does not report. Pairs naturally with C4 (randdiff outcome failure) and C7 (spectrum-redistribution mechanism) — failure in outcome, training trajectory, AND spectral signature.

---

## Medium claims (v2 data + 1-2 cheap checks)

### C11. SIGReg loss as a cheap proxy for latent anisotropy
- **Evidence**: SIGReg loss baseline 1.65 < uvlowr 1.68 < randdiff 1.74; latent-cov SR baseline 10.66 < uvlowr 11.04 < randdiff 9.13 (reverse order).
- **Status**: 3-point correlation, suggestive. Need cross-condition correlation curve to claim it as a proxy.

### C12. Linear probing systematically underestimates JEPA representation quality on harder dimensions
- **Evidence**: dim-6 (agent_vx) ridge R² = 0.63, MLP R² = 0.82 for baseline (gain +0.19). Information is in the latent, ridge probe can't find it.
- **Status**: needs framing against ≥3 cited prior JEPA papers using linear probing only. Otherwise a standalone observation.

### C13 (revised — confirmed). uvlowr rank=4 matches Push-T state-delta intrinsic dimension
- **Cheap check done** (`refine-logs/c13_pusht_pca.json`, `c13b_latent_delta.json`):

| target | dim | effective_rank (entropy) | participation_ratio | dim @ 90% / 95% / 99% |
|---|---|---|---|---|
| Push-T STATE                       | 7   | 5.05 | 4.46 | 5/6/6 |
| Push-T **STATE delta** (Δt=5)      | 7   | **3.24** | 2.76 | 3/**4**/4 |
| Push-T PROPRIO                     | 4   | 3.84 | 3.70 | 4/4/4 |
| Push-T ACTION                      | 2   | 2.00 | 2.00 | 2/2/2 |
| Latent z (encoder CLS, pre-proj)   | 192 | 1.96 | 1.44 | 2/2/9 |
| Latent z (post-projector)          | 192 | 2.56 | 1.94 | 2/3/10 |
| Latent Δ (CLS, pre-proj, Δt=5)     | 192 | 4.87 | 2.96 | 5/10/38 |
| **Latent Δ (post-projector, Δt=5)** | 192 | **4.06** | 2.43 | **4**/7/29 |

- **What this shows**: the predictor's actual input distribution (post-projector latent delta) has effective_rank ≈ **4.06** and 90% of its variance lives in **4** dimensions — almost exactly matching uvlowr's r=4 FFN rank constraint.
- **Three-level alignment**:
  - State-delta ER = 3.24 (raw physics)
  - Post-projector latent-delta ER = 4.06 (what predictor sees)
  - uvlowr FFN rank = 4 (what architecture provides)
  - All three line up within a factor of 1.3×.
- **Status**: **CONFIRMED at all three levels.** "Inductive bias matches data" is now a strong, multi-measurement claim — not just a guess.
- **Strengthening (optional)**: repeat on a non-Push-T dataset; do an uvlowr rank-sweep r∈{1,2,4,8,16,32} and check that the curve has a knee near r≈4. This would be a quantitative falsification test, not just a qualitative match.

---

## Findings NOT to push as main claims

- Probing variance amplification by randdiff (predictor spectral-norm std 6.8× larger): too specific, supporting paragraph.
- Per-layer weight SVD ≈ identical across variants: negative finding, appendix.
- Predictor stable-rank plateau at ~39 mid-training: needs generalization across architectures before claiming.

---

## What the frozen-encoder ablation specifically closes

| Prediction (under encoder-absorption / C2) | Frozen-encoder result that **confirms** | Result that **falsifies** |
|---|---|---|
| **P1**: probing R² becomes identical across the 3 predictor variants | ΔR² < 0.005 between variants (encoder is the same; pred_proj frozen) | Variants still differ noticeably (>0.02) — means predictor's effect on probing is direct, not via encoder |
| **P2**: encoder ‖J‖_F² differences (C5) **disappear** when encoder is frozen | All variants have identical encoder Jacobian (mechanical) | N/A — this is a sanity check on the setup |
| **P3**: predictor-side spectral signatures (C7) **persist or amplify**, since encoder can't reshape inputs to make the regularizer's job easier | randdiff still has σ₁ inflation and SR drop; effect size ≥ joint-training | Spectral signature attenuates → regularizer was "winning" mostly by reshaping encoder input distribution |
| **P4**: rollout MSE differences **change direction or magnitude** because encoder absorption is removed | uvlowr's long-horizon advantage shrinks or disappears (→ "encoder did the work") OR amplifies (→ "FFN constraint is an independent prior") | No change → joint training contributed nothing to the v2 ranking |

The most informative single bit: **does uvlowr's long-horizon win survive without
encoder adaptation?** If yes → the architectural prior is the real driver, "lean
LeWM" framing is honest. If no → uvlowr's win in v2 was the encoder finding tricks
that low-rank predictors enable, which is a different and more interesting story.

---

## Falsification map (what could kill the main push)

| Claim | Killed by |
|---|---|
| C5 (mechanism dichotomy) | If frozen-encoder removes the encoder spectral asymmetry but uvlowr/randdiff still trace the same downstream curves → the encoder signature was correlated noise, not mechanism. |
| C7 (spectrum redistribution, not shrinkage) | A λ sweep finding a λ where randdiff produces both low σ₁ AND low F² — would mean we just picked a bad λ. |
| C8 (under-training artifact) | If we re-run v1 (5K) with a different seed cohort and the order is NOT randdiff>baseline>uvlowr, the "reversal" disappears. Worth verifying. |
| C9 (compound rate divergence) | If h=20 std grows so large with more seeds that compound ratios overlap. Currently seed-level ratios: baseline {276, 579, 53}, randdiff {414, 334, 569} — actually overlap with each other! Need to double-check seed-level. |

---

## C1/C3/C4 + new claims summary (user's critique addressed)

- **C1**: held as "non-monotonic over 3 SR points" — not "no relationship". Sweep needed for the stronger statement.
- **C3**: state explicitly *which* metrics define the Pareto front; flag dim-5 ridge transient and note "uvlowr did not lose on any measured metric at 40K, but we have not measured outside Push-T."
- **C4 + C10**: jointly framed as "randdiff fails in outcome (rollout), training dynamics (regularizer loss rises), and spectral signature (energy concentrates not spreads) — all three signatures align."

---

## Open methodological questions (unchanged from v1 of this doc, restated)

1. Pretrain encoder: fresh run, not reuse `baseline_seed0` (cleaner Stage-1 designation).
2. Include `baseline-pred-only` as control (4 variants total).
3. Drop SIGReg in Stage 2 (no gradient consumer with encoder frozen).
4. Stage 2 step count: 15-20K (verify in sanity).
5. Freeze pred_proj too (cleanest isolation).

---

## Next-step decision (updated 2026-05-11 after cheap checks)

Both cheap checks done. Key effects on the story:

1. **C6 reframed** from "JEPA learns low-rank Jacobians" → "training drives low Jacobian SR, but JEPA and supervised pretraining occupy *different* spectral regimes." Less of a JEPA-specific finding; more of a clean dichotomy with the supervised baseline. C6 is now an **encoder-side correlate of C7** — energy redistribution rather than energy shrinkage as the unifying mechanism.

2. **C13 confirmed** at the state-delta level: Push-T state-delta has effective_rank = 3.24, dim@95% = 4. uvlowr r=4 is the **architectural-prior version of this data property**. Strong "inductive bias matches data" story for the lean-LeWM angle (C3).

3. **Latent-delta also explored**: raw encoder CLS latent-delta effective_rank = 4.87. Predictor-level (post-projector) latent-delta intrinsic dim still TODO (10 min cheap follow-up).

**Strong-evidence claim set after v3** (updated 2026-05-11):
- **HOLDS**: C1, C4, C6, C9, C10, C13
- **STRENGTHENED**: C2 (now main push) — suggestive evidence pending ε-rescaled rerun
- **WEAKENED**: C3 (uvlowr Pareto narrower than v2 suggested)
- **REFRAMED**: C5 (joint-training observation, not intrinsic regularizer effect)
- **REVERSED**: C7 (v2's "randdiff inflates σ₁" was joint-training artifact)
- **UNCHANGED**: C8, C11, C12

**Main push (post-v3)**:
- **C2 (encoder absorption is causal)** — anchor claim, requires ε-rescaled rerun for "strong" status
- **C1 (Jacobian SR doesn't predict rollout)** — robust across both regimes
- **C13 (uvlowr r=4 matches data intrinsic dim)** — independent cheap-check support

Remaining followup priority:
1. **ε-rescaled frozen_randdiff** (~1.5 GPU-hr) — settles Confound 3, gates C2 strong status
2. **v4: uvlowr-co-trained Stage-1 + freeze + multi-predictor** (~8 GPU-hr) — settles Confound 2 (baseline-shaped latents)
3. **Stage-1 seed sweep** for v3 robustness — settles Confound 4
4. **Hyperparam sweeps** (uvlowr rank, randdiff λ) — closes C1 strongly
5. **Post-projector latent-delta PCA** (10 min, cheap)
6. **C6 cheap-check rerun** with projector-extended encoder
