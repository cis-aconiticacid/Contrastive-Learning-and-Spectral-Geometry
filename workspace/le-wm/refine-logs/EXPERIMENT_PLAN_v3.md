# Experiment Plan — Frozen-Encoder Ablation (LeWM low-rank predictor v3)

**Problem**: In joint JEPA training (v2 results), "predictor regularization" claims are confounded by encoder absorption: gradients from the regularizer flow into the encoder, which adapts its latent shape, so what we measured as "predictor Jacobian rank" is actually a property of the *composite* encoder+predictor system. We cannot causally isolate the predictor's intrinsic regularization effect without breaking the gradient path.

**Method thesis**: Freeze the encoder + projector + pred_proj after Stage-1 pretraining, then train only the predictor with each of the three regularizers. This **causally isolates** the predictor's intrinsic effect because the encoder can no longer compensate. Comparison with v2 joint-training data quantifies the encoder's contribution.

**Date**: 2026-05-11

**Budget**: 6-8 GPU-hours on AutoDL 4090

**Inputs**:
- v2 results (`/workspace/lewm_autodl_results_v2/`) — the joint-training comparison condition (already collected)
- Code: `/workspace/le-wm` (LeWM patched codebase + 5 callbacks)

**REVISIONS during execution (2026-05-11)**:
- Originally planned to reuse v2's `baseline_seed0_weights.ckpt` as the frozen Stage-1 encoder. **This was changed**: we trained a fresh 20K-step `stage1_baseline_seed0` because v2's ckpts were on a released AutoDL instance and not available locally.
- Stage-1 step count: planned 40K (matching v2); reduced to **20K** to fit the 8 GPU-hour budget after the instance had to be recreated 2× due to network/disk issues.
- Stage-2 step count: planned 10K; reduced to **8K** for same budget reason.
- M2 robustness check (NICE-TO-HAVE): not executed.
- See `/workspace/le-wm/refine-logs/EXPERIMENT_TRACKER.md` for actual values used.

---

## Claim Map

| Claim | Why it matters | Minimum convincing evidence | Block |
|---|---|---|---|
| **C2 causal**: in joint JEPA, the encoder absorbs predictor-side regularization | Reframes "predictor regularization" research — claims about predictor properties in joint training are confounded | (P1) Probing R² across 3 variants is mechanically identical (encoder is frozen) AND (P3) predictor-side spectral signatures (C5/C7) persist or amplify when encoder can't absorb | B1 |
| **C13 robust**: uvlowr r=4 matches data intrinsic dim (state-Δ=3.24, latent-Δ post-projector=4.06) and the architectural prior helps independently of encoder absorption | Distinguishes "lean LeWM" (architectural prior is real) from "encoder learned to exploit a low-rank predictor" (uvlowr win was emergent) | (P4) uvlowr's long-horizon rollout advantage either survives or disappears under frozen-encoder. Either outcome is informative. | B1 |
| **Anti-claim 1**: encoder absorption is negligible (joint vs frozen results are similar) | Would falsify C2 — predictor regularization is "direct" after all | All 4 P1-P4 predictions fail. Spectral signatures and rollout MSEs match v2 within seed-noise. | B1 |
| **Anti-claim 2**: frozen-encoder results are dominated by Stage-1 ckpt artifacts (not the regularizer) | Reviewer's first attack — "you just used one bad encoder" | (B2) Replicate one variant on a different Stage-1 ckpt (baseline_seed1) — outcome is consistent within seed noise | B2 |

**Note**: this experiment does NOT aim to prove uvlowr or randdiff is "better." It aims to **disambiguate** what part of the v2 ranking comes from the predictor vs. from encoder adaptation. Both positive and negative outcomes of P1-P4 are publishable as mechanism findings.

---

## Paper Storyline

**Main paper (if this becomes a paper) must prove:**
- C2 (encoder absorption is causal)
- Mechanism dichotomy (C5+C7) — supplemented by the frozen-encoder verification that the spectral signatures are predictor-intrinsic
- C13 alignment (uvlowr r=4 matches data dim) — including the falsification test of "does the alignment matter when encoder is frozen?"

**Appendix:**
- v2 joint-training tables (for direct comparison)
- C6 control (random / ImageNet / JEPA encoder Jacobian)
- C8 under-training reversal evidence
- C9 compound error growth rates
- C10 randdiff regularizer trajectory

**Cut (or future work):**
- λ sweep for randdiff (resolves "is λ=0.01 just a bad pick?")
- uvlowr rank sweep r∈{1,2,4,8,16,32} (resolves "is r=4 special or just lucky?")
- Push-T planning success rate (separate eval pipeline not built)
- Non-Push-T datasets (generalization beyond manipulator-block dynamics)

---

## Experiment Blocks

### Block 0 (M0): Sanity — frozen-encoder forward + backward correctness

- **Claim tested**: implementation is correct (encoder really is frozen; predictor really does learn; callbacks fire)
- **Why this block exists**: every prior incident (cudnn segfault, watcher race, eval_rollout horizons) cost half a day. A 10-minute sanity is cheap insurance.
- **Dataset / split**: Push-T expert, val split for callbacks; train split for fitting
- **Compared systems**: just `baseline-predictor-only` (no regularizer)
- **Metrics**: pred_loss (should decrease ≥30% from step 0 to 1K), probing R² @ step 0 and step 1K, grad norm (should be 0 on encoder + projector + pred_proj; nonzero on predictor)
- **Setup**: AutoDL 4090, batch 32, 1K steps, frozen-encoder mode on, single seed
- **Success criterion**: pred_loss drops noticeably; encoder param grad norms are exactly 0; callbacks all produce non-empty output files
- **Failure interpretation**: bug in freeze logic; do not proceed to M1 until fixed
- **Table/figure target**: not in paper; only diagnostic
- **Priority**: MUST-RUN. Estimated cost: 0.2 GPU-hours (1K steps × 0.5 s/step)

### Block 1 (M1): Main anchor — frozen-encoder ablation, 3 variants × 2 seeds × 10K steps

- **Claim tested**: C2 (encoder absorption causal), C5 + C7 + C13 carryover
- **Why this block exists**: this IS the experiment. The frozen-encoder runs let us check P1-P4 from CLAIM.md.
- **Dataset / split**: Push-T expert, identical to v2 train/val splits
- **Compared systems**:
  1. `frozen-baseline` (predictor-only, no regularizer) — control
  2. `frozen-uvlowr_r4` — architectural FFN rank-4
  3. `frozen-randdiff_λ0.01` — Scarvelis-Solomon penalty
  4. *Reference for comparison*: v2 results (joint-training, already collected)
- **Metrics** (decisive first):
  - **P1 metric**: probing R² (overall, 7-d state) — expected identical within ~0.005 across variants if C2 true; >0.02 spread if false
  - **P3 metric**: predictor Jacobian σ₁ + SR + F² (final step) — expected randdiff still shows σ₁ inflation + SR drop; uvlowr still ties baseline
  - **P4 metric**: rollout MSE at h∈{1,3,5,10,15,20} — uvlowr's long-horizon advantage either survives (→ architectural prior real) or disappears (→ encoder absorption explains v2 ranking)
  - **Secondary**: encoder Jacobian (must be IDENTICAL across variants — sanity), latent covariance SR, latent-delta intrinsic dim, training-time compute
- **Setup details**:
  - Stage-1 source: `/workspace/stablewm_home/baseline_seed0/baseline_seed0_weights.ckpt` (encoder + projector + pred_proj reused, `requires_grad=False`)
  - SIGReg loss DROPPED in Stage-2 (no gradient consumer)
  - Predictor: same arch as v2 (depth=6, heads=16, dim_head=64, mlp_dim=2048); `+predictor.ffn_rank=4` for uvlowr; `+loss.rand_diff.weight=0.01` for randdiff
  - Optimizer: AdamW lr=1e-3, betas=(0.9, 0.95), wd=0.01 (per v2)
  - Batch 32, 10K steps, 2 seeds {0, 1}
  - Callbacks: same 5 as v2 (Jacobian probe, Probing, LatentCov, EngineeringMetrics, PreClipGradNorm)
  - Probe interval: every 500 steps (same as v2)
  - Step 0 baseline measurement: must include (control point)
- **Success criterion**: each variant produces clean curves, eval.json with all 6 rollout horizons, final_analyses.json with weight SVD + encoder Jacobian + MLP probe
- **Failure interpretation**:
  - All P1-P4 fail → C2 is wrong; encoder doesn't absorb meaningfully. Reframe paper around "predictor regularization in JEPA: a direct effect" with v2 data as primary.
  - P1 passes but P3/P4 fail → encoder mechanically frozen but predictor's "intrinsic" effect is too weak to detect at this seed count → would need more seeds.
  - P1 passes, P3 passes, P4 mixed → encoder absorption explains part of v2 ranking; "lean LeWM" claim survives partially. Most interesting / likely outcome.
- **Table/figure target**: Table 2 (joint vs frozen rollout MSE side-by-side); Figure 2 (Jacobian spectral signatures in both regimes)
- **Priority**: MUST-RUN. Estimated cost: 6 runs × (10K steps × 0.5 s/step + ~0.3h analysis overhead) ≈ **6.0 GPU-hours**

### Block 2 (M2): Stage-1 robustness check — 1 variant on alternate ckpt

- **Claim tested**: anti-claim 2 — "results are dominated by which Stage-1 ckpt we picked"
- **Why this block exists**: a reviewer will ask "what if baseline_seed0 happened to be a bad encoder?" Need to show robustness.
- **Dataset / split**: same as M1
- **Compared systems**: one variant (e.g., `frozen-randdiff`) on Stage-1 from `baseline_seed1` instead of `seed0`. Should produce qualitatively the same result (P3 spectral signature replicates).
- **Metrics**: predictor Jacobian σ₁ + SR + F², rollout MSE at h∈{1,5,10}
- **Setup**: identical to M1 except Stage-1 source = baseline_seed1 ckpt
- **Success criterion**: σ₁ and SR for randdiff replicate (within seed noise) on the new Stage-1 ckpt → robustness confirmed
- **Failure interpretation**: results swing wildly → Stage-1 ckpt is a major confound; full paper requires multiple Stage-1 seeds
- **Table/figure target**: appendix sanity table
- **Priority**: NICE-TO-HAVE (gated on M1 outcome — only worth running if M1 is positive). Estimated cost: 1 run × 1 GPU-hour

### Block 3 (M3): Implementation polish + result aggregation

- **Claim tested**: none — execution discipline
- **Why this block exists**: clean numbers and aggregate CSVs ready for paper plotting
- **Tasks**:
  - Run `aggregate.py` on M1+M2 outputs
  - Build a single "joint vs frozen" comparison CSV (variant × condition × metric)
  - Compute v2-vs-frozen deltas for each P1-P4 prediction
  - Update CLAIM.md verdict block with "CONFIRMED / FALSIFIED / MIXED" for each P
- **Priority**: MUST-RUN (cheap, 0.1 GPU-hours)

---

## Run Order and Milestones

| Milestone | Goal | Runs | Decision gate | Cost (GPU-h) | Risk |
|---|---|---|---|---|---|
| **M0** | Sanity: encoder really frozen, predictor learns, callbacks fire | 1 run × 1K steps × 1 seed | If pred_loss not dropping ≥30% OR encoder grad norms nonzero → STOP, fix freeze logic | 0.2 | impl bug — high mitigation: assertions in freeze logic |
| **M1** | Frozen-encoder ablation (causal isolation of predictor) | 3 variants × 2 seeds × 10K steps = 6 runs | P1: ΔR² < 0.005 across variants? P3: spectral signatures persist? P4: uvlowr long-h advantage survives? | 6.0 | high seed variance on h≥15 from v2 — partial mitigation: 2 seeds is tight, may need to drop h=20 metric |
| **M2** | Robustness to Stage-1 source | 1 variant × 1 seed × 10K steps on baseline_seed1 Stage-1 | Conditional: run only if M1 P3 positive. If M2 spectral signature differs >50% from M1 → flag confound | 1.0 | low |
| **M3** | Aggregate + verdict | analysis only | none | 0.1 | low |
| **Total** | | 8 runs | | **7.3 GPU-h** | within budget |

---

## Compute and Data Budget

- **Total GPU-hours**: 7.3 (within 6-8h budget)
- **Data prep**: none (Push-T already on AutoDL at `/workspace/stablewm_home/pusht_expert_train.h5`)
- **Human eval**: none
- **Biggest bottleneck**: predictor-only training step time (~0.5 s/step) — vs joint training ~0.7 s/step. If actually 0.7 s/step, budget tightens to 10h; cut to 1 seed in M1 if needed.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Freeze-mode bug (encoder still gets gradients) | medium | high — invalidates entire experiment | M0 sanity asserts grad norm on encoder params == 0 |
| h=15/20 rollout MSE std too large with 2 seeds | high | medium — P4 verdict ambiguous | report h≤10 as primary, h=15/20 as supplementary; pre-register reliance on h=10 |
| Stage-1 ckpt artifacts dominate (anti-claim 2) | low (v2 baseline_seed0 was unremarkable) | high — single-ckpt result | M2 robustness check on baseline_seed1 |
| AutoDL instance interruption | medium | low — nohup + periodic SFTP pull | reuse v2 watcher pattern (`auto_pipeline.py` with pgrep precheck) |
| 10K steps insufficient (predictor still converging) | medium | medium — undertraining repeats v1→v2 reversal | M0 verifies pred_loss curve shape; if not converged, extend to 15K (eats into M2 budget) |
| BatchNorm in projector / pred_proj in eval mode drifts | medium | medium — eval-mode BN stats |  use `model.eval()` for frozen modules; verify BN running stats are fixed |

---

## Implementation Notes

What changes vs. v2 codebase:
1. **`train.py`**: add hydra flag `+freeze.encoder=true` that
   - loads encoder+projector+pred_proj state from a Stage-1 ckpt
   - sets `requires_grad=False` on those modules
   - calls `.eval()` on them (BN running stats frozen)
   - removes SIGReg loss from optimizer (no consumer)
   - filters optimizer parameters to predictor + action_encoder only
2. **`run_full.sh`**: new variant list `frozen-baseline / frozen-uvlowr_r4 / frozen-randdiff` with the freeze flag
3. **Callbacks**: no changes needed (they all read state via `pl_module.model`, which still has the frozen modules)
4. **`final_analyses.py`**: encoder Jacobian + per-weight SVD should still run; will show identical encoder Jacobian across variants (= sanity)
5. **`eval_rollout.py`**: no changes

**Assert in M0**: after loading + freezing,
```python
for name, p in pl_module.model.encoder.named_parameters():
    assert not p.requires_grad, f"encoder param {name} NOT frozen"
for name, p in pl_module.model.projector.named_parameters():
    assert not p.requires_grad, f"projector param {name} NOT frozen"
```

---

## Final Checklist

- [x] Main paper tables covered (joint vs frozen comparison, spectral signatures)
- [x] Novelty isolated (frozen encoder eliminates absorption pathway)
- [x] Simplicity defended (no extra components, just freeze + filter optimizer)
- [x] Frontier component justified — N/A, no LLM/VLM/Diffusion in this work
- [x] MUST-RUN vs NICE-TO-HAVE separated (M0/M1/M3 MUST; M2 conditional)
- [x] Failure interpretations specified for each P1-P4 outcome
- [x] Anti-claims listed (encoder absorption negligible; Stage-1 ckpt artifacts)
- [x] Within budget (7.3 / 8 GPU-hours)
