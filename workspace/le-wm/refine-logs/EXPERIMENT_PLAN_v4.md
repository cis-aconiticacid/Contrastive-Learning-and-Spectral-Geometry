# Experiment Plan v4 — Settling the v3 Confounds (LeWM low-rank predictor)

**Problem**: v3 frozen-encoder ablation produced one strong directional finding (randdiff predictor-Jacobian signature **reverses** between v2 joint training and v3 frozen training — suggestive evidence for C2: encoder absorption is causal). But the reversal sits on top of three open confounds:
  - **Confound 2** — baseline-shaped Stage-1 latents (the frozen encoder was co-trained with a uniform predictor, so uvlowr/randdiff were never given a latent space "shaped for them")
  - **Confound 3** — randdiff ε=0.05 not rescaled to v3's (different) frozen-latent norm, so the reversal could be partial ε mis-calibration
  - **Confound 4** — single Stage-1 seed, single Stage-1 architecture (no Stage-1 robustness check)

Plus a long-pending **C13/C3 sharpening**: uvlowr's r=4 win was claimed to match Push-T's intrinsic dim (~3-4), but we have never tested a rank-sweep, so r=4 vs r=8 vs r=16 could just be lucky.

**Method thesis (v4)**: Run three targeted ablations — (a) ε-sensitivity sweep over the v3 frozen Stage-1 ckpt, (b) **uvlowr-co-trained Stage-1** followed by a 3-variant freeze ablation, (c) uvlowr **rank-sweep** on the v3 baseline Stage-1 — that together discharge Confounds 2/3 and quantitatively close C13.

**Date**: 2026-05-12

**Budget**: 12-13 GPU-hours on AutoDL 4090-48G (~¥40), well within remaining ¥175.32 balance.

**Inputs**:
- v3 Stage-1 ckpt: `/workspace/lewm_autodl_results_v3/stage1_baseline_seed0_weights.ckpt` (216 MB, reuse)
- v2 reference data: `/workspace/lewm_autodl_results_v2/` (40K joint training, 3 variants × 3 seeds)
- v3 reference data: `/workspace/lewm_autodl_results_v3/`
- Code: `/workspace/le-wm` (already patched with freeze + KeepFrozenInEvalCallback from v3)

---

## Claim Map

| Claim | Why it matters | Minimum convincing evidence | Linked Block |
|---|---|---|---|
| **C2** (encoder absorption is causal in joint JEPA) — main push, currently "suggestive" | Reframes any "predictor regularization" result in JEPA; central paper claim | The v3 σ₁/SR/F² reversal survives both an ε-sweep (no v2-like inflation at any tested ε) AND a different Stage-1 (uvlowr-co-trained encoder produces same direction of randdiff Jacobian shrinkage) | B1, B2 |
| **C13** (uvlowr r=4 matches Push-T intrinsic dim ≈ 4) — currently 3-point alignment, no falsification | Anchors the "lean LeWM" framing in C3; a knee at r≈4 would be a clean predictive falsification of "the gain is just from parameter cut" | A rank-sweep r∈{1,2,4,8,16} (one Stage-1 condition, frozen-encoder) shows a knee/plateau in rollout MSE near r=4, NOT a smooth monotone curve | B3 |
| **Anti-claim A** (the v3 reversal is ε mis-calibration) | Falsifies C2; reviewer's first attack | One ε value in the sweep reproduces v2-like σ₁ inflation (+30% or more) and SR collapse | B1 |
| **Anti-claim B** (the v3 reversal depended on baseline-shaped Stage-1) | Falsifies the "C2 is general" reading; would mean we found a Stage-1-specific artifact | uvlowr-co-trained Stage-1 produces randdiff Jacobian behavior that matches v2 joint training, not v3 frozen | B2 |
| **Anti-claim C** (uvlowr's r=4 is "lucky", any low rank works) | Falsifies C13 alignment story | Rollout MSE is smooth in r (no knee), or knee sits at r≠4 in a way unrelated to data intrinsic dim | B3 |

**Note**: B1 and B2 discharge different confounds for the same claim (C2). Both need to land before we can call C2 "strong". B3 is independent and closes C13.

---

## Paper Storyline (post-v4)

**Main paper must prove:**
- **C2** as the anchor claim — supported by joint vs frozen reversal, robust to ε across a sweep, robust to Stage-1 source across two encoder shapes
- **C13** — uvlowr r=4 sits at a knee in rollout MSE, matching Push-T's intrinsic dim ≈ 4 (state-delta ER=3.24, post-projector latent-delta ER=4.06)
- Mechanism dichotomy (C5 reframed): joint training is what produces the encoder-side spectral asymmetry; the same regularizers acting on a fixed encoder don't reshape the latent space

**Appendix:**
- v2 joint-training tables (joint-condition reference)
- v3 frozen-encoder tables (frozen-condition reference, baseline-shaped Stage-1)
- C6 control (random / ImageNet / JEPA encoder Jacobian dichotomy)
- C8 under-training reversal evidence (v1 5K → v2 40K reversal)
- C9 compound error growth rates
- C10 randdiff regularizer trajectory

**Cut (or future work):**
- λ sweep for randdiff (deferred — C2/C13 closure has priority)
- Push-T planning success rate (separate eval pipeline build, not budget-justified at this stage)
- Non-Push-T datasets (out of scope, future paper)
- Stage-1 seed sweep on the v3 baseline ckpt (Confound 4 is the lowest-priority remaining one; B2's different-shaped Stage-1 already gives one robustness data point)

---

## Experiment Blocks

### Block 0 (M0): ε-norm calibration probe + setup sanity (cheap)

- **Claim tested**: implementation correctness; choice of ε sweep bracket
- **Why this block exists**: before running B1, we need to know the latent-norm scale on the v3 Stage-1 frozen encoder so the ε sweep brackets are sensible. Also re-verifies freeze callback fires after env rebuild.
- **Dataset / split**: Push-T expert, val split (256 samples)
- **Steps**:
  1. Load v3 Stage-1 ckpt, freeze encoder + projector + pred_proj
  2. Compute latent z-norm distribution (mean, p10, p50, p90) on val batch
  3. Compare to v2 latent z-norm (load one v2 baseline_seed0 ckpt if available; else use the cached v2 latent stats from `/workspace/lewm_autodl_results_v2/*/final_analyses.json` if they were logged)
  4. Choose ε sweep bracket: 4 values spanning `[ε_v2 · norm_v3 / norm_v2 / 3, ε_v2 · norm_v3 / norm_v2 · 3]`, plus original ε=0.05 as one of the 4 if it falls in the bracket
  5. Run 1K-step `frozen_sanity` to re-verify freeze logic on the new instance
- **Success criterion**: latent norm computed; 4 ε values picked; sanity passes (pred_loss drop ≥30%, encoder grad norm = 0)
- **Failure interpretation**: if v2 latent norms cannot be recovered, fall back to a fixed geometric sweep ε ∈ {0.01, 0.025, 0.05, 0.1, 0.2}
- **Cost**: 0.3 GPU-hours (mostly the sanity rerun)
- **Priority**: MUST-RUN

### Block 1 (B1, M1): ε-sensitivity sweep for frozen randdiff — discharges Confound 3

- **Claim tested**: the v3 randdiff predictor-Jacobian reversal is NOT due to ε mis-calibration
- **Why this block exists**: the single most-cited Anti-claim A. A clean ε sweep that nowhere reproduces v2-like σ₁ inflation kills it.
- **Dataset / split**: Push-T expert, identical train/val to v3
- **Compared systems**:
  - 4 frozen runs at the chosen ε values (from M0 calibration), one seed each
  - For comparison: v3 baseline `frozen_baseline_seed0` and `frozen_randdiff_seed0` (already collected, no rerun)
- **Metrics** (decisive first):
  - **Predictor Jacobian σ₁, SR, F²** at final step — *the* sign-flip indicator
  - **Regularizer loss trajectory** (`fit/rand_diff_loss`) — verifies the penalty actually fires at each ε
  - **Rollout MSE at h∈{1,5,10}** — secondary, contextualizes how the spectral signature couples to behavior
- **Setup details**:
  - Stage-1 source: v3 `stage1_baseline_seed0_weights.ckpt`
  - 8K predictor-only steps, seed=0, batch 32 (same as v3 M1)
  - SIGReg dropped (no consumer with frozen encoder)
  - Hydra overrides: `+loss.rand_diff.weight=0.01 +loss.rand_diff.eps={ε_i} +loss.rand_diff.num_dirs=2`
- **Success criterion**:
  - No ε in {ε_1, ε_2, ε_3, ε_4} reproduces v2-like signature (σ₁ ≥ baseline +30% OR SR ≤ baseline −30%) → C2 strong on this confound
  - σ₁ and SR shifts are smooth in ε (no sign flip) → reversal is genuinely the encoder's role, not an ε scale artifact
- **Failure interpretation**:
  - One or more ε values produce v2-like inflation → reversal was ε-driven; C2 is downgraded back to "suggestive"; need a different design (varying λ, num_dirs, or finite-difference scheme) to isolate
  - Mixed: 2 of 4 ε values reproduce, 2 don't → middle-ground; report as "C2 holds in a regime; outside that regime ε scaling matters" and add the regime characterization to the paper
- **Table/figure target**: appendix Table A1 (ε sweep), main paper Fig 2 panel inset (the curve of σ₁(ε), SR(ε))
- **Priority**: MUST-RUN
- **Cost**: 4 runs × 0.8 GPU-h = 3.2 GPU-hours

### Block 2 (B2, M2): uvlowr-co-trained Stage-1 + freeze multi-predictor — discharges Confound 2

- **Claim tested**: C2 generalizes across encoder shapes; the v3 reversal is not a baseline-shape artifact
- **Why this block exists**: in v3, Stage-1 was trained with a uniform full-rank predictor, so the encoder shape was "baseline-like". Anti-claim B says "you only tested in baseline-shaped latents, and randdiff doesn't suit baseline-shaped latents". B2 retrains Stage-1 with uvlowr_r4 as the joint predictor, freezes the resulting (uvlowr-shaped) encoder, and reruns the 3-variant ablation. If the same C2 signature appears, C2 is robust across encoder shapes.
- **Dataset / split**: Push-T expert, identical to v2/v3
- **Compared systems**:
  - **Stage-1**: `stage1_uvlowr_r4` (joint 20K, uvlowr_r4 predictor) — new
  - **Stage-2 (frozen)**: 3 variants on the new Stage-1
    - `frozen_baseline_on_uvlowr_stage1` — predictor-only, no regularizer (control)
    - `frozen_uvlowr_r4_on_uvlowr_stage1` — matched predictor, the "co-shaped" path
    - `frozen_randdiff_on_uvlowr_stage1` — cross-shaped path (the falsifier)
  - 2 seeds each (total: 6 frozen runs)
- **Metrics** (decisive first):
  - **Predictor Jacobian σ₁, SR, F²** for randdiff — sign of v2 inflation vs v3 shrinkage in the new latent space
  - **Rollout MSE at h∈{1,3,5,10,15,20}** — same horizons as v3 for direct comparison
  - **Cross-Stage-1 comparison plot**: for each variant, plot σ₁ on (baseline-Stage-1) vs σ₁ on (uvlowr-Stage-1) — does the reversal hold in both shapes?
- **Setup details**:
  - Stage-1: 20K joint, batch 32, full LeWM training stack with `+predictor.ffn_rank=4`, SIGReg active, seed 0
  - Stage-2: 8K predictor-only, batch 32, encoder+projector+pred_proj frozen (KeepFrozenInEvalCallback), SIGReg dropped, seeds {0,1}
  - ε for randdiff: chosen from M0/B1 result. If B1 finds a robust ε range, use the midpoint; else use ε=0.05
- **Success criterion**:
  - **Strong**: randdiff predictor σ₁ < baseline σ₁ AND SR ≥ baseline SR in both Stage-1 shapes → reversal generalizes → C2 strong on Confound 2
  - **Mixed**: reversal direction holds in one Stage-1 only → C2 holds conditionally; paper must scope the claim
  - **Failed**: in uvlowr-shaped Stage-1, randdiff shows v2-like inflation → the v3 result was Stage-1-shape-specific; reframe the paper around "uvlowr-vs-baseline shaping" rather than "encoder absorption"
- **Table/figure target**: main paper Table 3 (cross-Stage-1 cross-variant); Fig 2 expanded to show both Stage-1 shapes
- **Priority**: MUST-RUN
- **Cost**: Stage-1 (3 GPU-h) + 6 frozen runs × 0.8h = **7.8 GPU-hours**

### Block 3 (B3, M3): uvlowr rank sweep — closes C13 / sharpens C3

- **Claim tested**: C13 quantitatively — rollout MSE has a knee near r=4 corresponding to Push-T's data intrinsic dim, not a smooth monotone curve
- **Why this block exists**: C13 currently rests on a 3-point alignment ("state-delta ER=3.24, latent-delta ER=4.06, uvlowr r=4"). A reviewer will say "you picked r=4 because it worked." A rank sweep produces a curve; if it has a knee near 4 with worse performance both below and above (or a clear plateau starting at r=4), that's a quantitative falsification test, not a coincidence.
- **Dataset / split**: Push-T expert, same as B1
- **Compared systems**: `frozen_uvlowr_r{r}` for r ∈ {1, 2, 4, 8, 16}, frozen on v3 baseline Stage-1, 1 seed each
- **Metrics**:
  - **Rollout MSE at h∈{1,5,10}** — primary; the knee should show up here
  - **Param count and FLOPS** — confirms the count-vs-rank relationship is monotone
  - **Predictor Jacobian σ₁, SR, F²** — secondary; do high-rank uvlowr Jacobians approach baseline?
  - **Probing R² (eval.json)** — mechanically identical across all 5 (frozen encoder) → confirms freeze; no signal
- **Setup details**:
  - Stage-1: v3 `stage1_baseline_seed0_weights.ckpt` (same as v3)
  - Stage-2: 8K predictor-only, batch 32, seed 0, `+predictor.ffn_rank={r}` for the 5 values
  - Other hyperparameters identical to v3 M1
- **Success criterion**:
  - **Strong C13**: rollout MSE at h=10 has a minimum (or clear plateau onset) at r=4. Worse at r=1, r=2 (under-parameterized); plateau-tied at r=8, r=16 (over-parameterized but no extra gain) → matches data dim story
  - **Weakest C13**: monotone in r (smaller is always worse OR larger is always better) → "lean LeWM" framing requires reformulation
- **Failure interpretation**:
  - Monotone decreasing in r → uvlowr's r=4 win in v2/v3 was an under-parameterization sweet spot, not a data-dim match; C13 reframed as parameter-budget story
  - Monotone increasing in r (r=1 wins, r=16 loses) → the "low-rank prior helps" claim simplifies further: any low-rank constraint works, the specific r=4 was lucky
- **Table/figure target**: main paper Fig 3 (rank-vs-rollout curve); inline in C13 section of CLAIM.md
- **Priority**: MUST-RUN (one of the cheapest ways to upgrade a 3-point alignment to a quantitative knee finding)
- **Cost**: 5 runs × 0.8h = 4.0 GPU-hours

### Block 4 (B4, M4): Aggregate + verdict mapping

- **Claim tested**: none — execution discipline
- **Tasks**:
  - Run `aggregate.py` on B1+B2+B3 outputs → CSVs
  - Build the cross-condition table: variant × Stage-1-shape × ε × metric
  - Compute "v2 vs v3 vs v4" deltas for the predictor Jacobian σ₁/SR/F²
  - Update CLAIM.md and `/workspace/lewm_autodl_results_v4/ANALYSIS.md` with per-block verdict
  - Update `project_lewm_lowrank.md` memory with v4 outcome
- **Cost**: 0.2 GPU-hours
- **Priority**: MUST-RUN

---

## Run Order and Milestones

| Milestone | Goal | Runs | Decision gate | Cost (GPU-h) | Risk |
|---|---|---|---|---|---|
| **M0** | ε-norm probe + freeze sanity rerun on fresh instance | 1 sanity run | Pred_loss drop ≥30% AND encoder grad norm = 0 → proceed. If freeze fails → STOP, debug (most likely env issue) | 0.3 | low — same code as v3, just verifying env |
| **M1** (B1) | ε sweep on frozen randdiff | 4 runs × 1 seed × 8K | If ANY ε reproduces v2-like inflation → flag Anti-claim A as live; still proceed to B2/B3 but note caveat in B4. If none → C2 strong on Confound 3 | 3.2 | medium — possible that one ε does flip the sign; report whichever way it lands |
| **M2** (B2) | uvlowr-Stage-1 + freeze 3-variant | 1 stage-1 + 6 frozen runs | If randdiff Jacobian in uvlowr-Stage-1 matches v3 direction (shrinkage) → C2 strong on Confound 2. If matches v2 direction (inflation) → C2 is Stage-1-shape-specific, reframe | 7.8 | medium — Stage-1 has its own training-noise; budget for one retry if Stage-1 doesn't converge cleanly |
| **M3** (B3) | uvlowr rank sweep | 5 runs × 1 seed × 8K | Looking for knee at r=4 in rollout-MSE-vs-r curve. Any shape → publishable; smooth-monotone is the weakest outcome | 4.0 | low — independent of M1/M2 outcomes |
| **M4** (B4) | Aggregate + verdict | analysis only | none | 0.2 | low |
| **Total** | | 16 runs (4 sweep + 7 freeze + 1 stage-1 + 1 sanity + 3 already-collected v3 references reused) | | **15.5 GPU-h** | **~¥47** of ¥175 balance |

**Parallelism**: M0 must precede M1 (need ε bracket). M2's Stage-1 can run in background while M1 runs in foreground (Stage-1 needs full GPU). M3 can run after M2's Stage-2 frees the GPU. Single-GPU instance, no parallel jobs.

---

## Compute and Data Budget

- **Total GPU-hours**: 15.5 (≈¥47 on AutoDL 4090-48G at ¥3/hr)
- **Remaining balance after v4**: ~¥128 (room for ~40 more GPU-hours for v5 or follow-ups)
- **Data prep**: none (Push-T expert already on the per-instance volume; ckpts in `/workspace/lewm_autodl_results_v3/`)
- **Human eval**: none
- **Biggest bottleneck**: B2 Stage-1 (~3 GPU-h serial; cannot be parallelized on a single instance without renting more GPUs)

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ε sweep finds an ε that reproduces v2-like inflation | medium | high — would soften C2 back to "suggestive" | accept the result; report honestly; the ε-dependent regime characterization is itself a publishable finding |
| B2 Stage-1 (uvlowr-shaped) doesn't converge cleanly in 20K | medium | medium — would make B2 verdict ambiguous | budget +5K extension if pred_loss visibly still trending; cap at 25K |
| AutoDL network/disk issues (per v3 lessons) | medium | medium | reuse `flock`-guarded setup script from v3; create instance with `expand_system_disk_by_gb=80` from the start; abandon and recreate if first speedtest <2 MB/s |
| B3 rank=1 or rank=16 produces NaNs / training instability | low | low | extreme ranks have known fragility; if NaN, drop the offending r and report sweep over remaining values |
| Single-seed B1/B3 too noisy for verdict | medium | medium — h=15/20 was already noisy with n=2 in v3 | report B1/B3 at h≤10 only as primary; h=15/20 as supplementary; if one specific verdict turns on h=15, extend that single run to seed 1 |
| Stage-1 ckpt (v3 baseline) was a "lucky" seed | low — v3 already passed M0 sanity | high — would invalidate everything built on it | B2's uvlowr-Stage-1 is itself a different Stage-1 seed AND a different Stage-1 shape, so we get an implicit Confound-4 check |

---

## Implementation Notes

What changes vs. v3 codebase:
1. **`train.py`**: no new code; reuse `+freeze.enabled=true +freeze.stage1_ckpt=<path>` from v3 (works for both Stage-1 ckpts)
2. **`run_pipeline_v4.sh`** (new): orchestrator with 4 phases:
   - Phase A: B1 ε sweep (4 runs serial)
   - Phase B: B2 Stage-1 (1 run, 20K) → Stage-2 (6 runs serial)
   - Phase C: B3 rank sweep (5 runs serial)
   - Phase D: aggregate + verdict
3. **Hydra overrides for B1**: `+loss.rand_diff.eps=<ε_i>` for each of 4 ε values
4. **Hydra overrides for B3**: `+predictor.ffn_rank=<r>` for r ∈ {1, 2, 4, 8, 16}
5. **Stage-1 source for B2**: train fresh with `+predictor.ffn_rank=4 +loss.rand_diff.weight=0.0` (NO randdiff in Stage-1, only uvlowr architectural FFN constraint)
6. **Aggregation script**: extend `aggregate.py` to emit `v4_cross_condition.csv` with columns `(stage1_source, variant, ε, seed, h, mse, σ₁, SR, F²)`

**Assert in M0**:
```python
# v3 freeze sanity asserts still apply:
for name, p in pl_module.model.encoder.named_parameters():
    assert not p.requires_grad, f"encoder param {name} NOT frozen"
# v4 additional: ε passed through correctly
assert cfg.loss.rand_diff.eps == cfg_eps_arg, f"ε override not applied: {cfg.loss.rand_diff.eps}"
```

---

## Final Checklist

- [x] Main paper tables covered (cross-Stage-1, ε sweep, rank sweep)
- [x] Novelty isolated (C2 holds across ε AND across encoder shapes)
- [x] Simplicity defended (rank sweep shows whether r=4 is the data-matched sweet spot or just a sweep artifact)
- [x] Frontier component justified — N/A, no LLM/VLM/Diffusion in this work
- [x] MUST-RUN vs NICE-TO-HAVE separated (all 4 milestones MUST-RUN; nice-to-haves like λ sweep and Stage-1 seed sweep explicitly deferred)
- [x] Failure interpretations specified for each block (B1 ε-reproduces-v2, B2 Stage-1-shape-specific, B3 monotone)
- [x] Anti-claims listed (A: ε mis-calibration, B: Stage-1-shape artifact, C: r=4 is lucky)
- [x] Within budget (15.5 / ~58 remaining GPU-hours at ¥3/hr against ¥175 balance)

---

## Lessons inherited from v2 → v3 (to avoid repeating)

- **AutoDL disk**: create instance with `expand_system_disk_by_gb=80` from the start (Push-T h5 decompress needs ~27 GB+ headroom)
- **AutoDL network**: if pip speedtest <2 MB/s on first 30 seconds, release immediately and recreate; tsinghua/aliyun mirrors via `~/.pip/pip.conf` mandatory
- **flock for setup**: wrap setup script with `flock -n /workspace/_setup_v4.lock` to prevent retry-loop clobbering
- **pkill safety**: never `pkill -f "pip install"` from a shell that contains "pip install" in its cmdline — kills self. Use explicit PIDs from `ps -ef | grep <token>`
- **SSH cd prefix**: paramiko `exec_command` runs from `/root` by default. Always prefix repo commands with `cd /workspace/le-wm &&`
- **Picklable freeze**: don't use `module.train = types.MethodType(...)` (breaks `torch.save`); reuse v3's `KeepFrozenInEvalCallback`
- **AutoDL release**: ¥3/hr while running; release immediately after `scp` pull completes
- **Watcher race condition**: use pgrep-precheck in `auto_pipeline.py` to skip phases already done; do NOT rely on `nohup` + file watcher alone

---

## Composition with downstream skills

After v4 lands:
- `/auto-review-loop` to iterate on the CLAIM.md verdict with the new B1/B2/B3 evidence
- `/experiment-bridge` if the v4 verdict gates yet another experimental round (e.g., λ sweep if C2 still partial)
