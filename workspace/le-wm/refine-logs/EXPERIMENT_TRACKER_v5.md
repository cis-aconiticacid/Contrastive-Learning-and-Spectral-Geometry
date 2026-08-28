# Experiment Tracker v5 — Post-Novelty-Check Pivot

**Created**: 2026-05-12
**Plan**: `EXPERIMENT_PLAN_v5.md`
**Framing**: encoder-mediated compensation (C2) + nuclear-norm over-contraction (C_dyn) as anchor claims
**Stage-1 ckpt for B1/B3/B4/B7/B8**: `/workspace/lewm_autodl_results_v3/stage1_baseline_seed0_weights.ckpt` (216 MB)
**v5 results dir** (to create): `/workspace/lewm_autodl_results_v5/`

## Pre-registered predictions (write down BEFORE running)

| ID | Block | Prediction | Verdict gate |
|---|---|---|---|
| P-C2 | B1 | For ≥3 of 4 ε values, frozen_randdiff σ₁ < frozen_baseline σ₁ | Pass → C2 holds; fail → reframe |
| P-Cdyn | B4 | \|λ_max\|(λ=0.1) ≤ \|λ_max\|(λ=0) − 0.05 | Pass → C_dyn confirmed; fail → re-examine |
| P-C13 | B3 | Rollout MSE@h=10 minimum at r ∈ {2, 4}, NOT at r ∈ {8, 16} | Pass → knee at low rank; fail → demote C13 |

If 2+ of 3 pre-registered predictions hold → proceed to paper draft. Else → full reframe.

## Status legend
- TODO / RUNNING / DONE / NOT RUN

## Runs

| Run ID | Milestone | Block | Purpose | Variant / Override | Stage-1 source | Steps | Seeds | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| R200 | M0 | — | Latent-norm probe + ε bracket (reuse v4) | analysis only | v3 baseline | — | — | MUST | **DONE locally** | `v4_eps_bracket.json` — ε∈{0.0125, 0.05, 0.2, 0.8}, ‖z‖=33.28 |
| R201 | M0 | — | Freeze sanity on AutoDL (env reproducibility) | `frozen_sanity` | v3 baseline | 1000 | {0} | MUST | TODO | gate: pred_loss drop ≥30%, enc grad=0 |
| **B1 — ε sensitivity sweep on frozen randdiff** ||||||||||
| R210 | M1 | B1 | ε = 0.0125 | `frozen_randdiff +loss.rand_diff.eps=0.0125` | v3 baseline | 8000 | {0} | MUST | TODO | |
| R211 | M1 | B1 | ε = 0.05 (matches v3) | `frozen_randdiff +loss.rand_diff.eps=0.05` | v3 baseline | 8000 | {0} | MUST | NOT RUN | reuse v3 R006 (frozen_randdiff_seed0): σ₁=1.61, SR=22.7, h=10=0.961 |
| R212 | M1 | B1 | ε = 0.2 | `frozen_randdiff +loss.rand_diff.eps=0.2` | v3 baseline | 8000 | {0} | MUST | TODO | |
| R213 | M1 | B1 | ε = 0.8 | `frozen_randdiff +loss.rand_diff.eps=0.8` | v3 baseline | 8000 | {0} | MUST | TODO | |
| **B2 — uvlowr-co-trained Stage-1 + freeze multi-predictor** ||||||||||
| R220 | M2 | B2 | Stage-1 with uvlowr_r4 predictor (joint) | `stage1_uvlowr +predictor.ffn_rank=4` | — | 20000 | {0} | MUST | TODO | full LeWM stack; SIGReg ACTIVE; ~3 GPU-h |
| R230 | M2 | B2 | frozen baseline on uvlowr Stage-1 | `frozen_baseline` | uvlowr Stage-1 (R220) | 8000 | {0} | MUST | TODO | dep R220 |
| R231 | M2 | B2 | frozen baseline on uvlowr Stage-1, seed 1 | `frozen_baseline` | uvlowr Stage-1 (R220) | 8000 | {1} | MUST | TODO | dep R220 |
| R232 | M2 | B2 | frozen uvlowr (co-shaped) | `frozen_uvlowr_r4 +predictor.ffn_rank=4` | uvlowr Stage-1 (R220) | 8000 | {0} | MUST | TODO | dep R220 |
| R233 | M2 | B2 | frozen uvlowr, seed 1 | `frozen_uvlowr_r4 +predictor.ffn_rank=4` | uvlowr Stage-1 (R220) | 8000 | {1} | MUST | TODO | dep R220 |
| R234 | M2 | B2 | frozen randdiff (cross-shaped: the falsifier) | `frozen_randdiff` (ε from B1 winning value) | uvlowr Stage-1 (R220) | 8000 | {0} | MUST | TODO | dep R220 + B1 verdict |
| R235 | M2 | B2 | frozen randdiff, seed 1 | `frozen_randdiff` | uvlowr Stage-1 (R220) | 8000 | {1} | MUST | TODO | dep R220 |
| **B3 — uvlowr rank sweep (defends C13)** ||||||||||
| R240 | M3 | B3 | r = 1 | `frozen_uvlowr_r1 +predictor.ffn_rank=1` | v3 baseline | 8000 | {0} | MUST | TODO | NaN watch — extreme low rank |
| R241 | M3 | B3 | r = 2 | `frozen_uvlowr_r2 +predictor.ffn_rank=2` | v3 baseline | 8000 | {0} | MUST | TODO | |
| R242 | M3 | B3 | r = 4 (reuse v3 R004) | `frozen_uvlowr_r4 +predictor.ffn_rank=4` | v3 baseline | 8000 | {0} | MUST | NOT RUN | reuse v3 R004: h=10 = 0.318 |
| R243 | M3 | B3 | r = 8 | `frozen_uvlowr_r8 +predictor.ffn_rank=8` | v3 baseline | 8000 | {0} | MUST | TODO | |
| R244 | M3 | B3 | r = 16 | `frozen_uvlowr_r16 +predictor.ffn_rank=16` | v3 baseline | 8000 | {0} | MUST | TODO | |
| **B4 — randdiff λ dose-response (defends C_dyn)** ⭐ NEW ||||||||||
| R250 | M4 | B4 | λ = 0 (reuse v3 R002 — frozen_baseline) | `frozen_baseline` | v3 baseline | 8000 | {0} | MUST | NOT RUN | reuse v3 R002: σ₁=2.05, SR=20.8, h=10=0.267 |
| R251 | M4 | B4 | λ = 0.001 | `frozen_randdiff +loss.rand_diff.weight=0.001` | v3 baseline | 8000 | {0} | MUST | TODO | |
| R252 | M4 | B4 | λ = 0.003 | `frozen_randdiff +loss.rand_diff.weight=0.003` | v3 baseline | 8000 | {0} | MUST | TODO | |
| R253 | M4 | B4 | λ = 0.01 (reuse v3 R006) | `frozen_randdiff +loss.rand_diff.weight=0.01` | v3 baseline | 8000 | {0} | MUST | NOT RUN | reuse v3 R006: σ₁=1.61, h=10=0.961 |
| R254 | M4 | B4 | λ = 0.03 | `frozen_randdiff +loss.rand_diff.weight=0.03` | v3 baseline | 8000 | {0} | MUST | TODO | |
| R255 | M4 | B4 | λ = 0.1 | `frozen_randdiff +loss.rand_diff.weight=0.1` | v3 baseline | 8000 | {0} | MUST | TODO | |
| **B5 — Post-hoc DMD extraction** ⭐ NEW ||||||||||
| R260 | M5 | B5 | Extract DMD spectrum on ALL v4 frozen ckpts | analysis only | n/a | — | — | MUST | TODO | reuse `tier1_dump_trajectories.py` + `tier1_metrics.py`; ~5 min/ckpt × ~20 ckpts |
| **B6 — Aggregation** ||||||||||
| R270 | M6 | B6 | Aggregate + verdict update | analysis only | — | — | — | MUST | TODO | output: `lewm_autodl_results_v5/ANALYSIS.md` |
| **B7 — Optimizer/LR robustness (NICE)** ||||||||||
| R280 | M7 | B7 | frozen_baseline @ LR=5e-4 | `frozen_baseline +optimizer.lr=5e-4` | v3 baseline | 8000 | {0} | NICE | TODO | |
| R281 | M7 | B7 | frozen_baseline @ LR=5e-5 (control) | reuse R202 | v3 baseline | 8000 | {0} | NICE | NOT RUN | reuse v3 R002 |
| R282 | M7 | B7 | frozen_randdiff @ LR=5e-4 | `frozen_randdiff +optimizer.lr=5e-4` | v3 baseline | 8000 | {0} | NICE | TODO | |
| R283 | M7 | B7 | frozen_randdiff @ LR=5e-5 (control) | reuse v3 R006 | v3 baseline | 8000 | {0} | NICE | NOT RUN | reuse |
| **B8 — 2nd Stage-1 baseline seed (NICE)** ||||||||||
| R290 | M8 | B8 | Stage-1 baseline seed 1 (joint) | `stage1_baseline` | — | 20000 | {1} | NICE | TODO | for cross-seed Stage-1 ablation |
| R291 | M8 | B8 | frozen_baseline on Stage-1 seed 1 | `frozen_baseline` | R290 | 8000 | {0} | NICE | TODO | dep R290 |
| R292 | M8 | B8 | frozen_uvlowr on Stage-1 seed 1 | `frozen_uvlowr_r4` | R290 | 8000 | {0} | NICE | TODO | dep R290 |
| R293 | M8 | B8 | frozen_randdiff on Stage-1 seed 1 | `frozen_randdiff` | R290 | 8000 | {0} | NICE | TODO | dep R290 |
| **B9 — Second dataset (CUT for v5)** ||||||||||
| — | — | B9 | PointMaze / Reacher Stage-1 + freeze 3v | — | — | — | — | CUT | DEFERRED | follow-up paper |

**Total new runs**: 22 must-run + 8 nice-to-have = 30 new training runs + 2 analysis-only blocks.
**Reused from v3**: R002 (frozen_baseline_seed0), R004 (frozen_uvlowr_r4_seed0), R006 (frozen_randdiff_seed0).

## Decision gates (post-execution, to fill in)

### After M0 (R201): GATE
- Sanity pred_loss drop: ___ %  encoder grad norm: ___ → PASS / FAIL

### After M1 (R210-R213): Anti-claim A verdict
- For each ε, σ₁(frozen_randdiff) − σ₁(frozen_baseline):
  - ε=0.0125: ___
  - ε=0.05 (R211 reused = R006): −0.44 (−21%)
  - ε=0.2: ___
  - ε=0.8: ___
- **P-C2 prediction (≥3/4 ε produce σ₁ shrinkage)**: PASS / FAIL
- **Anti-claim A verdict**: ___

### After M2 (R220-R235): Anti-claim B verdict
- randdiff direction on uvlowr-Stage-1:
  - on baseline-Stage-1 (v3 reference): σ₁ −21%, SR +10%, F² −32%
  - on uvlowr-Stage-1 (R234, R235): σ₁ ___%, SR ___%, F² ___%
- **Anti-claim B verdict**: ___

### After M3 (R240-R244): Anti-claim D verdict / C13 sharpening
- Rollout MSE@h=10 vs r:
  - r=1: ___
  - r=2: ___
  - r=4 (v3 R004): 0.318
  - r=8: ___
  - r=16: ___
- Knee location: ___
- **P-C13 prediction (min at r∈{2,4})**: PASS / FAIL
- **Anti-claim D verdict**: ___

### After M4 (R250-R255): Anti-claim C verdict / C_dyn dose-response
- |λ_max| vs λ:
  - λ=0 (R250 = v3 R002): 1.095 (from tier-1)
  - λ=0.001: ___
  - λ=0.003: ___
  - λ=0.01 (R253 = v3 R006): 0.996 (from tier-1)
  - λ=0.03: ___
  - λ=0.1: ___
- Monotone? ___
- **P-Cdyn prediction (|λ_max|(λ=0.1) ≤ |λ_max|(λ=0) − 0.05)**: PASS / FAIL
- **Anti-claim C verdict**: ___

### After M5 (R260): DMD replication at 8K
- |λ_max|(8K) for tier-1 variants:
  - frozen_baseline: 3K=1.095 → 8K=___
  - frozen_uvlowr: 3K=1.073 → 8K=___
  - frozen_randdiff: 3K=0.996 → 8K=___
- Verdict: tier-1 numbers replicate at full budget? ___

### Final claim status (to update CLAIM.md post-M6)

| Claim | Pre-v5 | Post-v5 verdict | Driving evidence |
|---|---|---|---|
| C2 (encoder absorption is causal) | suggestive | TBD via B1 + B2 + B7 | sign-flip robust to ε, Stage-1 shape, LR |
| C_dyn (over-contraction failure) | TBD | TBD via B4 + B5 | λ dose-response curve |
| C13 (uvlowr r=4 matches data dim) | confirmed 3-point | TBD via B3 | knee in rollout-vs-r curve |
| C3 (uvlowr Pareto) | weakened | TBD via B3 + B2 | rank sweep + cross-Stage-1 |
| C4 (randdiff hurts long-h) | confirmed | TBD via B4 | dose-response on rollout MSE |
| C5 (encoder Jac dichotomy under joint reg) | reframed | unchanged | v2-only phenomenon |
| C6 (training-driven low-rank ViT) | confirmed | unchanged | from cheap check |
| C7 (randdiff inflates σ₁) | reversed | unchanged (v5 confirms reversal) | v2-only artifact |

## Reuse / external references

- v2 reference: `/workspace/lewm_autodl_results_v2/` (3v × 3s × 40K joint)
- v3 results: `/workspace/lewm_autodl_results_v3/`
- v3 Stage-1 ckpt: `/workspace/lewm_autodl_results_v3/stage1_baseline_seed0_weights.ckpt`
- v3 frozen_baseline_seed0 (R002): rollout h=10 = 0.267, σ₁ = 2.05, SR = 20.8
- v3 frozen_uvlowr_r4_seed0 (R004): rollout h=10 = 0.318
- v3 frozen_randdiff_seed0 (R006): rollout h=10 = 0.961, σ₁ = 1.61, SR = 22.7
- Tier-0 results: `refine-logs/tier0_results.json`, `tier0_correlation_matrix.json`
- Tier-1 results (3K-step local repro): `refine-logs/tier1_traj/predicted_metrics.json`
- Tier-1 DMD: baseline |λ_max|=1.095 (5 unstable), uvlowr |λ_max|=1.073 (3 unstable), randdiff |λ_max|=0.996 (0 unstable)

## Budget tracking

- Planned MUST: 19.5 GPU-h (~¥59)
- Planned NICE: +8.6 GPU-h (~¥26)
- Spent on v5: ¥0 (not started; tier-0/tier-1 was local, no AutoDL cost)
- Balance pre-v5: ¥175.32
- Expected balance post-v5 MUST-only: ~¥116
- Expected balance post-v5 MUST+NICE: ~¥90

## Operational checklist (v3 lessons; do BEFORE first launch)

- [ ] AutoDL instance with `expand_system_disk_by_gb=80` from start
- [ ] First pip speedtest >2 MB/s; if not, release + recreate
- [ ] `~/.pip/pip.conf` tsinghua mirror
- [ ] `flock -n /workspace/_setup_v5.lock` wraps setup
- [ ] All SSH cmds prefix `cd /workspace/le-wm &&`
- [ ] Upload v3 Stage-1 ckpt to instance before B1 launch
- [ ] Release immediately after final `scp` pull
