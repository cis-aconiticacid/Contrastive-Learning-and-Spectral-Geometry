# Toy Pilot: Low-Rank Predictor Regularization (4060 8GB local)

**Date**: 2026-05-09
**Goal**: Compare two ways of imposing low-rank dynamics on a JEPA-style predictor:
- **Variant A (`uv_lowr`)** — explicit UV factorization of the FFN linears (LoRA-style).
  Variational nuclear norm: `‖W‖_* = min_{W=UV^T} ½(‖U‖_F² + ‖V‖_F²)`,
  optimized for free by AdamW weight decay. **No Jacobian computation.**
- **Variant B (`rand_diff`)** — keep predictor full rank, add Scarvelis & Solomon (2024)
  randomized one-sided finite-difference penalty
  `E_{v∼Sphere} ‖p(z+εv, c) − p(z, c)‖₂ / ε`
  as auxiliary loss. **Approximates ‖J_p‖_*.**

Both target the same theoretical object: nuclear-norm of the predictor's Jacobian.

## Setup

**Why a synthetic toy first?** The full LeWM stack (Lightning + stable-pretraining +
ViT encoder + 224×224 images + HF dataset download) was too heavy for a 30-min iteration
loop on 8 GB. Instead I copied the actual `ARPredictor` Transformer from
`le-wm/module.py` and trained it on a synthetic dynamical system whose intrinsic
dimension is *known* to be low rank, so the regularizer mechanism gets a clean signal.

**Data generator** (controlled):
```
z_{t+1} = z_t + α · tanh( U V^T z_t + B a_t )    α=0.3, U,V ∈ ℝ^{64×4}
```
- latent `d = 64`, true dynamics rank = 4 (force lives in 4-d subspace)
- action_dim = 4, history_size = 3, rollout horizon = 6 future steps
- 4096 train trajectories, 512 val trajectories

**Predictor** (smaller copy of LeWM ARPredictor):
- depth = 3, heads = 4, dim_head = 32, mlp_dim = 256, hidden_dim = 64
- ConditionalBlock with AdaLN-zero, identical forward to `le-wm/module.py`

**Training**: 2000 steps, batch 128, AdamW lr=3e-4, wd=1e-3, gradient clip 1.0,
3 seeds (0, 1, 2).

## Results (3-seed mean ± std)

| Variant         | Params       | Train MSE       | 1-step rollout  | 4-step rollout  | 6-step rollout  | Mem    | ms/step |
|-----------------|--------------|-----------------|-----------------|-----------------|-----------------|--------|---------|
| baseline        | 278,208      | 0.0107 ± 0.001  | 0.0135 ± 0.001  | 0.121 ± 0.044   | 0.348 ± 0.159   | 48 MB  | 7.2     |
| uv_lowr (r=4)   | **187,584**  | 0.0109 ± 0.001  | 0.0137 ± 0.002  | 0.122 ± 0.044   | 0.350 ± 0.161   | 47 MB  | 7.4     |
| rand_diff (λ=0.05) | 278,208   | 0.0134 ± 0.001  | 0.0181 ± 0.002  | 0.142 ± 0.047   | 0.383 ± 0.166   | 101 MB | 34.0    |

**Per-step rollout MSE (mean over seeds):**
```
step          1        2        3        4        5        6
baseline    0.014    0.029    0.063    0.121    0.214    0.348
uv_lowr     0.014    0.029    0.063    0.122    0.216    0.350
rand_diff   0.018    0.038    0.077    0.142    0.242    0.383
```

## Take-aways

1. **`uv_lowr` matches baseline at 33% fewer params, identical mem & speed.**
   This is exactly the "compute/memory-efficient long-horizon JEPA" pitch — the rank-4
   architectural constraint is *free* when the true dynamics is rank 4.
2. **`rand_diff` (single λ=0.05) is uniformly worse and 5× slower.** Penalty is
   over-regularizing; needs a smaller λ. Sweep in progress.
3. **Rollout error compounding shape is identical** across all variants — i.e. `uv_lowr`
   doesn't *prevent* compounding, it just doesn't make it worse despite using fewer
   params. The "low-rank stops compounding" claim is **not** supported in this toy.

## Caveats

- This toy uses synthetic data with **known** rank-4 dynamics. The result mostly tests
  the question "*if* the dynamics is rank-r, can a UV-factored predictor capture it?"
  Answer: yes, trivially. The harder question — "*is* LeWM's pixel-driven latent
  dynamics actually low rank?" — needs real LeWM data.
- 6-step rollout might not be long enough to surface rollout-compounding differences.
- Only 2000 training steps; baseline/uv_lowr might still be undertraining.

## Rank sweep for `uv_lowr` (single seed=0, 2000 steps)

| Variant       | Params       | Train MSE | 1-step   | 4-step   | 6-step   | Mem    | ms/step |
|---------------|--------------|-----------|----------|----------|----------|--------|---------|
| baseline      | 278,208      | 0.01013   | 0.01186  | 0.09882  | 0.27447  | 48.3   | 7.45    |
| uv_lowr r=4   | **187,584**  | 0.01018   | 0.01181  | 0.09864  | 0.27430  | 47.2   | 7.21    |
| uv_lowr r=8   | 195,264      | 0.01019   | 0.01199  | 0.09871  | 0.27361  | 47.3   | 8.02    |
| uv_lowr r=16  | 210,624      | 0.01022   | 0.01194  | 0.09865  | 0.27375  | 47.5   | 12.61   |
| uv_lowr r=32  | 241,344      | 0.01009   | 0.01191  | 0.09819  | 0.27269  | 48.0   | 8.43    |

→ Even at the *true* rank r=4, performance is identical to full-rank baseline. The
predictor truly only needs rank-4 capacity for this rank-4 dynamical system. Memory
savings are minimal at this small scale (FFN is 64→256→64; the savings only matter
when the inner dim is much larger, like the real LeWM case 1024→2048→1024).

## λ sweep for `rand_diff` (single seed=0, 2000 steps)

| Variant            | Train MSE | 1-step   | 4-step   | 6-step   | reg val | ms/step |
|--------------------|-----------|----------|----------|----------|---------|---------|
| baseline           | 0.01013   | 0.01186  | 0.09882  | 0.27447  | —       | 7.45    |
| uv_lowr r=4        | 0.01018   | 0.01181  | 0.09864  | 0.27430  | —       | 7.21    |
| rand_diff λ=0.001  | 0.01012   | 0.01190  | 0.09901  | 0.27453  | 1.02    | 105.0   |
| rand_diff λ=0.01   | 0.01044   | 0.01279  | 0.10357  | 0.28232  | 0.85    | 41.2    |
| rand_diff λ=0.05   | 0.01277   | 0.01590  | 0.11747  | 0.30655  | 0.76    | 35.0    |
| rand_diff λ=0.1    | 0.01791   | 0.02024  | 0.13496  | 0.33622  | 0.68    | 38.4    |
| rand_diff λ=0.5    | 0.13029   | 0.10456  | 0.36069  | 0.65768  | 0.26    | 40.6    |

→ **No `rand_diff` λ matches baseline**. At λ→0 it's a tie (i.e. the regularizer
turns off). At any meaningful strength (λ ≥ 0.01) it hurts. The penalty
`E_v ‖p(z+εv)−p(z)‖` shrinks the predictor's local Lipschitz constant, but in this
toy it just suppresses useful signal.

## Toy verdict

| Mechanism      | Performance       | Compute / memory | Implementation cost |
|----------------|-------------------|------------------|---------------------|
| **uv_lowr**    | **= baseline**    | **−33% params, =mem, =speed** | Trivial: factor each FFN linear |
| **rand_diff**  | **≤ baseline (best λ)** | **+5× time, +2× mem** | Need to choose ε, λ; 2 forward passes |

**On this synthetic problem with known rank-4 dynamics, the architectural-low-rank
route (Variant A) wins on every axis.** The randomized-differential nuclear-norm
penalty (Variant B) is empirically dominated by simply factoring the FFN.

## Caveats and open questions

- This toy does **not** test rollout error compounding sharply enough — all variants
  show the same exponential growth (since the dynamics is bounded and the predictor
  is essentially "right enough"). Real LeWM Push-T should expose larger compounding.
- The toy is generative — there's no encoder learning. Real LeWM has a ViT encoder
  whose latent might or might not actually be low-rank. This is the key thing to
  test next.
- We did not try λ between 0.001 and 0.01 for `rand_diff`. There may be a narrow
  Goldilocks zone, but the trend (worse with more reg) suggests not.

## Next step

Run the same baseline / uv_lowr (rank=4, 8) comparison on **real LeWM Push-T data**
locally on the 4060 (the dataset is 13 GB compressed; should fit). If the signal
holds — uv_lowr matching baseline at fewer params — that confirms the user's pitch
on a real pixel-driven JEPA. If local 4060 OOMs, fall back to AutoDL 4090.

## Files

- `toy_pilot.py` — self-contained training/eval script
- `sweep.sh` — rank/lambda sweep
- `results_seed{0,1,2}.json` — raw 3-seed results
- `sweep_runs/*.json` — per-config sweep results
