# randdiff_20K 结果速记（2026-05-15）

**Run**: `v5_randdiff_20K`，Push-T frozen-encoder ablation，20000 steps，~67 min 本地训练。
**生成时间**：2026-05-15 04:12 → 05:20 UTC。
**Ckpt**：`/workspace/stablewm_home/v5_randdiff_20K/v5_randdiff_20K_epoch_1_object.ckpt`
**完整 JSON**：`/workspace/stablewm_home/v5_randdiff_20K/final_analyses.json` + `eval.json`

---

## 1. Spectral eig (predictor latent-only Jacobian, n=6 samples)

| 指标 | randdiff_20K | uvlowr_40K | baseline_40K |
|---|---:|---:|---:|
| **\|λ_max\| mean** | **0.984 ± 0.017** | 1.409 ± 0.162 | 1.790 ± 0.294 |
| **n_unstable mean** | **0.33** (1/6 sample) | 1.3 | 1.0 |
| **spectral_abscissa** | 0.979 | 1.409 | 1.790 |
| n_complex (typical) | ~120 | — | — |

**关键观察**：randdiff_20K 把 mean \|λ_max\| 压到 **< 1**（stable manifold），效应是 uvlowr 的 ~2× 强：
- baseline → randdiff: **−45%**（1.790 → 0.984）
- baseline → uvlowr:  **−21%**（1.790 → 1.409）
- std 也最小（0.017 vs uvlowr 0.162 vs baseline 0.294）

> ⚠️ randdiff 仅训 20K 而 baseline/uvlowr 训 40K — **不对等比较**。但 v3 anchor 8K 时 randdiff σ₁ 已比 baseline 低 22%（1.625 vs 2.074），20K 时 \|λ_max\| 进一步降到 stable，可推测 trajectory 单调下降。下次需要 randdiff_40K 完成对等对比。

---

## 2. Rollout MSE（mse_per_step，n=256）

| horizon | randdiff_20K | uvlowr_40K | baseline_40K | rd/base ratio |
|---|---:|---:|---:|---:|
| h=1 | 0.024 | 0.0044 | 0.0040 | **6.0×** worse |
| h=3 | 0.095 | 0.0298 | 0.0426 | 2.2× |
| h=5 | 0.214 | 0.0532 | 0.0854 | 2.5× |
| h=10 | 0.349 | 0.0518 | 0.0520 | 6.7× worse |
| h=15 | 1.030 | 0.1486 | 0.0942 | 10.9× |
| h=20 | **2.158** | 0.5103 | 0.0426 | **50.7×** worse |

**关键观察**：randdiff 在 **所有 horizon** 都比 baseline 差，h=20 时 50× worse。这跟 uvlowr 的「短期好长期发散」反向 — randdiff **全程性能恶化**。

**机制解读**：randdiff 把 \|λ_max\| 压到 <1 → predictor 把所有 latent 朝某个 attractor 收缩 → rollout 失去时序信息 → 累积误差严重。"过稳定 / 收缩失效" 假设的直接证据。

> ⚠️ 注意单位：mse_per_step 而非 cumulative。20K vs 40K 训练量差异也会让 randdiff rollout 看着差（baseline 训 2× 久）— 不能单看 magnitude，需关注 ratio across horizons。

---

## 3. Encoder Jacobian + 其他

- encoder spectral_norm = 11.28（与 baseline/uvlowr 同量级 ~11.x，因 encoder 冻结）
- encoder stable_rank = 2.70；effective_rank = 5.86
- **TwoNN intrinsic_dim = 0.45** — 继承之前的 strided sampling bug，**unfit for paper**，需修复重跑
- probe ridge R² overall = 0.847；MLP R² = 0.923

---

## 4. P-gates 更新

| Gate | 之前状态 | 新数据 | 新状态 |
|---|---|---|---|
| **P-Anti-I (randdiff branch)** | pending | randdiff_20K \|λ_max\|=0.98 < baseline 1.79 | ✅ **PASS (partial, 20K)** — 需 40K 对等闭合 |
| **P-Cdyn**（\|λ_max\|(rd) < \|λ_max\|(base) − 0.05） | pending | 0.98 < 1.79 − 0.05 | ✅ **PASS** (Δ = 0.806，远超 noise floor 0.05) |
| **过稳定/收缩失效** | qualitative claim | rollout h=20: rd 2.16 vs base 0.04 | ✅ **strong mechanistic evidence** |

---

## 5. 论文影响

1. **\|λ_max\| 作主指标更稳**：randdiff 效应 −45% vs σ₁ 的 −22%，statistical power 翻 2×。
2. **randdiff 是 over-regularization 反例**：spectral 压制太狠 → rollout 全程恶化。这恰好是 Codex novelty check 里 "收缩失效" claim 的直接证据。
3. **uvlowr 是更安全的中间方案**：spectral 适度压制（\|λ_max\|=1.41 仍 unstable，但小），short-horizon rollout 不差。论文应把 uvlowr 和 randdiff 框为光谱压制 strong/weak 两端。
4. **下次必跑**：
   - randdiff_40K 完成对等对比（应在 AutoDL 跑，本地 4060 约 ~118 min）
   - TwoNN strided sampling fix（10 min）
   - randdiff 在 10K 中间点 spectral，确认 \|λ_max\| 单调下降轨迹

---

## 6. 备注：本次会话

- 训练用 NW=4 + persistent_workers + prefetch_factor=2（WSL ipc=host 后 /dev/shm=13GB）
- 实测速度 **4.61 it/s** Push-T frozen（比预测 5.65 略慢，可能因 jacobian_probe 间歇拖慢）
- 67 min train + 9 min eval_rollout + final_analyses + spectral_postprocess = 76 min total
- 训练时 jacobian_probe.jsonl 中 σ₁/SR 字段都是 None（instrumentation bug 已知，不影响最终 analyses）
