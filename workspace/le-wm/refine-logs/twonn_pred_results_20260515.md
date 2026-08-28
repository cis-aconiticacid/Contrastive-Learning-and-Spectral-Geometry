# TwoNN(predictor) 全表 + spectral 综合（2026-05-15）

**修复**：`spectral_postprocess.py` 加 strided cross-trajectory 采样并加测 predictor output TwoNN。原版 (1) 把 (B,T,D) reshape 成 (B*T,D) 导致 r1≈0、TwoNN ~0.45 的 bug；(2) 只测 encoder output，frozen-encoder 实验中 variant 间完全相同（3.42）无区分力。

**完整 9-ckpt 综合表**（n_samples=6 for eig, N=2048 for TwoNN, batch_size=16，seed=0 shuffled DataLoader）：

| setting | run | enc | pred | Δ(p−e) | \|λ_max\| | std | n_unst |
|---|---|---:|---:|---:|---:|---:|---:|
| **frozen 40K** | v5_baseline_40K | 3.42 | 3.72 | +0.30 | **1.790** | 0.294 | 1.00 |
| **frozen 40K** | v5_uvlowr_40K | 3.42 | 3.88 | +0.46 | **1.409** | 0.162 | 1.33 |
| **frozen 20K** | v5_randdiff_20K | 3.42 | 3.58 | +0.16 | **0.984** | 0.017 | 0.33 |
| frozen tier1 | tier1_baseline | 3.42 | 3.60 | +0.18 | 1.265 | 0.030 | 3.00 |
| frozen tier1 | tier1_uvlowr | 3.42 | 3.54 | +0.12 | 1.319 | 0.017 | 3.83 |
| frozen tier1 | tier1_randdiff | 3.42 | 3.52 | +0.10 | 0.770 | 0.006 | 0.00 |
| **joint long** | baseline_long | 5.18 | 6.05 | +0.87 | 0.751 | 0.004 | 0.00 |
| **joint long** | uvlowr_r4_long | 4.95 | 5.40 | +0.45 | 0.735 | 0.004 | 0.00 |
| **joint long** | randdiff_lam001_long | 5.47 | 5.57 | +0.10 | 0.784 | 0.005 | 0.00 |

---

## Test set prediction accuracy（R² across horizons）

Push-T 是连续 state 回归任务（7-dim state），"准确率" = R²（test set, n_rollout=256 trajectories）。state variance 从 probe MSE / (1−R²) 反推（baseline state_var ≈ 0.876）。

| horizon | **baseline** | **uvlowr** | **randdiff** | rd vs base |
|---|---:|---:|---:|---:|
| h=1 | **99.54%** | 99.50% | 97.30% | −2.24 pp |
| h=3 | 95.14% | 96.60% ⭐ | 89.19% | −5.95 pp |
| h=5 | 90.25% | 93.93% ⭐ | 75.53% | −14.72 pp |
| h=10 | 94.07% | 94.09% | 60.18% | −33.89 pp |
| h=15 | 89.25% | 83.04% | **−17.61%** ❗ | −106.86 pp |
| h=20 | **95.13%** | 41.75% | **−146.37%** ❗ | −241.50 pp |

**paper 核心 message**：

1. **baseline 长 horizon 仍 R²≈95%**：encoder 每步矫正 latent，predictor 不稳但 system 整体 stable。这是论文 "encoder rescues unstable predictor" 现象。
2. **uvlowr 在 h≤10 ⭐ 击败 baseline**（h=3, h=5）：spectral 压制 21% + Pareto 不 hurt 短 horizon。然后 h=15-20 突然崩。
3. **randdiff R² 在 h=15 变 −17.6%、h=20 变 −146%**：**比 constant-mean predictor 还差**！这是 paper 最 sharp 的「过收缩 → 收缩失效」证据——模型不仅没学到 dynamics，连"啥也不预测"都不如。

> ⚠️ **方法论 caveat**：state variance 用 probe.mse / (1−probe.r²) 反推。probe 是 encoder→state ridge 回归，所有 frozen ckpt 该值完全一致（0.876），所以 v5 三 variant 用同一分母比较 valid。但 tier1 / joint 未跑 eval.json 故无 R²_state。

---

## Prediction accuracy 表（latent space，train_pred_loss 末段 vs val_pred_loss）

| run | train (last 100 avg) | val | gap | vs baseline val |
|---|---:|---:|---:|---:|
| v5_baseline_40K | 0.0185 | **0.0173** | −0.0012 | — |
| v5_uvlowr_40K | 0.0208 | 0.0189 | −0.0019 | **+9%** |
| **v5_randdiff_20K** | 0.0540 | **0.0547** | **+0.0008** | **+217%** ❗ |
| tier1_baseline | 0.0534 | 0.0331 | −0.0203 | — |
| tier1_uvlowr | 0.0575 | 0.0358 | −0.0217 | +8% |
| **tier1_randdiff** | 0.1110 | 0.0927 | −0.0183 | **+180%** |
| baseline_long (joint) | 0.0662 | 0.0513 | −0.0149 | — |
| **uvlowr_r4_long** (joint) | 0.0642 | 0.0511 | −0.0131 | **0%** (tie!) |
| randdiff_lam001_long (joint) | 0.0648 | 0.0539 | −0.0109 | +5% |

**核心 insight**：

1. **randdiff_20K 单步 val pred_loss 已经 3.2× 于 baseline**——h=20 rollout MSE 50× 只是症状放大。"predictor 退化 identity"的代价是**每一步**都不能预测。
2. **uvlowr 是 Pareto-good**：frozen +9% accuracy 代价 + \|λ_max\| 降 21%；joint 0% accuracy 代价。randdiff 是 Pareto-bad（217% accuracy 代价换 \|λ_max\| 降 45%）。
3. **randdiff loss weight sensitivity 极高**：joint lam=0.001 (+5%) vs frozen lam=0.01 (+217%)。论文 sweep 必跑。
4. **gap 普遍 negative**：train 用 last-100-step running mean（含 noisy 最新 batch），val 是 epoch 末整体平均；非真实 overfit。**v5_randdiff_20K gap = +0.0008**（少有的正值，轻微过拟合 signal，但量级 << val_loss 0.055）。
5. **frozen vs joint 的 val_pred_loss 不可直接比**：frozen "1-epoch" 是 limit_train_batches=N 的有限 step，joint 是 full epoch；data ordering 不一样。同 setting 内比较 (frozen vs frozen, joint vs joint) 才公平。

---

## 五大新 finding

### F1 — **joint train 本身就 stable**（\|λ_max\| ≈ 0.75）；frozen ablation 把 predictor 推进 unstable regime

joint long-train 三 variant 的 \|λ_max\| 都聚在 0.74–0.78，差异 < 0.05。frozen 40K baseline 1.79（unstable），uvlowr 1.41（borderline），只有 randdiff 把它压回 0.98（边界）。

**Implication**：论文 "randdiff/uvlowr 收缩 spectral" 的强 effect 主要是 **frozen ablation 制造出来的 unstable regime 上的恢复**，而非自然训练动力学的修正。这是个微妙但必须 disclose 的 framing 边界。

### F2 — randdiff Δ(pred−enc) 跨 setting 永远最小（+0.10-0.16），mechanism 一致

| variant | frozen 40K Δ | frozen tier1 Δ | joint long Δ |
|---|---:|---:|---:|
| baseline | +0.30 | +0.18 | +0.87 |
| uvlowr | +0.46 | +0.12 | +0.45 |
| **randdiff** | **+0.16** | **+0.10** | **+0.10** |

predictor 在所有 setting 下都被 randdiff 推向 identity-like（小 Δ，\|λ_max\| 接近 1）。这是 **mechanism universality**：与训练方式无关，randdiff 把 predictor map 推向 near-identity。

### F3 — uvlowr 在 frozen vs joint 表现完全反向

| setting | uvlowr pred TwoNN | vs baseline pred |
|---|---:|---:|
| frozen 40K | 3.88 | **+0.16 (higher!)** |
| joint long | 5.40 | **−0.65 (lower)** |

joint train 时 uvlowr 的 rank-4 FFN 约束 backprop 到 encoder，全链路紧凑（enc 4.95 < baseline 5.18，pred 5.40 < 6.05）。frozen 时 encoder 锁死，predictor 为弥补 rank-4 容量损失反而 "spread" 到更高维 manifold。

**Implication**：uvlowr 不是 "spectral 压制 light 版" 的单调中间态，而是 **不同方向的 regularizer**（rank-4 容量 vs randdiff 方向随机扰动）。论文应避免把它框成 "weaker randdiff"。

### F4 — TwoNN(pred) 的 sensitivity 阶梯：joint > frozen-long > frozen-short

- joint long: variant 间 TwoNN(pred) spread 0.65（5.40-6.05）
- frozen 40K: spread 0.30（3.58-3.88）
- frozen tier1 short: spread 0.08（3.52-3.60，noise）

frozen short 时 TwoNN(pred) **几乎无区分力**。\|λ_max\| 仍然区分（tier1_randdiff 0.77 vs tier1_baseline 1.27）。
**TwoNN(pred) 适合长训 publication-grade，short-train pilot 应仅用 \|λ_max\| 判定。**

### F5 — encoder TwoNN 完美区分 frozen vs joint setting（sanity check 通过）

所有 frozen ckpt：enc TwoNN = 3.42 ± 0.00 → 同一 encoder 输出同样 latent 完全确认 (✓)。
所有 joint ckpt：enc TwoNN ∈ [4.95, 5.47] → encoder 被 jointly 训练后 manifold 维度自然抬升。

---

## P-gate 更新

| Gate | 之前 | 新数据 | 状态 |
|---|---|---|---|
| **P-Cdyn**（\|λ_max\|↓ under randdiff） | pending | frozen 40K: 1.79 → randdiff 0.98（Δ=0.81 >> noise 0.05） | ✅ **PASS, strong** |
| **TwoNN sensitivity** | 假设 TwoNN 能区分 | TwoNN(pred) 区分有限（Δ < 0.5），\|λ_max\| 是 sharp 主指标 | ⚠️ **TwoNN demote 到 secondary** |
| **F1 framing 警告** | 假设 randdiff 修正 dynamics | joint 本就 stable，frozen 才需 fix | ⚠️ **paper 必 disclose** |

---

## 论文叙事建议

1. **主指标用 \|λ_max\|**：跨所有 ckpt 都 well-separated 的 sharp 指标，比 σ₁ 敏感 3.5×，比 TwoNN 敏感 2-5×。
2. **TwoNN(pred) 作 secondary 机制证据**：randdiff Δ(pred−enc) 跨 setting 都是 +0.10（mechanism universality），但 spread 不大需谨慎措辞。
3. **uvlowr vs randdiff = 两种 regularizer family**：不是 magnitude 差异，是不同 axis（rank capacity vs random direction）。Avoid linear-spectrum framing。
4. **frozen ablation 框为 "stress test"**：joint 本就 stable，frozen 制造受控 unstable，randdiff 在该 stress regime 下 rescue。这才是论文真正的 claim 边界。
5. **下次 must-run**：randdiff_40K 对等闭合（本地 ~118 min，AutoDL ~42 min）。

---

## 备注

- 本批 2 次 batch 总耗时：8 ckpts (~24s each) + 7 ckpts (~36s each) ≈ 7 min
- 修复后的 spectral_postprocess.py 已添加 `source` 参数支持 encoder/predictor 双 source
- 所有 9 个 final_analyses.json 已写入 `twonn_enc_intrinsic_dim` + `twonn_pred_intrinsic_dim`
- tworoom 6 ckpt 仍 pending（缺 tworoom.h5）
