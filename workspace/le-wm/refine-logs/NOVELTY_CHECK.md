# Novelty Check — LeWM low-rank predictor → mechanistic-compensation pivot

**Date**: 2026-05-12
**Reviewer**: Codex via `mcp__codex__codex` (gpt-5.4, xhigh reasoning)
**Thread**: `019e1a53-cb09-7d83-be95-c6333091d1e0`
**Source transcript**: `9852dbec-9cde-45d2-aa26-3ca90055adf0.jsonl`, line 6466
**Brief trace**: `.aris/traces/novelty-check/2026-05-12_run01/codex_trace.md` (32 lines)

This is the canonical novelty-check document. Section 1-7 reproduce the
Codex report verbatim (translated section headers preserved). Section 8 adds
the attack → v5.2 experiment-block mapping done after the report.

---

## 1. 提议的方法（一句话）

在 LeWM JEPA 上对比两个 predictor 正则（uvlowr UV 因子化 vs Scarvelis-Solomon randdiff 核范数），用 frozen-encoder ablation 隔离 encoder co-adaptation 的贡献，并用 DMD/Koopman 分析 predictor 学到的隐空间动力学。

---

## 2. Claim-by-claim 评分

| Claim | Novelty | Closest prior |
|---|---|---|
| **C2 reversal**（randdiff predictor-Jacobian signature 在 joint vs frozen 之间方向翻转） | **MEDIUM-HIGH** | 没找到完全对应的 |
| **过稳定/收缩失效**（randdiff → \|λ_max\|<1, n_unstable=0, rollout 因收敛到 fixed point 而爆炸） | **MEDIUM-HIGH** in JEPA context | Koopman AE stability work (arXiv:2212.12086, 2602.02592, 2602.02832) 通用稳定性研究存在，但 JEPA 上是新的 |
| **TLPS as diagnostic** | **TRIVIAL alone** | Wang/LeCun *Temporal Straightening* (arXiv:2603.12231) — 同实验室、同 Push-T、相同公式 `1−cos`，作为 loss |
| **C13 uvlowr r=4 matches data ID** | **WEAK** | TD-JEPA (arXiv:2510.00739) 已经把 JEPA 和 low-rank latent dynamics 联系起来；reviewer 会说没有 rank sweep 是 post-hoc |
| **C6 JEPA vs ImageNet 谱区分** | **LOW** | 表征性观察，不能撑论文 |
| **Cross-metric correlation 结构变化** | **NOT a novelty claim** | 只能当 supporting evidence，叫 "causal" 会被打 |
| **Koopman 应用到 JEPA** | **已被占** | Ruiz-Morales et al. (AAAI 2026 / arXiv:2511.09783) |
| **Frozen-encoder + predictor-only** 作为 setup | **已标准化** | DINO-WM (arXiv:2411.04983)、What Drives Success (arXiv:2512.24497) 都用 |

---

## 3. Codex 新发现的 prior（人工搜索遗漏的）

| Paper | arXiv | Overlap |
|---|---|---|
| **Learning Invariant Visual Representations for Planning with JEPA-WMs** | 2602.18639 | 用增强 predictive objective 改 representation，缩小 latent — 跟 C2 narrative 部分重叠 |
| **TD-JEPA** | 2510.00739 | 已连 JEPA 与 low-rank long-term dynamics |
| **DINO-WM** | 2411.04983 (PMLR 2025) | frozen encoder + predictor-only 已是标准 setup |
| **What Drives Success in Physical Planning with JEPA-WMs** | 2512.24497 | 研究 rollout loss / predictor 架构 / context，但**不**涉及 low-rank 或 nuclear-norm |
| **Eigenvalue init/reg for Koopman AEs** | 2212.12086 | 通用稳定性研究 |
| **Spectral-control Koopman Transformers** | 2602.02592 | 已在用 spectral control |
| **Continuous-Time Koopman AEs** | 2602.02832 | 同上 |

---

## 4. 总体 Novelty 评分

| Framing | Score | Verdict |
|---|---|---|
| **Broad framing** —"我们把 low-rank/nuclear-norm 加到 JEPA predictor" | **4 / 10** | incremental |
| **Tight mechanistic framing** —"joint JEPA 训练时 predictor regularization 不是 predictor-local，encoder co-adapt 能反转可见效应；frozen encoder 暴露真正的 predictor-side 行为；核范数会把 predictor 推成收缩映射" | **6 / 10** | moderate, plausibly publishable |

**Recommendation**: **PROCEED with tight framing**。不要 abandon，**abandon 的是 weak framing**。

---

## 5. Reviewer 的攻击点（按 Codex 排序）

1. "Causal 措辞太强 —— freeze/unfreeze 是 *one* intervention，不是完整机理证明"
2. "Frozen-encoder + predictor-only 训练在 JEPA-WM 里已是常规（DINO-WM）"
3. "TLPS 已在 Wang et al.（同实验室！）；Koopman/DMD 已在 Ruiz-Morales et al."
4. "Push-T-only, 一个架构, 一个 rank, 一个 nuclear-norm 方法, seed 太少"
5. "你 post-hoc 量了一堆 metric，pre-registered 的最小机理测试在哪？"
6. "Rank-4 alignment 没 rank sweep → numerology"
7. "Reversal 可能是 optimizer/schedule 依赖，不是 'encoder absorption'"

---

## 6. 推荐 paper positioning

1. **Headline pivot**：从"low-rank predictor for JEPA"改写成 **"Encoder-mediated compensation can mask and even invert predictor-side regularization effects in end-to-end JEPAs"**
2. **Pillar 1**：joint vs frozen 下的 spectral-signature 符号翻转（C2）
3. **Pillar 2**：randdiff 让 predictor 过收缩 / 收敛到 fixed point → rollout 失败（the over-stabilization 发现）
4. **Demote 到 supporting**：C13、TLPS、cross-correlation
5. **必跑**：
   - **rank sweep** r ∈ {1, 2, 4, 8, 16}（已在 v4 plan 里）— 救 C13
   - **λ sweep** for randdiff — 反驳 "λ=0.01 lucky"
   - **第二个 dataset** — 反驳 "Push-T only"（最大的攻击面，但成本最高）
   - **第二个 encoder seed Stage-1**（v4 B2 已经覆盖）

### Diagnostic 在 paper 里如何用

- **TLPS, curvature**：和 Wang et al. 同源 → 引用他们，把我们的当 "diagnostic, not loss"，篇幅控制在一段
- **DMD / \|λ_max\| / n_unstable**：reference Ruiz-Morales et al. 把 Koopman 框架引入 JEPA，但说"他们用 Koopman 解释 encoder 学到 invariants，我们用 DMD 揭示 predictor regularizer 的 contraction failure" → distinct angle
- **PR / cross-metric correlation**：纯辅助证据，不当主轴

---

## 7. 是否 abandon？

**否**。Tight framing 还能撑 ~6/10 paper（workshop 没问题，正会要看 v4 怎么落）。

**关键 caveat**: v4 必须跑 rank sweep + λ sweep，否则 C13 和 C4 都站不住，最终 paper 就只剩 C2 一根支柱。

### Sources (Codex cited)
- [Temporal Straightening for Latent Planning, arXiv:2603.12231](https://arxiv.org/abs/2603.12231)
- [Koopman Invariants as Drivers of Emergent Time-Series Clustering in JEPAs, arXiv:2511.09783](https://arxiv.org/abs/2511.09783)
- [Nuclear Norm Regularization for Deep Learning, arXiv:2405.14544](https://arxiv.org/abs/2405.14544)
- [Learning Invariant Visual Representations for Planning with JEPA-WMs, arXiv:2602.18639](https://arxiv.org/abs/2602.18639)
- [TD-JEPA, arXiv:2510.00739](https://arxiv.org/abs/2510.00739)
- [DINO-WM, arXiv:2411.04983](https://arxiv.org/abs/2411.04983)
- [What Drives Success in Physical Planning with JEPA-WMs, arXiv:2512.24497](https://arxiv.org/abs/2512.24497)
- [LeWorldModel, arXiv:2603.19312](https://arxiv.org/abs/2603.19312)

---

## 8. Attack → v5.2 block mapping (added post-Codex)

每条 reviewer 攻击对应 v5.2 plan 里哪个实验块 / pre-registered gate 去反驳：

| # | Codex attack | v5.2 block | Anti-claim ID | Pre-registered gate |
|---|---|---|---|---|
| 1 | "Causal 措辞太强" | (framing fix in writing — use "mechanistic", not "causal") | — | n/a |
| 2 | "Frozen-encoder predictor training 已是常规 (DINO-WM)" | (citation fix — lead with DINO-WM, position our contribution as diagnostic not technique) | — | n/a |
| 3 | "TLPS already in Wang et al.; DMD/Koopman 已在 Ruiz-Morales" | **Demoted** TLPS / DMD / cross-corr to appendix (per §4 above) | — | n/a (scope cut) |
| 4 | "Push-T only, 1 arch, 1 rank, small seeds" | **B9** PointMaze (Anti-K), **B10** DINOv2 (Anti-J), **B3** rank sweep (Anti-D), **B0'** n=7 noise floor (Anti-F) | Anti-K, Anti-J, Anti-D, Anti-F | P-Anti-K, P-Anti-J, P-C13, σ_noise |
| 5 | "Post-hoc metrics; preregistered test 在哪？" | All P-* pre-registered in plan before runs; B0' provides σ_noise so 2σ thresholds 不是 post-hoc | — | P-C2, P-Cdyn, P-Anti-* |
| 6 | "Rank-4 没 rank sweep → numerology" | **B3** rank sweep r ∈ {1, 2, 4, 8, 16} × 2 seeds | Anti-D | P-C13: argmin in r ∈ {2, 4} |
| 7 | "Reversal 可能 optimizer/schedule-dependent" | **B7** LR / optimizer robustness (4-cell check) | Anti-E | P-Anti-E |

> Anti-A (ε mis-calibration) → B1; Anti-B (Stage-1 shape) → B2; Anti-C (λ-specific) → B4 monotone; Anti-H ((ε,λ) interaction) → B5'; Anti-I (8K under-training) → B4'; Anti-L (Stage-1 seed) → B8 — 这些是 plan 自查后追加的，未在 Codex 原 attack list 里。

---

## 9. v3/v4 → v5 pivot summary

| Aspect | v3/v4 framing | v5 framing (post-pivot) |
|---|---|---|
| Headline | "Lean LeWM via low-rank predictor regularization" | "Encoder-mediated compensation can mask and even invert predictor-side regularization effects" |
| Pillar 1 | C13 (rank=4 matches intrinsic dim) | **C2 (sign reversal under freeze)** |
| Pillar 2 | C3 (uvlowr Pareto improvement) | **C_dyn (nuclear-norm → over-contraction)** |
| Demoted | — | C13 (supporting), TLPS, cross-corr, C6 |
| Differentiator | "We try low-rank in JEPA" (overlaps with TD-JEPA arXiv:2510.00739) | "Sign-reversed predictor-Jacobian under freeze/unfreeze + λ dose-response of \|λ_max\|" (not found in prior art) |

---

## 10. Open follow-ups

- **Cross-dataset replication** beyond Push-T 是 attack #4 的最大攻击面 → v5.2 promoted **B9 (PointMaze → TwoRoom fallback)** 和 **B10 (DINOv2)** to MUST. As of 2026-05-14, TwoRoom 6 runs 完成（`v5.2_autodl_analysis.md`），P-Anti-K 看起来 PASS (randdiff σ₁=1.519 < baseline σ₁=1.678, ~8σ)。
- **"Causal" 语言**需要在写论文时全文 sweep 改为 "mechanistic"。
- v5.2 **B4'-40K spectral 补充**（2026-05-14 通过 `spectral_postprocess.py` 完成）：baseline \|λ_max\|=1.79 / uvlowr \|λ_max\|=1.41 → 比 σ₁ 差距敏感 3.5×，强化 attack #6 反驳。

---

## 11. 引用本文档的位置

- `EXPERIMENT_PLAN_v5.md` §"Problem" — 引用 §4 的 headline grade
- `EXPERIMENT_PLAN_v5.md` §"Pivot summary (per novelty check)" — 使用 §9 的 table
- `EXPERIMENT_PLAN.md` (v5.2) anti-claim table — 由 §8 attack mapping 推出
- `EXPERIMENT_PLAN.md` §"Demoted to supporting (per novelty check)" — 来自 §6 推荐
