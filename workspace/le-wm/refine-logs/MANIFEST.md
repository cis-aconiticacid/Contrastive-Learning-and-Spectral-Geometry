# Manifest — refine-logs outputs

| Date | File | Skill | Purpose |
|---|---|---|---|
| 2026-05-11 (earlier session) | `CLAIM.md` | research-refine | Claim discovery, updated post-v3 (C1-C13 with v2+v3 verdict status) |
| 2026-05-11 (earlier session) | `EXPERIMENT_PLAN.md` | experiment-plan | v3 frozen-encoder ablation plan (with post-execution REVISIONS) |
| 2026-05-11 (earlier session) | `EXPERIMENT_TRACKER.md` | experiment-plan | v3 run-by-run with outcomes |
| 2026-05-11 (earlier session) | `c13_pusht_pca.json` | cheap-check | C13 state/proprio/action PCA |
| 2026-05-11 (earlier session) | `c13b_latent_delta.json` | cheap-check | C13 post-projector latent-delta PCA |
| 2026-05-11 (earlier session) | `c6_vit_jacobian_control.json` | cheap-check | C6 random/ImageNet/JEPA ViT Jacobian dichotomy |
| **2026-05-12** | **`EXPERIMENT_PLAN_v4.md`** | experiment-plan | v4 plan: discharge v3 confounds (ε sweep, uvlowr-Stage-1 freeze, rank sweep) |
| **2026-05-12** | **`EXPERIMENT_TRACKER_v4.md`** | experiment-plan | v4 run table (R100-R150) with TODO status, decision gates, operational checklist |
| 2026-05-12 | `MANIFEST.md` | experiment-plan | This file (originally) |
| 2026-05-12 | `tier0_pr_vs_step.png` | tier-0 analysis | PR trajectory v2 vs v3, 3 variants |
| 2026-05-12 | `tier0_spectral_gap.png` | tier-0 analysis | σ₁/σ₁₀ over training |
| 2026-05-12 | `tier0_sv_decay_final.png` | tier-0 analysis | log-log SV decay final-step, 3 variants |
| 2026-05-12 | `tier0_results.json` | tier-0 analysis | PR/gap/β per variant×seed×step |
| 2026-05-12 | `tier0_correlation_matrix.png` | tier-0 cross-metric | 3-panel correlation: v2 / v3 / pooled (n=9/6/15) |
| 2026-05-12 | `tier0_correlation_matrix.json` | tier-0 cross-metric | Pearson correlations + raw per-run data |
| 2026-05-12 | `v4_eps_bracket.json` | R100 probe | ε bracket selection {0.0125, 0.05, 0.2, 0.8} from frozen latent norm |
| 2026-05-12 | `tier1_traj/predicted_metrics.json` | tier-1 trajectory | TLPS/curvature/DMD/TwoNN/R²_h per variant (3K-step local repro) |
| 2026-05-12 | `tier1_traj/predicted_metrics_summary.csv` | tier-1 trajectory | Headline table |
| 2026-05-12 | `tier1_traj/{baseline,uvlowr,randdiff}_traj.npz` | tier-1 trajectory | predicted_z + encoded_z + state_gt per variant |
| 2026-05-12 | `tier1_decay_curves.png` | tier-1 trajectory | R²_h + TLPS bars + |λ_max| bars |
| 2026-05-12 | `tier1_dmd_spectrum.png` | tier-1 trajectory | 4-panel complex-plane DMD eigenvalues + unit circle |
| 2026-05-12 | `tier1_velocity_distributions.png` | tier-1 trajectory | ‖v‖ / ‖a‖ histograms |
| 2026-05-12 | `tier1_correlation_matrix.png` | tier-1 trajectory | Per-traj metric × metric correlation |
| **2026-05-12** | **`EXPERIMENT_PLAN_v5.md`** | experiment-plan | v5 plan (intermediate, full-detail with execution playbook) |
| **2026-05-12** | **`EXPERIMENT_TRACKER_v5.md`** | experiment-plan | v5 tracker with R200-R293, pre-registered predictions P-C2 / P-Cdyn / P-C13 |
| **2026-05-12** | **`EXPERIMENT_PLAN.md` (canonical)** | experiment-plan | **PROMOTED v5 + execution playbook (Hydra commands, wall-clock schedule, decision tree)** |
| **2026-05-12** | **`EXPERIMENT_TRACKER.md` (canonical)** | experiment-plan | **PROMOTED v5 tracker** |
| **2026-05-12** | `EXPERIMENT_PLAN_v3.md` | experiment-plan | Archived v3 plan (preserves CLAIM.md / ANALYSIS.md references) |
| **2026-05-12** | `EXPERIMENT_TRACKER_v3.md` | experiment-plan | Archived v3 tracker |
| **2026-05-12** | `/workspace/le-wm/run_pipeline_v5.sh` | experiment-plan | v5 orchestrator: M0/B1/B2/B3/B4/B5/B6 with reuse-from-v3 skip logic |
| **2026-05-12** | **`EXPERIMENT_SPEC.md`** | experiment-plan | **DETAILED technical spec — model arch (every layer's params), regularizer math (SIGReg/uvlowr/randdiff exact formulas), all metric definitions, probing protocol, DMD algorithm, v3 baseline reference numbers for reuse-anchor points, expected effect-vs-noise table** |
| 2026-05-12 | EXPERIMENT_PLAN.md (v5.1 update) | experiment-plan | **post-self-review revisions**: B0' noise floor, B4' 40K extension, B5' (ε,λ) corners, B5 nonlinear ρ, M1→M2 abort gate, Stage-1 convergence gate, NDE/DEQ contrast motivation, B10 DINOv2 cross-pretraining (NICE), Anti-claims F/G/H/I/J |
| 2026-05-12 | EXPERIMENT_SPEC.md (v5.1 update) | experiment-plan | §6.6.b nonlinear ρ algorithm, §8 P-* with σ_noise, §11b B0'/B4'/B5'/B10 specs |
| 2026-05-12 | EXPERIMENT_TRACKER.md (v5.1) | experiment-plan | R202-R204 (B0'), R256 (B4'), R261-R264 (B5'), R300-R303 (B10) |
| 2026-05-12 | `/workspace/le-wm/tier1_dinov2_adapter.py` | experiment-plan | B10 DINOv2 ViT-S/14 encoder swap: build_adapter() + DINOv2Encoder + DINOv2JEPA + train loop |
| **2026-05-14** | **`NOVELTY_CHECK.md`** | (manual consolidation) | **Canonical novelty-check doc**: Codex headline grade, prior art (7 papers), reviewer attack list (7), attack→v5.2 block mapping, v3/v4→v5 pivot summary. Source: `.aris/traces/novelty-check/2026-05-12_run01/codex_trace.md` |
| 2026-05-14 | `v5.2_autodl_analysis.md` | (analysis agent) | P-Anti-I partial + P-Anti-K complete from 9 AutoDL runs (B4' Push-T 40K + TwoRoom 6-run cross-dataset) |
| 2026-05-14 | `/workspace/le-wm/spectral_postprocess.py` | (instrumentation) | Post-hoc 192×192 latent-only predictor-Jacobian eigendecomp + TwoNN, writes back to `final_analyses.json`. Covers \|λ_max\|/n_unstable/spectral_abscissa fields the training-time probe omitted |
| 2026-05-14 | `v5.2_results_analysis.md` (also dated `_20260514`) | analyze-results | Raw tables (B4' 40K, TR 6-run, v3 anchor, rollout) + 5 findings (P-Anti-K PASS, \|λ_max\| 3.5× sensitive than σ₁, P-Anti-I uvlowr branch PASS, uvlowr h≥15 rollout divergence, TR uvlowr seed1 outlier) + P-gate status + next-step priority |
| **2026-05-15** | `randdiff_20K_results_20260515.md` | (local exec) | Push-T frozen randdiff 20K-step result: \|λ_max\|=0.984±0.017 (n=6), rollout MSE h=20 = 2.16 (50× worse than baseline), P-Anti-I randdiff branch partial PASS |
| **2026-05-15** | `twonn_pred_results_20260515.md` | (instrumentation + analysis) | Full 9-ckpt TwoNN(enc) + TwoNN(pred) + \|λ_max\| matrix across frozen/joint settings. Fixed strided-sampling bug (0.45→3-6 range) and added predictor-source TwoNN. Five new findings: joint train is already stable; randdiff Δ(pred-enc) universally smallest; uvlowr frozen-vs-joint reversal; TwoNN sensitivity ladder; framing constraints |
| **2026-05-15** | `/workspace/le-wm/spectral_postprocess.py` (v2) | (instrumentation) | Two fixes: (1) strided cross-trajectory sampling in `collect_latents`; (2) added `source={"encoder","predictor"}` parameter and writes both `twonn_enc_intrinsic_dim` + `twonn_pred_intrinsic_dim` to final_analyses.json |
