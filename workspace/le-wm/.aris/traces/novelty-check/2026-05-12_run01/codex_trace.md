# Codex novelty-check trace
- Date: 2026-05-12
- Reviewer: gpt-5.4 (mcp__codex__codex, xhigh reasoning)
- Thread ID: 019e1a53-cb09-7d83-be95-c6333091d1e0

## Headline
- Broad framing 4/10; tight mechanistic framing 6/10
- DO NOT abandon, abandon only weak framing

## Key prior-art papers flagged (NEW vs my search)
- Learning Invariant Visual Representations for Planning with JEPA-WMs, arXiv:2602.18639 — augmented predictive objective reshapes representation, reduces latent size
- TD-JEPA, arXiv:2510.00739 — connects JEPA to low-rank long-term dynamics
- DINO-WM (PMLR 2025) arXiv:2411.04983 — established frozen-encoder + predictor-only training as standard
- What Drives Success in Physical Planning with JEPA-WMs, arXiv:2512.24497 — studies rollout losses + predictor architecture (but NOT low-rank / nuclear-norm)
- Eigenvalue init/reg for Koopman AEs arXiv:2212.12086
- Learnable Koopman-Enhanced Transformer arXiv:2602.02592
- Continuous-Time Koopman AEs arXiv:2602.02832

## Recommended positioning (verbatim)
- Lead with "mechanistic misattribution under joint training"
- Lead with "sign reversal under frozen-vs-joint intervention"
- Second pillar: "over-contractive fixed-point failure under randdiff"
- Demote: uvlowr rank=4, TLPS

## Reviewer attack list (anticipate)
1. "Your 'causal' language is too strong"
2. "Frozen-encoder predictor training is already common"
3. "TLPS already in Wang et al.; DMD/Koopman already in Ruiz-Morales et al."
4. "Push-T-only, one architecture, one rank, small seeds"
5. "Many post-hoc metrics; where is the minimal preregistered test?"
6. "Rank-4 alignment not convincing without rank sweep"
7. "Maybe reversal is optimizer/schedule-dependent"
