# Novelty Check Report — SAE-Manifold × GRAPE Bridge

**Date:** 2026-05-03
**Source proposal:** `/workspace/readme.md`
**Reviewer:** Codex (gpt-5.4, xhigh reasoning) + cross-checked literature
**Trace:** `/workspace/.aris/traces/novelty-check/2026-05-03_run01/`

---

## Proposed Method (1 sentence)

Pretrain Transformers under matched compute/data with varied position encodings (RoPE, ALiBi, LieRE, GRAPE-M, GRAPE-AP / PaTH), then train identical SAEs on each and test whether richer Lie-group geometric priors in the PE produce **less dilution** on the SAE-Manifold benchmark.

---

## Core Claims

| # | Claim | Novelty | Closest Prior |
|---|-------|---------|---------------|
| 1 | "SAE-Manifold and GRAPE describe the same object (Lie-group orbits) from opposite sides" — the hidden-duality framing | **MEDIUM** | Synthesis of GRAPE (arXiv:2512.07805) + SAE-Manifold (arXiv:2604.28119) + Projecting Assumptions (arXiv:2503.01822) + Engels et al. (arXiv:2405.14860). "Same object" is too strong — GRAPE acts on q/k space, SAE-Manifold studies hidden states. Reviewers will accept it as a motivating lens, not core novelty. |
| 2 | Geometric PE → less SAE dilution (causal hypothesis) | **HIGH** | No direct prior found. Nearest ingredients: Projecting Assumptions (probe-induced visibility); LieRE / GRAPE / PaTH / CARoPE (PE generalizations). **This is the real novelty.** |
| 3 | Mechanism: RoPE's commuting/coordinate-aligned structure forces implicit cross-subspace coupling, which *causes* the dilution observed by SAE-Manifold | **LOW** | The "RoPE is restrictive" half is established (GRAPE, LieRE, PaTH, *Beyond Position* arXiv:2410.18067 ACL 2025, *Frayed RoPE* arXiv:2603.18017, *Unpacking PE* arXiv:2505.13027, *Circular Argument* arXiv:2511.08368). The jump from "restrictive" to "causes SAE dilution specifically" is unsupported, and **SAE-Manifold itself blames SAE design**, not base-model PE. **Danger zone.** |
| 4 | Use SAE-Manifold dilution as the evaluation metric for PE designs (instead of perplexity/RULER) | **HIGH** | Closest precedent **(missed in initial lit review)**: Irie et al., *Learning interpretable positional encodings depends on initialization*, ICML 2025 workshop, arXiv:2406.08272 — explicitly links PE choice/init to interpretability and generalization on toy + neuroscience tasks. Not LLM-scale, no SAE, no manifold metric, but it's the natural co-citation. |

---

## Closest Prior Work

| Paper | Year | Venue | Overlap | Key Difference |
|-------|------|-------|---------|----------------|
| Hindupur, Lubana, Fel, Ba — *Projecting Assumptions* | 2025 | NeurIPS | Same intuition: architecture biases visible concepts | Varies SAE, holds model fixed (proposal does the dual) |
| Bhalla, Fel, Lubana et al. — *SAE-Manifold* | 2026 | preprint | Provides the dependent variable (dilution metric) | Blames SAE design, not PE — proposal contradicts this attribution |
| Engels, Michaud, Liao et al. — *Not All LM Features Are 1-D Linear* | 2025 | ICLR | Provides the canonical manifold probe (days/months circles) | Doesn't vary PE |
| Ostmeier et al. — *LieRE* | 2025 | ICML | Critical baseline — learned Lie generators on vision | Vision-only, no SAE, no manifold metric |
| Yang, Shen et al. — *PaTH Attention* | 2025 | NeurIPS | Non-commuting path-dependent PE | No SAE eval; ≠ GRAPE-AP basis (correction from initial review) |
| Ruscio et al. — *Beyond Position: wavelet-like properties* | 2025 | ACL | RoPE models develop multi-resolution structure as compensation | **Cuts both ways**: supports mechanism story (compensation overhead) but undercuts (model already has the structure → dilution must be SAE-side) |
| Irie et al. — *Learning interpretable PEs depends on init* | 2025 | ICML wksp | Direct precedent for PE → interpretability link | Toy/neuroscience scale, no SAE, no LLM |
| van de Geijn et al. — *A Circular Argument* | 2025 | preprint | Tested whether RoPE equivariance matters; **null on vision** | Doesn't measure features/SAEs |

---

## Overall Novelty Assessment

- **Score: 6/10**
- **Recommendation: PROCEED WITH CAUTION**
- **Key differentiator:** No paper varies base-model PE and measures SAE feature geometry. Claims 2 + 4 carry the contribution.
- **Single biggest risk:** A reviewer will say *"Projecting Assumptions + SAE-Manifold already imply dilution may be probe-induced — your observed PE effect may just reflect what the SAE can see, not what the model represents."* Without an explicit deconfound, the paper collapses.

---

## Required Experimental Deconfounds (from this risk)

1. **Probe-induced vs representation-induced.** Run multiple SAE families (TopK + Matryoshka + Bilinear) on each PE-trained model. If the PE effect persists across SAE families it's a representation effect; if it tracks one family it's probe-induced.
2. **Direct geometric measurement that bypasses SAE.** On the Engels et al. days/months task, fit the 2-D circle directly via PCA/LDA on activations *before* SAE. If the circle is already cleaner under GRAPE without an SAE, that defeats the probe-induced critique.
3. **Capacity vs symmetry confound.** LieRE adds learned generators (capacity) without the path-integral structure; PaTH adds path dependence without the Lie-algebra parameterization. Treat them as independent baselines, not "another GRAPE variant."
4. **Latent-count sweep.** Michaud et al. (arXiv:2509.02565) showed dilution is regime-dependent. Report the metric as a curve, not a point.
5. **Compensation-overhead test (motivated by Ruscio et al. ACL 2025).** If RoPE models develop wavelet-like compensation in later layers and GRAPE models don't need to, that *is* the mechanism story — measure layer-wise effective rank or wavelet-consistency to test it directly.

---

## Suggested Positioning

**Use:** *"Does pretraining positional geometry change downstream manifold recoverability?"*
A measurement paper. Claims 2 + 4 are the contribution. Positions naturally next to Hindupur et al. as the *dual move*.

**Avoid:** *"We discovered the true cause of SAE dilution."*
Overclaims claim 3, which the experiments cannot establish. SAE-Manifold's own attribution will be cited against you.

---

## Corrections to the Prior Lit Review

Two errors in `/workspace/literature/literature_review.md` to fix:

1. **PaTH is not the basis for GRAPE-AP.** PaTH = multiplicative data-dependent Householder. GRAPE-AP = additive path-integral / accumulated bias. They are close in spirit, not the same object. (My §3 PaTH row claimed inheritance — should be reworded.)
2. **Missing co-citation.** Irie et al., *Learning interpretable positional encodings depends on initialization* (ICML 2025 workshop, arXiv:2406.08272) is the closest precedent to claim 4 and was missed.

Other newly-flagged additions to the corpus (verified real):
- Ruscio, Nanni, Silvestri — *Beyond Position: wavelet-like properties* (ACL 2025, arXiv:2410.18067)
- Yin & Penghang — *Frayed RoPE and Long Inputs* (arXiv:2603.18017)
- Veisi et al. — *CARoPE* (arXiv:2507.23083)
