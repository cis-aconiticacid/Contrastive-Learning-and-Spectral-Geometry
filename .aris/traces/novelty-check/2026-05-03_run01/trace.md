# Novelty-Check Codex Trace

- Date: 2026-05-03
- Skill: novelty-check
- Run: 01
- Reviewer: Codex MCP, model_reasoning_effort=xhigh
- Thread ID: 019debfb-1d60-7d02-a70c-22d322967bca
- Topic: SAE-Manifold × GRAPE bridge (readme.md proposal)

## Reviewer verdict (summary)

- Overall: 6/10, PROCEED WITH CAUTION
- Claim 1 (hidden-duality framing): MEDIUM — synthesis, not discovery
- Claim 2 (causal hypothesis: geometric PE → less SAE dilution): **HIGH — the real novelty**
- Claim 3 (mechanism story: RoPE commuting structure → dilution): **LOW — danger zone**, because SAE-Manifold itself blames SAE design
- Claim 4 (evaluation framework: SAE-Manifold metrics for PE): HIGH — but risk of being seen as benchmarking

## Most dangerous reviewer attack

> Projecting Assumptions + SAE-Manifold already imply that dilution may be probe-induced; the observed PE effect may reflect what the SAE can see, not what the model actually represents. If the experiment doesn't deconfound that, the paper collapses.

## Papers Codex flagged that I missed (verified real)

1. Ruscio, Nanni, Silvestri — *Beyond Position: emergence of wavelet-like properties in Transformers* (ACL 2025, arXiv:2410.18067) — RoPE transformers spontaneously develop multi-resolution wavelet structure as compensation. Cuts both ways.
2. Yin & Penghang — *Frayed RoPE and Long Inputs: A Geometric Perspective* (arXiv:2603.18017) — geometric analysis of RoPE failure modes.
3. Veisi, Fartoot, Amirzadeh — *Context-aware Rotary Position Embedding* (CARoPE, arXiv:2507.23083) — input-dependent RoPE frequencies; another GRAPE baseline.
4. Irie et al. — *Learning interpretable positional encodings...depends on initialization* (ICML 2025 workshop, arXiv:2406.08272) — **closest precedent to claim 4: explicitly links PE choice to interpretability and generalization.** Not direct overlap (no SAE, no LLM scale, no manifold metrics), but the natural co-citation.

## Codex correction

PaTH ≠ basis for GRAPE-AP. PaTH is multiplicative data-dependent Householder; GRAPE-AP is additive path-integral. Close in spirit, not the same object. **My lit-review claim "PaTH should be a baseline" stands, but the framing "PaTH is the inspiration for GRAPE-AP" should be removed.**

## Suggested reframing

- Strong frame: *"Does pretraining positional geometry change downstream manifold recoverability?"*
- Weak frame to avoid: *"We discovered the true cause of SAE dilution."*

## Full reviewer reply

(Recorded inline above; verbatim Codex output is the assistant message in the parent thread. This trace summarizes the actionable content.)
