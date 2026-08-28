# Literature Review: Geometric Position Encodings × SAE Manifold Capture

**Topic:** A hidden duality between SAE-Manifold (Bhalla et al. 2026) and GRAPE (Zhang et al. ICLR 2026) — does Lie-group-structured position encoding improve SAE manifold capture?
**Compiled:** 2026-05-03
**Sources scanned:** arXiv (web), Semantic Scholar (web), OpenReview, NeurIPS/ICLR proceedings. No local PDFs in `/workspace`. Zotero/Obsidian not configured.

---

## 1. Anchor papers (the proposal's two pillars)

| Paper | Venue | Method | Key Result | Relevance to Us | Source |
|-------|-------|--------|------------|-----------------|--------|
| Bhalla, Fel, Rager, Feucht, Haklay, Wurgaft, Boppana, Kowal, Shyam, Merullo, Geiger, Lubana — *Do Sparse Autoencoders Capture Concept Manifolds?* (2026, arXiv:2604.28119) | preprint | Tests every major SAE family (ℓ₁, JumpReLU, TopK, BatchTopK, Matryoshka) for two failure modes: "global" (compact atom-group spans manifold) vs "local" (tiled detectors) capture. Identifies a **dilution regime** where SAEs sit in a fragmented mix of both. | All current SAE architectures are stuck in dilution; manifolds must be recovered post-hoc via Ising-style atom grouping. Loss rewards reconstruction, not geometric coherence. | **The dependent variable** of the proposal. We need to reproduce their dilution metric and use it as the comparison probe. | web |
| Zhang, Chen, Liu, Qin, Yuan, Xu, Yuan, Gu, Yao — *GRAPE: Group Representational Position Encoding* (ICLR 2026, arXiv:2512.07805) | ICLR 2026 | Unifies multiplicative SO(d) rotations and additive GL unipotent biases under a single Lie-group action **G(n) = exp(n·ω·L)**. RoPE = commuting, coordinate-aligned, log-uniform spectrum special case. ALiBi/FoX = rank-1 unipotent special cases. **GRAPE-M** adds rank-2 non-commuting mixtures; **GRAPE-AP** is path-integral / accumulated-path. Trained on fineweb-edu, evaluated on WikiText-103, PG19, RULER, needle-in-haystack vs RoPE/ALiBi/FoX/NoPE/YaRN/RoPE-Mixed/LieRE. | **The independent variable** of the proposal. Provides a tunable family of geometric priors so we can sweep "amount of Lie structure injected" while holding loss/data fixed. | web |

---

## 2. SAE concept-geometry lineage (what SAE-Manifold builds on)

| Paper | Venue | Method | Key Result | Relevance to Us | Source |
|-------|-------|--------|------------|-----------------|--------|
| Engels, Michaud, Liao, Gurnee, Tegmark — *Not All Language Model Features Are One-Dimensionally Linear* (ICLR 2025, arXiv:2405.14860) | ICLR 2025 | Uses SAEs on GPT-2 / Mistral 7B / Llama 3 8B; intervention experiments isolate 2-D **circular features for days-of-week and months-of-year** that drive modular-arithmetic behavior. | Direct empirical refutation of the strict LRH; first canonical multi-dim feature. **The original "clock" finding** that SAE-Manifold formalizes. | Establishes that the manifolds we hope to recover *exist in real models*; gives a concrete probe target (modular-arithmetic circles). | web |
| Park, Choe, Veitch — *The Linear Representation Hypothesis and the Geometry of Large Language Models* (arXiv:2311.03658) | preprint → ICML 2024 | Formalizes LRH causally; gives an inner-product structure aligned with concept inheritance. | Provides the LRH baseline that SAE-Manifold and Engels et al. generalize. | Useful framing for "LRH is the manifold-dim-1 special case." | web |
| Li, Michaud, Tegmark — *The Geometry of Concepts: Sparse Autoencoder Feature Structure* (arXiv:2410.19750, Entropy 2025) | Entropy | Three-scale geometry of SAE dictionaries: atomic "crystals" (parallelograms beyond king-queen), brain-scale "lobes" (math/code clusters), galactic-scale eigenspectrum. | Shows non-trivial structure already lives in the dictionary; complementary to the Bhalla critique that single atoms still miss the manifold. | Background — argues that even fragmented dictionaries carry latent geometric organization. | web |
| Modell, Rubin-Delanchy, Whiteley — *The Origins of Representation Manifolds in Large Language Models* (arXiv:2505.18235) | preprint | Argues cosine similarity ≈ shortest on-manifold geodesic; proposes feature-space → concept-space distance correspondence. | Theoretical justification for treating representations as Riemannian objects, not vector dictionaries. | Gives the geometric vocabulary (intrinsic geometry, geodesics) the proposal will need to phrase its evaluation. | web |
| Michaud, Gorton, McGrath — *Understanding sparse autoencoder scaling in the presence of feature manifolds* (arXiv:2509.02565) | preprint | Adapts capacity-allocation theory; identifies a regime where **multi-dim features cause SAEs to learn far fewer features than they have latents** — a quantitative form of dilution. | A scaling-law sibling of SAE-Manifold; predicts that "amount of dilution" depends on number/curvature of underlying manifolds. | Suggests the proposal should report results as a function of latent count, not just at one width. | web |
| Hindupur, Lubana, Fel, Ba — *Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry* (NeurIPS 2025, arXiv:2503.01822) | NeurIPS 2025 | Bilevel-optimization view: "an SAE does not just reveal concepts — it determines what can be seen at all." Different SAE architectures expose different concepts; SAEs fail when concepts are heterogeneously dimensional or non-linearly separable. | **Closest sibling to our hypothesis**, but on the other side: it varies *the SAE* and holds the model fixed. The proposal varies the *model* and holds the SAE fixed. Authors overlap with SAE-Manifold (Lubana, Fel) — likely the same research line. | The natural co-citation. We should frame the proposal as "the dual move": instead of changing the lens, change the object. | web |
| Dooms, Gauderis — *Finding Manifolds With Bilinear Autoencoders* (arXiv:2510.16820) | preprint | Quadratic/polynomial decoder replaces linear SAE atoms; targets curved structure directly. | One of the SAE-architecture-side answers to dilution. | Useful baseline: if model-side fix (GRAPE) reduces dilution as much as a manifold-aware decoder, that supports the proposal's punchline. | web |
| Various — *SAEBench / SynthSAEBench / Position: Feature Consistency in SAEs* (arXiv:2503.09532, 2602.14687, 2505.20254) | misc 2025–26 | Benchmarks for SAE evaluation, feature consistency across runs, synthetic ground-truth feature recovery (incl. feature manifolds). | Provides ready-made evaluation harnesses; SynthSAEBench in particular has a manifold benchmark we could adopt. | Gives us off-the-shelf metrics so we don't reinvent the dilution measure. | web |

---

## 3. Lie-group / geometric position encoding lineage (what GRAPE builds on)

| Paper | Venue | Method | Key Result | Relevance to Us | Source |
|-------|-------|--------|------------|-----------------|--------|
| Su et al. — *RoFormer / RoPE* (arXiv:2104.09864) | NCLP 2024 (Neurocomputing) | Block-diagonal SO(2) rotations per pair of channels; commuting, coordinate-aligned. | The default position encoding in modern LLMs (Llama, Mistral, Gemma, etc.). | The control condition. SAE-Manifold's findings are all on RoPE-trained models — so we already have the "RoPE → dilution" half of the chart. | web |
| Press, Smith, Lewis — *ALiBi* | ICLR 2022 | Linear additive distance penalty; nilpotent action under GRAPE's lens. | One of two unipotent special cases inside GRAPE-A. | Useful as a non-rotational PE baseline — distinguishes "rotation matters" from "any non-RoPE PE matters." | web |
| Lin, Xu et al. — *Forgetting Transformer (FoX)* (ICLR 2025 / COLM 2025, arXiv:2503.02130) | ICLR 2025 | Data-dependent forget-gate down-weighting of attention; needs no positional embedding. | Subsumed by GRAPE-A as another nilpotent unipotent case. | A non-PE alternative that nonetheless inherits position-like behavior — a clean control if we want "no explicit PE" in the sweep. | web |
| Ostmeier, Axelrod, Varma, Moseley, Chaudhari, Langlotz — *LieRE* (ICML 2025, arXiv:2406.10322) | ICML 2025 | Replaces fixed 2-D RoPE blocks with **learned dense skew-symmetric generators** → general SO(d) rotations. Validated on 2-D / 3-D vision. | The most direct precursor to GRAPE-M; GRAPE generalizes LieRE to commuting + non-commuting + additive. | Critical baseline. If a learned-generator PE alone (LieRE) already changes SAE manifold capture, the result is mostly about expressivity, not about the Lie-group structure per se. | web |
| Yang, Shen et al. — *PaTH Attention: Position Encoding via Accumulating Householder Transformations* (NeurIPS 2025, arXiv:2505.16381) | NeurIPS 2025 | Cumulative product of data-dependent Householder reflections — non-commutative, non-invertible, not simultaneously diagonalizable. Path-dependent: order of tokens matters. | This **is** the inspiration for GRAPE-AP (path-integral / accumulated-path). The two are closely related and PaTH should be a baseline. | If PaTH already produces "more manifold-like" representations (by virtue of path dependence), the GRAPE-AP advantage may be inherited from PaTH rather than from the Lie-algebra parameterization. | web |
| van de Geijn, Lüddecke, Turishcheva, Ecker — *A Circular Argument: Does RoPE need to be Equivariant for Vision?* (arXiv:2511.08368) | preprint | Proposes Spherical RoPE (non-commutative); finds equivariance is **not** essential for vision PE quality. | Cautionary precedent: in vision, breaking RoPE's symmetry didn't help. We should be ready for the same null on language. | A direct prior null result on a closely related question. The proposal must explain why language SAEs would behave differently. | web |
| Heo, Park et al. — *Rotary Position Embedding for Vision Transformer (RoPE-Mixed)* (ECCV 2024, arXiv:2403.13298) | ECCV 2024 | Learnable mixed-axis RoPE frequencies for ViTs. | Another RoPE generalization GRAPE subsumes. | Mostly background. | web |
| Selective Rotary Position Embedding (arXiv:2511.17388) | preprint 2025 | Token-selective RoPE application. | Another sibling. | Useful if we want a high-coverage baseline table. | web |

---

## 4. The bridge — does anyone connect PE choice to SAE / interpretability?

**This is the gap the proposal targets, and it is genuinely empty.** Searches across `("position encoding" OR RoPE) × (SAE OR interpretability OR feature geometry)` returned only:

- The "Projecting Assumptions" paper (Hindupur et al., NeurIPS 2025) — same intuition (architecture constrains visible concepts) but **architecture = SAE, not base model**.
- Literature on RoPE's *internal* effects (e.g. NeurIPS 2024 *What Rotary Position Embedding Can Tell Us*; arXiv:2505.13027 *Unpacking Positional Encoding*) — these probe attention heads/spectra, not SAE feature geometry.
- Position-bias papers (arXiv:2502.01951) — characterize attention patterns, not learned features.

**No paper in the searched corpus evaluates SAE feature geometry across PE-varied models.** The proposal would be the first.

---

## 5. Synthesis — landscape map and where the proposal lands

**Two streams have converged on Lie-group structure from opposite directions, exactly as the README claims.**

On the **interpretability side**, the field moved from "concepts are linear directions" (Park & Choe 2023) to "some concepts are 2-D circular manifolds" (Engels et al. 2024) to "almost all interesting concepts may be manifolds, and current SAEs systematically fail to capture them" (Bhalla et al. 2026). The mechanism of failure has a name now — *dilution* — and a quantitative scaling description (Michaud et al. 2025). The intermediate paper most relevant to this proposal is Hindupur, Lubana, Fel & Ba's *Projecting Assumptions* (NeurIPS 2025), which establishes that SAE architecture biases *which* concepts are visible. Lubana and Fel are co-authors on both *Projecting Assumptions* and SAE-Manifold — this is a coherent research program asking "why do SAEs fail on geometric structure?" and so far answering on the lens side.

On the **architecture side**, RoPE's commuting, coordinate-aligned rotations have been progressively generalized: LieRE (Ostmeier et al., ICML 2025) introduced learned dense skew-symmetric generators; PaTH Attention (Yang et al., NeurIPS 2025) added data-dependent, non-commutative Householder accumulation; GRAPE (Zhang et al., ICLR 2026) unified all of these — plus ALiBi and FoX — under one Lie-group action **G(n) = exp(n·ω·L)**. Every empirical justification of these generalizations to date is on perplexity, length extrapolation, or RULER — i.e. *capability* metrics. **No paper has yet asked whether the richer geometric prior produces more interpretable internal representations.**

**The gap the proposal targets is real.** The natural co-citation, *Projecting Assumptions*, makes the dual move (vary SAE, hold model fixed); the proposal makes the symmetric move (vary model, hold SAE fixed). The cleanest framing for the writeup is: "Hindupur et al. asked which concepts a given lens *can* see; we ask which concepts a given object *makes visible*." This positioning makes the proposal complementary rather than competitive with the active SAE-Manifold research line, and gives a natural collaboration target (Lubana / Fel).

**Risks visible in the literature.**
1. *Vision precedent is a null.* van de Geijn et al. (2511.08368) tried Spherical RoPE on vision and found equivariance didn't help. Language ≠ vision, but the proposal should pre-register why language SAE manifolds (which carry semantic, not spatial, structure) would respond differently.
2. *Confound with PaTH/LieRE expressivity.* If GRAPE-trained models show better manifold capture, the effect could be inherited from raw expressivity (more parameters in the PE) rather than from the Lie-algebra structure. The experimental design needs LieRE and PaTH as separate baselines, not just RoPE.
3. *Scaling regime matters.* Michaud et al. (2509.02565) shows that dilution depends on latent count vs feature manifold count. Any single-width SAE comparison risks reporting a regime-dependent artifact. Sweep latent counts.
4. *SAE-Manifold's own metric is post-hoc Ising recovery.* That metric is not yet standardized; SynthSAEBench (2602.14687) and SAEBench (2503.09532) may give cleaner alternatives.

**Suggested experimental skeleton (informed by the literature).**
- Train ≥4 small models matched on params/data/loss-curve, varying only PE: {RoPE, ALiBi, LieRE, GRAPE-M, GRAPE-AP/PaTH}.
- Train identical SAE family (TopK + Matryoshka, the two cleanest) on each, sweep latent count.
- Evaluate on (a) the modular-arithmetic days/months task from Engels et al. as a known-manifold probe, (b) SAE-Manifold's dilution metric, (c) SynthSAEBench manifold subscores. Report all three.
- A null on (a) but positive on (b)-(c), or vice versa, would itself be informative.

---

## 6. References (BibTeX)

```bibtex
@article{bhalla2026sae,
  title={Do Sparse Autoencoders Capture Concept Manifolds?},
  author={Bhalla, Usha and Fel, Thomas and Rager, Can and Feucht, Sheridan and Haklay, Tal and Wurgaft, Daniel and Boppana, Siddharth and Kowal, Matthew and Shyam, Vasudev and Merullo, Jack and Geiger, Atticus and Lubana, Ekdeep Singh},
  journal={arXiv preprint arXiv:2604.28119},
  year={2026}
}

@inproceedings{zhang2026grape,
  title={Group Representational Position Encoding},
  author={Zhang, Yifan and Chen, Zixiang and Liu, Yifeng and Qin, Zhen and Yuan, Huizhuo and Xu, Kangping and Yuan, Yang and Gu, Quanquan and Yao, Andrew Chi-Chih},
  booktitle={ICLR},
  year={2026},
  note={arXiv:2512.07805}
}

@inproceedings{engels2024notall,
  title={Not All Language Model Features Are One-Dimensionally Linear},
  author={Engels, Joshua and Michaud, Eric J. and Liao, Isaac and Gurnee, Wes and Tegmark, Max},
  booktitle={ICLR},
  year={2025},
  note={arXiv:2405.14860}
}

@inproceedings{hindupur2025projecting,
  title={Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry},
  author={Hindupur, Sai Sumedh R. and Lubana, Ekdeep Singh and Fel, Thomas and Ba, Demba},
  booktitle={NeurIPS},
  year={2025},
  note={arXiv:2503.01822}
}

@article{li2025geometry,
  title={The Geometry of Concepts: Sparse Autoencoder Feature Structure},
  author={Li, Yuxiao and Michaud, Eric J. and Tegmark, Max},
  journal={Entropy},
  volume={27}, number={4}, pages={344}, year={2025},
  note={arXiv:2410.19750}
}

@article{michaud2025scaling,
  title={Understanding sparse autoencoder scaling in the presence of feature manifolds},
  author={Michaud, Eric J. and Gorton, Liv and McGrath, Tom},
  journal={arXiv preprint arXiv:2509.02565},
  year={2025}
}

@article{modell2025origins,
  title={The Origins of Representation Manifolds in Large Language Models},
  author={Modell, Alexander and Rubin-Delanchy, Patrick and Whiteley, Nick},
  journal={arXiv preprint arXiv:2505.18235},
  year={2025}
}

@inproceedings{ostmeier2025liere,
  title={LieRE: Lie Rotational Positional Encodings},
  author={Ostmeier, Sophie and Axelrod, Brian and Varma, Maya and Moseley, Michael E. and Chaudhari, Akshay and Langlotz, Curtis},
  booktitle={ICML},
  year={2025},
  note={arXiv:2406.10322}
}

@inproceedings{yang2025path,
  title={PaTH Attention: Position Encoding via Accumulating Householder Transformations},
  author={Yang, Songlin and Shen, Yikang and others},
  booktitle={NeurIPS},
  year={2025},
  note={arXiv:2505.16381}
}

@inproceedings{lin2025forgetting,
  title={Forgetting Transformer: Softmax Attention with a Forget Gate},
  author={Lin, Zhixuan and others},
  booktitle={ICLR},
  year={2025},
  note={arXiv:2503.02130}
}

@article{park2023lrh,
  title={The Linear Representation Hypothesis and the Geometry of Large Language Models},
  author={Park, Kiho and Choe, Yo Joong and Veitch, Victor},
  journal={arXiv preprint arXiv:2311.03658},
  year={2023}
}

@article{vandegeijn2025circular,
  title={A Circular Argument: Does RoPE need to be Equivariant for Vision?},
  author={van de Geijn, Chase and L{\"u}ddecke, Timo and Turishcheva, Polina and Ecker, Alexander S.},
  journal={arXiv preprint arXiv:2511.08368},
  year={2025}
}

@article{dooms2025bilinear,
  title={Finding Manifolds With Bilinear Autoencoders},
  author={Dooms, Thomas and Gauderis, Ward},
  journal={arXiv preprint arXiv:2510.16820},
  year={2025}
}
```
