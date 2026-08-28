# Contrastive Learning and Spectral Geometry

Archived research workspace containing two related but distinct exploratory
threads:

1. a proposal connecting geometric positional encodings, Lie-group structure,
   and concept manifolds in learned representations;
2. experiments on low-rank and spectral regularization in JEPA-style world
   models, including predictor/encoder Jacobian measurements.

The original directory was named `ûßfnŽ1`. This English title is used for
the archive without translating or strengthening its scientific claims.

## Evidence status

The world-model experiments produced several useful observations, but they do
not constitute a completed paper:

- predictor Jacobian rank was not monotonic with rollout quality in the tested
  configurations;
- the randdiff spectral signature reversed when the encoder was frozen,
  supporting an encoder-adaptation explanation;
- the apparent broad Pareto advantage of the low-rank predictor weakened under
  frozen-encoder controls;
- the frozen-encoder study retained confounds from encoder pretraining, epsilon
  calibration, a single Stage-1 seed, and limited training duration.

Read [`workspace/le-wm/refine-logs/CLAIM.md`](workspace/le-wm/refine-logs/CLAIM.md)
and the versioned analyses before quoting results. They record reversals,
falsifiers, and limitations alongside positive observations.

## Layout

- `research_notes/`: the original geometric-position-encoding proposal.
- `literature/`: literature and novelty reviews.
- `workspace/le-wm/`: upstream LeWM source plus local research modifications,
  experiment scripts, configurations, analyses, and compact results.
- `workspace/lowrank_toy/`: smaller pilot experiments.
- `workspace/lewm_autodl_results*`: retained metrics, plots, and analyses.
- `THIRD_PARTY.md`: upstream repositories and pinned commits.

## Deliberate exclusions

The original workspace was approximately 65 GB. This archive excludes a
46 GB HDF5 dataset, model checkpoints and weights, virtual environments,
generated caches, transfer ZIPs, AutoDL connection material, and unmodified
third-party checkouts. These exclusions do not remove the retained Markdown,
JSON, CSV, logs, plots, configurations, or source code used to interpret the
experiments.
