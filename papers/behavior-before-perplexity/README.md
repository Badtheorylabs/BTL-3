# BTL-3 compression research

This directory contains the public-facing technical record for the BTL-3
compact model.

## Documents

- `ARTICLE.md` is the long-form engineering article. It explains the project in
  chronological order, including the failed approaches and the practical
  recipe that produced the final artifact.
- `PAPER.tex` is the canonical research-paper source. It records the executed
  recipe, rejected stages, exact settings, artifact ledger, and claim limits.
- `CLAIMS.md` separates measured BTL results from inherited methods, external
  reports, hypotheses, and work that remains unmeasured.
- `references.bib` is the machine-readable bibliography.
- `figures/` contains figures generated only from public aggregate results and
  package manifests. No private evaluation cases are included.
- `output/pdf/` contains rendered PDFs.

## Contribution boundary

BTL does not claim to have invented vector quantization, affine integer
lattices, Hadamard incoherence processing, LDLQ, INT4 group quantization, LoRA,
or low-rank quantization-error correction. Those method families are prior art.

The BTL contribution is the measured system built from and around them:

1. behavior-first, artifact-faithful gates for agentic compression;
2. prefix replay and causal precision-island localization for a hybrid-attention
   decoder;
3. a failure-driven allocation of AVQ2, INT4, BF16 islands, vocabulary rescue,
   head correction, and behavior repair under an exact byte ceiling;
4. a complete native GGUF with checksummed manifests and packed CUDA/Metal execution; and
5. an empirical record showing why token agreement, perplexity, and local
   reconstruction error were not sufficient release criteria.

Together, these steps constitute a novel BTL compression cookbook: an ordered,
falsifiable systems method for preserving behavior under an exact physical
byte ceiling. The novelty claim applies to the cookbook, decision procedure,
and demonstrated artifact. A narrower claim that BTL invented a new universal
base quantizer would require broader public-model ablations and independent
comparison against faithful reference implementations.
