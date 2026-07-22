# BTL-3 compression claim ledger

The purpose of this ledger is to keep the article, paper, model card, and launch
copy synchronized.

## Measured artifact facts

| Claim | Evidence | Allowed wording |
|---|---|---|
| Final portable artifact | `compact/evidence/BTL-3-Compact-AVQ2.report.json` | 8,392,369,600 bytes, or 8.39 decimal GB |
| Tensor payload | final GGUF report | 8,381,262,032 bytes across 2,416 byte-verified tensors |
| Frozen provenance package | `btl-data-forge/artifacts/btl-3/compact-upload/package.json` | 8,572,070,080 bytes; retains superseded files not emitted in GGUF |
| Standalone loading | native CUDA and Metal smokes | Generates without loading the BF16 checkpoint |
| Decoder coverage | decoder manifest and runtime smoke | 64/64 decoder layers, no compatible dense decoder fallback |
| Tool-behavior retention | fresh sealed 100-turn gate | 83/90 teacher-correct turns retained, or 92.2% conditional retention |
| Absolute gate score | fresh sealed 100-turn gate | 83/100 absolute accuracy |
| Category result | fresh sealed gate | 100% conditional retention on single, parallel, sequential, and abstention; 30% on parallel-multiple |
| Malformed output | fresh sealed gate | 7%, entirely in parallel-multiple |
| RTX PRO 6000 native speed | three `llama-bench` runs | 84.70 +/- 0.37 prompt tok/s; 43.16 +/- 0.29 generation tok/s |
| Apple M2 native speed | compatibility smoke | 2.30 prompt tok/s; 2.48 generation tok/s |
| Final GGUF SHA-256 | final GGUF report | `2ddf9527620a17a2a6739d184a7096c45712092e6589128792ec6254e94dc30c` |

## Claims that must not be made

- Do not call 92.2% “intelligence retained.”
- Do not claim 92.2% coding retention; the compact artifact has not yet run the
  public coding suite.
- Do not claim every behavioral category passed 90%.
- Do not claim compatibility with stock Ollama or LM Studio engines; the
  included bridges use BTL's custom packed backend.
- Do not claim MLX, WebGPU, phone readiness, or unmeasured device throughput.
- Do not claim the private gate is a frontier benchmark.
- Do not claim BTL invented UniSVQ, vector quantization, LDLQ, Hadamard
  transforms, mixed precision, LoRA, or low-rank residual correction.
- Do not call the compression method “lossless.”

## BTL-specific contributions supported by artifacts

### Novel compression cookbook

BTL introduces a novel end-to-end cookbook for behavior-sensitive model
compression. Its novelty is the ordered decision procedure: artifact-faithful
behavior gates, prefix-cliff localization, measured precision allocation,
byte-neutral exchanges, vocabulary and head rescue, bounded behavior healing,
and standalone native proof under one exact physical byte ceiling. The claim
is cookbook and systems-methodology novelty, not invention of every underlying
quantization primitive.

### Behavior-first promotion

BTL used executable tool behavior, stopping, malformed-call rate, and
category-conditional retention as promotion gates. The project repeatedly
rejected candidates with respectable token or reconstruction metrics when
their emitted tool behavior was unusable.

### Hybrid-decoder failure localization

BTL replayed successively longer compressed prefixes and performed module
overrides at the first failing boundary. This localized interacting failures
that layer-local mean-squared error did not predict.

### Byte-constrained representation synthesis

The final package combines:

- AVQ2 vector codes for compatible decoder matrices;
- group-128 affine INT4 for full-attention projections;
- 12 physically retained BF16 islands;
- two measured BF16-to-INT4 demotions;
- one measured AVQ2-to-INT4 precision island;
- a 2.099-bpw embedding with 4,096 INT8-rescued rows;
- a 2.000-bpw language-model head;
- a rank-32 activation-weighted head residual;
- a 32.46 MB rank-8 behavior adapter; and
- compact non-matrix text state.

### Standalone proof

The native GGUF contains the text architecture contract and packed tensors;
the release includes the custom runner, tokenizer, and checksums. It does not
need the source BF16 checkpoint at load time.

## Prior-art-dependent components

| Component | Closest public method family |
|---|---|
| Four-weight vector codes and affine integer lattice | UniSVQ and vector quantization |
| Hadamard-style incoherence transform | QuIP#, QTIP, SpinQuant |
| Curvature-aware assignment and error propagation | GPTQ, SparseGPT, LDLQ-family methods |
| Group-wise affine INT4 | standard group quantization |
| Low-rank quantization-error correction | LQ-LoRA and related low-rank error-reconstruction work |
| Parameter-efficient behavior correction | LoRA / QLoRA family |
| Mixed precision and protected islands | mixed-precision quantization literature |

## Claims that need more evidence

The following are research hypotheses, not launch claims:

- behavior-first island selection generalizes to unrelated architectures;
- the BTL recipe beats faithful UniSVQ at equal average bit-width;
- the same allocation preserves public coding benchmarks at 90% or better;
- the 8.39 GB artifact is Pareto-optimal;
- the approach provides a production speedup before kernel optimization; and
- private conditional retention predicts long-horizon repository success.
