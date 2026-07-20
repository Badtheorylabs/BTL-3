# BTL-3 Compact GGUF exporter

## Scope

The exporter turns the frozen BTL-3 Compact package into the tensor contract
consumed by the custom Qwen3.6 llama.cpp backend. It never reconstructs a dense
decoder. Safetensors are mmap-backed and materialized one tensor at a time.
Every one of the 158 source files is size- and SHA-256-checked against the
frozen package manifest before planning or writing.

It exports:

- standard Qwen3.6 small-state tensors with official converter semantics;
- AVQ2 codes, affines, and exact Torch-seeded sign vectors;
- affine-INT4 codes, scales, and zero points;
- the two final INT4 demotions in place of their superseded BF16 islands;
- frozen BF16 islands;
- the rank-8 behavior LoRA;
- AVQ2 embedding and output tensors, rescued embedding rows, and head residual.

## Verified planning result

The full dry-run plans 2,416 tensors and 8,381,262,032 tensor bytes. The
170,510,920-byte reduction from the source package is expected: the package
retains two superseded BF16 island files for provenance, while the export emits
only their final INT4 demotions.

The exporter reports no unsupported representations and no native-runtime
contract gaps.

## Numerical conformance

Three representative layers were exported so every decoder representation was
materialized:

| Layer | Coverage | Tensors | Bytes | Payloads verified |
|---:|---|---:|---:|---:|
| 13 | AVQ2, BF16 island, final INT4 demotion | 28 | 322,295,104 | 28/28 |
| 40 | AVQ2, small state, behavior LoRA | 44 | 98,427,808 | 44/44 |
| 47 | AVQ2, standard affine INT4, behavior LoRA | 35 | 122,932,064 | 35/35 |

Every tensor passed:

- canonical-name equality;
- source-shape equality;
- exact GGUF storage-type equality;
- exact raw payload-byte equality after required Qwen transformations;
- exact regeneration of seeded signs.

The layer-40 artifact SHA-256 is
`e621b4cdb1e3f25cdde514427b14cd6d27bacbe72ad5850468311c693d4d79f7`.

This one-layer artifact is not a runnable model. A full export was deliberately
not started at this gate.

## Commands

```bash
.venv/bin/python tools/export_btl3_compact_gguf.py \
  --source /path/to/BTL-3/compact \
  --dry-run \
  --report artifacts/btl3-compact-gguf-dry-run.json

.venv/bin/python tools/export_btl3_compact_gguf.py \
  --source /path/to/BTL-3/compact \
  --conformance-layer 40 \
  --output artifacts/conformance/btl-3-compact-layer40.gguf \
  --report artifacts/conformance/btl-3-compact-layer40.report.json
```
