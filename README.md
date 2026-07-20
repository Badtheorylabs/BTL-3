# BTL-3 Compact runtime

Native execution and product integrations for the exact 8.39 GB BTL-3 Compact
artifact.

- OpenAI-compatible streaming server
- Apple Metal and NVIDIA CUDA packed kernels
- LM Studio native-generator integration
- Ollama CLI bridge and explicitly labeled patched-Ollama packaging
- deterministic installers, manifests, checksums, and conformance tests

## Models

- [BTL-3](https://huggingface.co/badtheorylabs/BTL-3): full-quality RL-0013
  adapter for Qwen3.6-27B.
- [BTL-3 Compact](https://huggingface.co/badtheorylabs/BTL-3-Compact): exact
  8.39 GB native GGUF and packaged integrations.

## Backend boundary

Integrations use the native packed llama.cpp implementation. Patched Ollama
spawns that runner directly. The LM Studio generator starts the same installed
runner automatically and then speaks its loopback OpenAI-compatible API.
The Python/CUDA reference is a research oracle, not a release backend.

## Verified status

The exact release GGUF is `8,392,369,600` bytes with SHA-256
`2ddf9527620a17a2a6739d184a7096c45712092e6589128792ec6254e94dc30c`.
All 2,416 tensor payloads were byte-verified during export.

The packaged macOS arm64 server passes native model load and generation.
OpenAI and Ollama transport behavior—including streaming, reasoning, tool
calls, and cancellation—is covered by protocol tests.

The Apple Metal backend accelerates every custom decode-critical path: AVQ2,
affine INT4, vocabulary projection, and rescued and ordinary embedding rows.
A full-model Apple M2 smoke measured 2.30 prompt tokens/second and 2.48
generated tokens/second. See
[the local launch guide](docs/launch-btl3-compact.md) for commands, supported
surfaces and device context guidance.

The exact GGUF also passes full native CUDA execution on an RTX PRO 6000
Blackwell Server Edition. Three native repetitions measured 84.70 prompt
tokens/second and 43.16 generated tokens/second. That validates the Linux
x86_64 Blackwell implementation, not every consumer package or GPU.

Reproducible CUDA 13.0.2 definitions target RTX 4090 (`sm_89`), RTX 5090
(`sm_120a`), Windows x64, and DGX Spark (`arm64`/`sm_121a`). Those exact
device/distribution combinations remain preview-only until their own
full-model conformance gates pass. See
[the CUDA launch guide](docs/launch-btl3-cuda.md).

The [consumer integration guide](docs/patched-ollama-and-lmstudio.md) documents
BTL-3 Patched Ollama and the official LM Studio generator. Stock Ollama and
the stock LM Studio GGUF engine do not execute AVQ2.

## Build the clean launch directory

The builder verifies the complete model and every runtime hash, excludes
development dependencies, separates supported and preview runtimes, and emits
`RELEASE_MANIFEST.json` plus `SHA256SUMS`:

```bash
.venv/bin/python tools/build_launch_release.py --help
```

## Compact GGUF exporter

The custom exporter preserves BTL-3 Compact's AVQ2, affine-INT4, vocabulary
rescue, and behavior-LoRA tensors without reconstructing dense weights:

```bash
.venv/bin/python tools/export_btl3_compact_gguf.py \
  --source /path/to/BTL-3/compact \
  --dry-run \
  --report artifacts/btl3-compact-gguf-dry-run.json
```

Use `--conformance-layer N --output PATH` to write and byte-verify one layer
before a full export. A conformance-layer file is intentionally incomplete and
cannot be served as a standalone model.
