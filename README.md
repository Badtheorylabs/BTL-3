<div align="center">

# BTL-3

### Native runtime and integrations for Bad Theory Labs' agentic coding model

**27B · 262K architecture · 8.39 GB Compact edition · OpenAI-compatible**

[Full model](https://huggingface.co/badtheorylabs/BTL-3) ·
[Compact model](https://huggingface.co/badtheorylabs/BTL-3-Compact) ·
[Bad Theory Labs](https://www.badtheorylabs.com/) ·
[Discord](https://discord.gg/QJBCcB7bF)

</div>

## What is BTL-3?

BTL-3 is a 27B coding and tool-use model built for repository agents,
structured function calling, verification, and recovery. This repository
contains the native execution stack for **BTL-3 Compact**, plus its LM Studio,
Ollama CLI, and OpenAI-compatible integrations.

BTL-3 Compact stores the complete text model in one
**8,392,369,600-byte GGUF** and executes its packed representation directly.
It never reconstructs or requires the original BF16 checkpoint.

## Highlights

- Native packed CUDA and Apple Metal kernels.
- OpenAI-compatible streaming, reasoning, tool calls, and cancellation.
- Verified macOS arm64 runtime.
- Reproducible Linux/Windows CUDA and DGX Spark packaging definitions.
- LM Studio native-generator integration.
- Ollama CLI bridge and explicitly labeled patched-runner packaging.
- Deterministic installers, manifests, checksums, and conformance tests.

## Model results

| Evaluation | BTL-3 result |
|---|---:|
| BFCL v4 AST | **88.5% (1097/1240)** |
| HumanEval | **95.12% (156/164)** |
| LiveCodeBench v6 | **88.1% (170/193)** |
| BigCodeBench-Hard Instruct | **26.35% (39/148)** |

The Compact artifact retained **92.2% (83/90)** of teacher-correct behavior
overall on a fresh 100-turn tool-contract gate.

## Native performance

| Device | Prompt processing | Generation | Status |
|---|---:|---:|---|
| RTX PRO 6000 Blackwell 96 GB | **84.70 tok/s** | **43.16 tok/s** | Exact GGUF, full CUDA offload |
| Apple M2 16 GB | **2.30 tok/s** | **2.48 tok/s** | Exact GGUF, Metal smoke |

## Quickstart

Download the packaged model and runtime:

```bash
hf download badtheorylabs/BTL-3-Compact \
  --local-dir BTL-3-Compact
cd BTL-3-Compact
```

Install the verified macOS package:

```bash
python3 tools/install_consumer_bundle.py \
  --runtime runtimes/supported/BTL-3-Compact-macos-arm64 \
  --model model/BTL-3-Compact-AVQ2.gguf
```

Or start it directly:

```bash
BTL3_MODEL="$PWD/model/BTL-3-Compact-AVQ2.gguf" \
BTL3_CTX_SIZE=4096 \
  runtimes/supported/BTL-3-Compact-macos-arm64/bin/btl3-server
```

Call the OpenAI-compatible endpoint:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "BTL-3",
    "messages": [{"role": "user", "content": "Fix the failing tests."}],
    "stream": true
  }'
```

## Integration boundary

The integrations use BTL-3's packed llama.cpp implementation:

- **LM Studio** starts or connects to the installed native runner through the
  included generator.
- **Ollama CLI** connects through the included compatibility bridge.
- **OpenAI SDKs** connect directly to the local server.

Stock Ollama and the stock LM Studio GGUF engine do not decode AVQ2 directly.
Preview packages are kept separate from verified runtime bundles.

## Repository layout

| Path | Purpose |
|---|---|
| `native/llama.cpp` | Packed native runtime and kernels |
| `integrations/lmstudio` | LM Studio generator |
| `integrations/ollama` | Ollama CLI integration |
| `launch` | Cross-platform launchers |
| `packaging/cuda` | Reproducible NVIDIA bundle definitions |
| `tools` | Export, install, validation, and packaging tools |
| `tests` | Protocol and release conformance tests |
| `docs` | Runtime, CUDA, and integration guides |

## Development

Run the maintained integration tests:

```bash
python3 -m venv .venv
.venv/bin/pip install pytest
.venv/bin/pytest -q
```

Validate the LM Studio package:

```bash
cd integrations/lmstudio/btl3-native
npm ci
npm run typecheck
```

## Artifact identity

| Item | Value |
|---|---|
| Model | BTL-3 Compact RL-0013 |
| Bytes | `8,392,369,600` |
| SHA-256 | `2ddf9527620a17a2a6739d184a7096c45712092e6589128792ec6254e94dc30c` |
| Verified tensor payloads | `2,416` |

## License

BTL-3 model artifacts are Apache-2.0. The native runtime is MIT-licensed.
See [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Run generated code and tool calls in a sandbox, and require explicit
confirmation before destructive or high-impact actions.
