---
license: apache-2.0
base_model: Qwen/Qwen3.6-27B
pipeline_tag: text-generation
library_name: llama.cpp
tags:
  - agent
  - code
  - reasoning
  - tool-use
  - gguf
  - quantized
---

# BTL-3 Compact

BTL-3 Compact is the portable native edition of BTL-3, Bad Theory Labs'
27B coding and tool-use model. The release stores the complete text model in
one 8.39 GB GGUF and executes it with BTL's packed AVQ2/UniSVQ llama.cpp
runtime. It does not reconstruct or require the original BF16 checkpoint.

The full-quality adapter is
[BTL-3](https://huggingface.co/badtheorylabs/BTL-3). Runtime source,
reproducible packaging, and integrations are maintained at
[Badtheorylabs/BTL-3](https://github.com/Badtheorylabs/BTL-3).

## Exact model

- File: `model/BTL-3-Compact-AVQ2.gguf`
- Bytes: `8,392,369,600` (7.82 GiB)
- SHA-256:
  `2ddf9527620a17a2a6739d184a7096c45712092e6589128792ec6254e94dc30c`
- Logical architecture: Qwen3.6-27B
- Frozen checkpoint: BTL-3 RL-0013
- Declared architecture context: 262,144 tokens

Device memory, KV cache, and runtime workspace determine the usable context on
each machine. Start with 4K on a 16 GB Mac, 16K on a 12–16 GB GPU, and 32K on
a 24 GB GPU, then raise it only after measuring headroom.

## Start the verified macOS runtime

Optionally install the runtime and model into the per-user BTL-3 directory:

```bash
python3 tools/install_consumer_bundle.py \
  --runtime runtimes/supported/BTL-3-Compact-macos-arm64 \
  --model model/BTL-3-Compact-AVQ2.gguf
```

Or run directly from the release directory:

```bash
BTL3_MODEL="$PWD/model/BTL-3-Compact-AVQ2.gguf" \
BTL3_CTX_SIZE=4096 \
  runtimes/supported/BTL-3-Compact-macos-arm64/bin/btl3-server
```

The API is OpenAI-compatible:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "BTL-3",
    "messages": [{"role": "user", "content": "Write a retrying fetch helper."}],
    "stream": true
  }'
```

## Integrations

- **LM Studio:** install the included `btl3-native` generator. It starts or
  connects to the native runner; LM Studio's stock GGUF engine does not decode
  AVQ2.
- **Ollama CLI on macOS:** run the included `btl3-ollama-bridge` beside the
  native server and point `OLLAMA_HOST` at it.
- **Patched Ollama:** packaging source is included, but stock `ollama create`
  cannot execute this GGUF and patched CUDA distributions remain release
  candidates until their target-device gates pass.

Compatibility is never faked: a preview runtime lives under
`runtimes/preview`, not `runtimes/supported`.

## Measured native performance

| Device | Prompt processing | Generation | Status |
|---|---:|---:|---|
| RTX PRO 6000 Blackwell 96 GB | 84.70 tok/s | 43.16 tok/s | Exact GGUF, full CUDA offload |
| Apple M2 16 GB | 2.30 tok/s | 2.48 tok/s | Exact GGUF, Metal smoke |

The RTX measurement used a 512-token prompt and 128 generated tokens over
three repetitions. The M2 result used a 128-token allocated context and is a
compatibility smoke, not a claim about newer Apple hardware.

RTX 4090, RTX 5090, DGX Spark, Windows, Ollama-patched CUDA packages, and phone
execution are not represented by these measurements.

## Verify before running

From this directory:

```bash
shasum -a 256 -c SHA256SUMS
```

`RELEASE_MANIFEST.json` separates verified runtimes from preview packages and
records the exact model identity. Generated code and tool calls should run in
a sandbox, and destructive or high-impact operations should require explicit
confirmation.
