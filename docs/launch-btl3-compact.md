# Launch BTL-3 Compact locally

This repository ships a relocatable macOS arm64 server bundle for the exact
BTL-3 Compact AVQ2 artifact:

- model: `BTL-3-Compact-AVQ2.gguf`
- bytes: `8,392,369,600`
- SHA-256:
  `2ddf9527620a17a2a6739d184a7096c45712092e6589128792ec6254e94dc30c`
- runtime: llama.cpp build 9596, commit `9fcaed763`

The model is external to the 28 MB runtime bundle. The launcher finds the model
in `artifacts/release`, in the bundle's `model` directory, or at `BTL3_MODEL`.

## Build and verify the bundle

Run this on an Apple Silicon Mac with Homebrew OpenSSL 3 installed:

```bash
rm -rf artifacts/runtime/BTL-3-Compact-macos-arm64
.venv/bin/python tools/build_macos_arm64_bundle.py
```

The builder copies the native dependency closure, rewrites absolute install
names and rpaths, ad-hoc signs the result, includes dependency licenses, and
writes `bundle-manifest.json`.

## OpenAI-compatible API

Start the server:

```bash
BTL3_CTX_SIZE=4096 \
  artifacts/runtime/BTL-3-Compact-macos-arm64/bin/btl3-server
```

Check it:

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/models
```

Call chat completions with any OpenAI-compatible client:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "BTL-3",
    "messages": [{"role": "user", "content": "Write a retrying fetch helper."}],
    "stream": true
  }'
```

Useful configuration:

| Variable | Default | Meaning |
|---|---:|---|
| `BTL3_HOST` | `127.0.0.1` | Listen address |
| `BTL3_PORT` | `8080` | OpenAI API port |
| `BTL3_CTX_SIZE` | `32768` | Allocated context |
| `BTL3_PARALLEL` | `1` | Concurrent slots |
| `BTL3_GPU_LAYERS` | `99` | Layers requested on Metal |
| `BTL3_API_KEY` | unset | Optional local bearer key |
| `BTL3_MODEL_ALIASES` | `BTL-3` | Comma-separated transport aliases |
| `BTL3_MAX_TOKENS` | `2048` | Default per-response generation cap |
| `BTL3_REPEAT_PENALTY` | `1.10` | Default repetition penalty |
| `BTL3_REPEAT_LAST_N` | `512` | Token window used by repetition penalty |
| `BTL3_ENABLE_THINKING` | `false` | Experimental opt-in; not recommended because thinking may repeat or fail to terminate |

On a 16 GB Mac, begin at 2K–4K context. A larger context raises KV-cache and
working-memory requirements. The model's declared context length is not a
promise that a particular device can allocate it.

### Thinking-mode status

Compact is released with thinking disabled by default. The non-thinking path
is the supported path for chat, coding, and tools in this revision. Thinking
remains available only as an experimental override:

```bash
BTL3_ENABLE_THINKING=true \
  artifacts/runtime/BTL-3-Compact-macos-arm64/bin/btl3-server
```

It is currently discouraged because it can enter repetitive procedural
reasoning or fail to reach a final answer. A later reasoning-policy repair will
receive a new artifact identity and validation report.

## LM Studio

LM Studio's official
[`openai-compat-endpoint`](https://www.lmstudio.ai/lmstudio/openai-compat-endpoint)
generator can target the local API. Its current model picker uses a fixed set
of model IDs, so expose one of those IDs as a **transport alias**:

```bash
BTL3_API_KEY=btl3-local \
BTL3_MODEL_ALIASES='BTL-3,gpt-4.1-2025-04-14' \
BTL3_CTX_SIZE=4096 \
  artifacts/runtime/BTL-3-Compact-macos-arm64/bin/btl3-server
```

Then:

1. Install the official plugin from its LM Studio Hub page.
2. Set **Override Base URL** to `http://127.0.0.1:8080/v1`.
3. Set its API key to `btl3-local`.
4. Select `gpt-4.1-2025-04-14`.

That name is only protocol routing. The served model remains BTL-3; the
integration does not claim it is an OpenAI model.

## Ollama CLI

Stock Ollama does not load BTL-3's custom AVQ2 GGUF. Do not use
`ollama create`. The supplied bridge lets the unmodified Ollama CLI speak to
the native BTL-3 server through Ollama's local HTTP protocol.

In terminal one:

```bash
BTL3_CTX_SIZE=4096 \
  artifacts/runtime/BTL-3-Compact-macos-arm64/bin/btl3-server
```

In terminal two:

```bash
BTL3_CTX_SIZE=4096 \
  artifacts/runtime/BTL-3-Compact-macos-arm64/bin/btl3-ollama-bridge
```

Then use the installed Ollama CLI:

```bash
OLLAMA_HOST=http://127.0.0.1:11435 ollama list
OLLAMA_HOST=http://127.0.0.1:11435 ollama show btl3-compact:latest
OLLAMA_HOST=http://127.0.0.1:11435 ollama run btl3-compact:latest
```

The bridge implements `/api/chat`, `/api/generate`, `/api/tags`, `/api/show`,
`/api/ps`, and `/api/version`. It maps streaming to Ollama NDJSON and preserves
reasoning, tool calls, structured output, cancellation, token counts, and the
common sampling options. Unsupported Ollama management endpoints return 501
instead of pretending to work.

## Validation status

The packaged executable has passed native model load, graph reservation,
`/health`, and `/v1/models` checks on Apple M2. Protocol tests pass using the
real installed Ollama CLI and a deterministic OpenAI-compatible upstream.

The native Metal runtime accelerates AVQ2, affine INT4, the vocabulary head,
and rescued and ordinary embedding rows without reconstructing dense weights.
On a base Apple M2 with 16 GB unified memory, a clean full-model smoke measured
2.30 prompt tokens/second and 2.48 generated tokens/second at a 128-token
allocated context. Treat those as one-device smoke results, not universal
throughput claims. Context size, prompt length, thermal state, and Apple chip
generation will change performance.

The exact GGUF has passed full-model native CUDA execution on an RTX PRO 6000
Blackwell Server Edition. Three repetitions measured 84.70 prompt
tokens/second and 43.16 generated tokens/second with full GPU offload. This
does not establish RTX 4090, RTX 5090, DGX Spark, or Windows compatibility;
each target still requires its own packaged-runtime gate. Do not reuse the old
Python/Triton H100 number as a native-runtime claim.

The native server may display roughly 7.7B `n_params`; that value counts packed
stored elements, not the logical 27B architecture. Product metadata exposed by
the bridge reports the logical model class.
