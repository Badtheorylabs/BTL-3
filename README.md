<div align="center">

# BTL-3

### Agentic coding, structured tool use, and native local inference

**27B · 262K architecture · 95.1% HumanEval · 88.5% BFCL v4 AST**

[BTL-3 weights](https://huggingface.co/badtheorylabs/BTL-3) ·
[BTL-3 Compact](https://huggingface.co/badtheorylabs/BTL-3-Compact) ·
[Bad Theory Labs](https://www.badtheorylabs.com/) ·
[Discord](https://discord.gg/QJBCcB7bF)

</div>

## The BTL-3 family

BTL-3 is Bad Theory Labs' 27B coding and tool-use model. It is built for
repository agents that must reason, act, inspect tool results, recover from
failures, and stop when no action is required.

The release has two distinct editions:

| | **BTL-3** | **BTL-3 Compact** |
|---|---|---|
| Purpose | Maximum-quality model release | Portable native local inference |
| Distribution | Rank-32 PEFT adapter | Complete standalone GGUF |
| Base requirement | Qwen3.6-27B at the pinned revision | None |
| Model payload | 933.97 MB adapter plus base model | 8.39 GB complete text model |
| Runtime | Transformers or vLLM | BTL packed llama.cpp |
| Precision | Base checkpoint precision | Mixed AVQ2, affine INT4, precision islands, compact corrections |
| Context architecture | 262,144 tokens | 262,144 tokens |
| Quality reference | Official coding and BFCL results | 92.2% overall conditional tool-contract retention |
| Best for | Servers, evaluation, maximum model quality | Macs, workstations, private/offline agents |
| Hugging Face | [`badtheorylabs/BTL-3`](https://huggingface.co/badtheorylabs/BTL-3) | [`badtheorylabs/BTL-3-Compact`](https://huggingface.co/badtheorylabs/BTL-3-Compact) |

**Choose BTL-3** when you can host Qwen3.6-27B and want the frozen RL-0013
checkpoint at full quality.

**Choose BTL-3 Compact** when you want one native 8.39 GB artifact without
loading or reconstructing the BF16 base model.

## Why BTL-3

- Agentic coding and structured execution in the supported non-thinking mode.
- Structured single, multiple, and parallel tool calls.
- Explicit abstention behavior: the model is trained to avoid acting when a
  tool call is unnecessary.
- Multi-turn recovery after failed actions and verifier feedback.
- Open weights and private, self-hosted deployment.
- A separate compact runtime that executes its packed weights directly rather
  than inflating them into a dense checkpoint.

## BTL-3 results

These results belong to the frozen **BTL-3 RL-0013** release.

| Evaluation | Score | Protocol |
|---|---:|---|
| BFCL v4 AST | **88.5% (1097/1240)** | Complete official full set |
| HumanEval | **95.12% (156/164)** | pass@1, thinking mode |
| LiveCodeBench v6 | **88.1% (170/193)** | Completed 193-case run, thinking mode |
| BigCodeBench-Hard Instruct | **26.35% (39/148)** | Official strict pass@1 |
| BigCodeBench functional tests | **59.25% (506/854)** | Supplementary test-level score |

### BFCL v4 category breakdown

| Category | Score |
|---|---:|
| Simple | **93.2%** |
| Multiple | **95.5%** |
| Parallel | **87.0%** |
| Parallel-multiple | **70.0%** |
| Irrelevance | **91.2%** |

The 91.2% irrelevance score measures a core agent property: knowing when a
request should not produce a tool call.

## BTL-3 Compact validation

Compact is evaluated separately because it is a different physical
representation of the BTL-3 checkpoint.

Its fresh sealed tool-contract gate contains 100 turns balanced across single,
parallel, sequential, parallel-multiple, and abstention behavior. The gate was
written after compression and repair choices were frozen.

| Metric | Result |
|---|---:|
| Full-precision BTL-3 teacher | **90/100** |
| BTL-3 Compact | **83/100** |
| Overall conditional retention | **92.2% (83/90)** |
| Single-call retention | **100%** |
| Parallel-call retention | **100%** |
| Sequential-call retention | **100%** |
| Abstention retention | **100%** |

This is an internal behavior-retention gate, not a public frontier benchmark.
The full protocol and category report ship with Compact at
`evidence/compact-validation.md`.

## Model specifications

### BTL-3

| Item | Value |
|---|---|
| Architecture | Qwen3.6-27B hybrid-attention causal language model |
| Base model | `Qwen/Qwen3.6-27B` |
| Base revision | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| Checkpoint | RL-0013 |
| Adapter | PEFT LoRA, rank 32, alpha 64 |
| Adapter bytes | 933,974,032 |
| Architectural context | 262,144 tokens |
| Maximum RL sequence length | 65,536 tokens |
| Launch benchmark context | 32,768 tokens |
| License | Apache-2.0 |

### BTL-3 Compact

| Item | Value |
|---|---|
| Lineage | Qwen3.6-27B → BTL-3 RL-0013 |
| Scope | Text-only coding, reasoning, and tool use |
| Layers | 64 |
| Model file | `BTL-3-Compact-AVQ2.gguf` |
| Model bytes | 8,392,369,600 |
| Model size | 8.39 GB decimal / 7.82 GiB |
| Packed tensor payloads | 2,416 |
| Runtime | BTL packed llama.cpp |
| Dense reconstruction | Never required |
| Model license | Apache-2.0 |
| Runtime license | MIT |

## Quickstart: BTL-3

### Transformers

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen3.6-27B"
base_revision = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
adapter_id = "badtheorylabs/BTL-3"

tokenizer = AutoTokenizer.from_pretrained(adapter_id)
base = AutoModelForCausalLM.from_pretrained(
    base_id,
    revision=base_revision,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, adapter_id)

messages = [
    {
        "role": "user",
        "content": "Inspect this repository, fix the failing tests, and explain the patch.",
    }
]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=4096)
completion = output[0, inputs.input_ids.shape[1]:]
print(tokenizer.decode(completion, skip_special_tokens=False))
```

### vLLM

```bash
vllm serve Qwen/Qwen3.6-27B \
  --revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 \
  --served-model-name BTL-3 \
  --enable-lora \
  --max-lora-rank 32 \
  --lora-modules BTL-3=/path/to/BTL-3 \
  --lora-target-modules \
    q_proj k_proj v_proj o_proj \
    in_proj_qkv in_proj_z in_proj_b in_proj_a out_proj \
    gate_proj up_proj down_proj \
  --reasoning-parser qwen3 \
  --language-model-only \
  --max-model-len 32768
```

For structured tools, enable the Qwen XML tool parser supported by the
installed vLLM release.

## Quickstart: BTL-3 Compact

Download the standalone package:

```bash
hf download badtheorylabs/BTL-3-Compact \
  --local-dir BTL-3-Compact
cd BTL-3-Compact
```

Install the verified host runtime and model from the downloaded release:

```bash
python3 tools/install_consumer_bundle.py --package .
```

The installer discovers the matching supported runtime, verifies every runtime
file and the complete model hash, and installs atomically. If the package only
contains a preview runtime for the host, installation stops instead of silently
claiming support. `--allow-preview` is an explicit opt-in for testing only.

Or start the packaged server directly:

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
    "messages": [
      {
        "role": "user",
        "content": "Write a retrying fetch helper and include tests."
      }
    ],
    "chat_template_kwargs": {"enable_thinking": false},
    "stream": true
  }'
```

> **Compact thinking-mode status:** thinking is disabled by default and is not
> recommended in this release. Its experimental opt-in can repeat procedural
> reasoning or fail to terminate. Use the supported non-thinking path for
> chat, coding, and tool use while the reasoning-policy repair is prepared.

## Compact representation

BTL-3 Compact is not a uniform scalar quantization. It combines:

- packed AVQ2 decoder tensors;
- affine INT4 tensors;
- two measured INT4 demotions;
- selected higher-precision behavior-sensitive islands;
- packed embedding and output matrices;
- a rank-32 output correction;
- a compact behavior adapter.

All 2,416 tensor payloads were byte-verified during export. The exact GGUF
then completed native autoregressive generation on Apple Metal and NVIDIA
CUDA without a persistent dense reconstruction.

## Compression research

The complete compression record is published with the repository:

- [Behavior Before Perplexity](papers/behavior-before-perplexity/output/pdf/btl-3-behavior-before-perplexity.pdf), the five-page research paper;
- [engineering article](papers/behavior-before-perplexity/output/pdf/btl-3-compression-engineering-article.pdf), the chronological build account;
- [canonical LaTeX source](papers/behavior-before-perplexity/PAPER.tex);
- [claim ledger](papers/behavior-before-perplexity/CLAIMS.md); and
- [checksums and research manifest](papers/behavior-before-perplexity/RESEARCH_MANIFEST.json).

The paper documents the executed recipe rather than only naming its method
families: calibration splits, FP64 second moments, 128-wide Hadamard blocks,
four-weight affine codes, block-LDLQ assignment, scale search, failed
reconstruction stages, packed-prefix bisection, measured precision islands,
behavior repair, vocabulary rescue, output-head correction, native export,
and the sealed artifact gate.

BTL claims novelty for the behavior-first cookbook and decision procedure,
not for the public quantization, LDLQ, Hadamard, or LoRA primitives it uses.

## Runtime and integration support

| Surface | Status | Notes |
|---|---|---|
| macOS arm64 / Metal | **Verified** | Packaged under `runtimes/supported` |
| NVIDIA CUDA kernels | **Verified on RTX PRO 6000** | Exact GGUF, full CUDA offload |
| Linux arm64 / DGX Spark package | Preview | Cross-compiled; target-device conformance pending |
| OpenAI-compatible server | **Verified** | Streaming, reasoning, tools, and cancellation |
| LM Studio | **Development preview** | Unpublished generator; Windows end-to-end validation pending |
| Ollama CLI | **Supported through bridge** | Preserves the familiar client surface |
| Stock Ollama model engine | Not direct | Stock `ollama create` does not decode AVQ2 |
| Stock LM Studio GGUF engine | Not direct | Use the included native generator |

Preview packages are never labeled as verified runtime bundles.

The LM Studio generator is not a stock GGUF import and is not yet a published
consumer plugin. Treat it as a development preview until the downloadable
Windows package passes installation, cancellation, and generation tests on
target NVIDIA hardware.

## Measured native performance

| Device | Prompt processing | Generation | Measurement |
|---|---:|---:|---|
| RTX PRO 6000 Blackwell 96 GB | **84.70 tok/s** | **43.16 tok/s** | 512-token prompt, 128 generated tokens, three runs |
| Apple M2 16 GB | **2.30 tok/s** | **2.48 tok/s** | Exact-model compatibility smoke |

The Apple M2 number is intentionally a compatibility result, not a projection
for M4/M5 hardware. RTX 4090, RTX 5090, DGX Spark, and Windows packages require
their own target-device measurements.

## Context and memory

Both editions inherit a 262,144-token architectural context. Actual usable
context depends on KV cache, runtime workspace, and device memory.

Conservative Compact starting points:

| Device budget | Starting context |
|---|---:|
| 16 GB Apple Silicon | 4K |
| 12–16 GB GPU | 16K |
| 24 GB GPU | 32K |

Raise context only after measuring peak memory on the target runtime.

## Repository layout

| Path | Purpose |
|---|---|
| `native/llama.cpp` | Native packed runtime and kernels |
| `integrations/lmstudio` | LM Studio native generator |
| `integrations/ollama` | Ollama CLI bridge |
| `launch` | Cross-platform server launchers |
| `packaging/cuda` | Reproducible NVIDIA bundle definitions |
| `tools` | Export, install, validation, and packaging tools |
| `tests` | Protocol and release conformance tests |
| `docs` | Runtime, CUDA, and integration guides |
| `artifacts` | Public conformance and release reports |

Model weights are distributed through Hugging Face, not GitHub.

## Build and test

Run the maintained Python integration suite:

```bash
python3 -m venv .venv
.venv/bin/pip install pytest
.venv/bin/pytest -q
```

Validate the LM Studio integration:

```bash
cd integrations/lmstudio/btl3-native
npm ci
npm run typecheck
```

Build native llama.cpp using the normal CMake flow for the target backend.
CUDA package definitions and launch instructions live in:

- [`docs/launch-btl3-cuda.md`](docs/launch-btl3-cuda.md)
- [`packaging/cuda`](packaging/cuda)

Consumer integration details live in:

- [`docs/launch-btl3-compact.md`](docs/launch-btl3-compact.md)
- [`docs/patched-ollama-and-lmstudio.md`](docs/patched-ollama-and-lmstudio.md)

## Reproducibility and integrity

### BTL-3 adapter

| Artifact | SHA-256 |
|---|---|
| `adapter_model.safetensors` | `37a8f519039707eba5906591cdb14268768db43f80489a9c2f83b3e51e5e89db` |

### BTL-3 Compact

| Artifact | Value |
|---|---|
| File | `BTL-3-Compact-AVQ2.gguf` |
| Bytes | `8,392,369,600` |
| SHA-256 | `2ddf9527620a17a2a6739d184a7096c45712092e6589128792ec6254e94dc30c` |
| Verified payloads | `2,416` |

The Compact release includes:

- `RELEASE_MANIFEST.json` for exact model and runtime identity;
- `SHA256SUMS` for package verification;
- native-load and exporter reports;
- measured runtime evidence;
- a separation between supported and preview backends.

## Intended use

- coding, debugging, and test-driven repair;
- repository and terminal agents;
- structured single, sequential, and parallel tool calls;
- multi-turn tasks with execution feedback and recovery;
- private, offline, or self-hosted inference.

Use non-thinking mode for production. Compact's experimental thinking override
is currently discouraged because it may repeat or fail to terminate.

## Operational guidance

Run generated code and tool calls in a sandbox. Require explicit confirmation
before destructive, privileged, financial, or otherwise high-impact actions.

## License and citation

BTL-3 model artifacts are Apache-2.0. The native runtime is MIT-licensed.
See [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

```bibtex
@software{btl3_2026,
  title  = {BTL-3: Agentic Coding, Structured Tool Use, and Native Local Inference},
  author = {Bad Theory Labs},
  year   = {2026},
  url    = {https://github.com/Badtheorylabs/BTL-3}
}
```

```bibtex
@techreport{btl3_compact_2026,
  title       = {Behavior Before Perplexity: A Failure-Driven Recipe for
                 Compressing a 27B Agentic Model to 8.39 GB},
  author      = {{Bad Theory Labs}},
  institution = {Bad Theory Labs},
  year        = {2026},
  month       = {July},
  url         = {https://github.com/Badtheorylabs/BTL-3/tree/main/papers/behavior-before-perplexity}
}
```

For questions and release updates, visit
[Bad Theory Labs](https://www.badtheorylabs.com/) or join the
[community Discord](https://discord.gg/QJBCcB7bF).
