# BTL-3 consumer integrations

This document separates three products that must not be conflated:

| Surface | How BTL-3 executes | Current status |
|---|---|---|
| Stock Ollama | Stock runner | Does not execute AVQ2 |
| **BTL-3 Patched Ollama** | Ollama directly spawns BTL's native `llama-server` | Packaging scaffold complete; target packages pending |
| LM Studio | Official generator plugin calls BTL's native local endpoint | Plugin type-checks; macOS native endpoint verified |

No HTTP translation bridge sits between patched Ollama and the runner.

## Why the runner replacement is small

At official Ollama commit
[`573386c35eac76124ffce571f4b0fefa0a7fe13c`](https://github.com/ollama/ollama/commit/573386c35eac76124ffce571f4b0fefa0a7fe13c),
`llm.NewLlamaServer` delegates every GGUF model to a `llama-server`
subprocess. Ollama locates that executable under `lib/ollama`, starts it on an
ephemeral loopback port, and speaks its native `/health`, `/completion`,
`/v1/chat/completions`, tokenization, and embedding endpoints. The relevant
upstream files are
[`llm/server.go`](https://github.com/ollama/ollama/blob/573386c35eac76124ffce571f4b0fefa0a7fe13c/llm/server.go),
[`llm/llama_server.go`](https://github.com/ollama/ollama/blob/573386c35eac76124ffce571f4b0fefa0a7fe13c/llm/llama_server.go),
and
[`llm/llama_binary.go`](https://github.com/ollama/ollama/blob/573386c35eac76124ffce571f4b0fefa0a7fe13c/llm/llama_binary.go).

BTL's native runner already exposes that contract. The patched distribution
replaces Ollama's runtime payload, not its public API, scheduler, CLI, model
store, or UI.

## Build the CUDA runner payload

Build on the target machine so CMake selects the target GPU:

```bash
cmake -S native/llama.cpp -B native/llama.cpp/build-btl3-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_BACKEND_DL=ON \
  -DGGML_CUDA=ON \
  -DLLAMA_CURL=OFF
cmake --build native/llama.cpp/build-btl3-cuda \
  --target llama-server --parallel
```

Stage the executable and its shared libraries. Use `cuda_v13` with CUDA 13 and
`cuda_v12` with CUDA 12:

```bash
payload=/tmp/btl3-runner
cuda_dir=cuda_v13
rm -rf "$payload"
mkdir -p "$payload/$cuda_dir"
cp native/llama.cpp/build-btl3-cuda/bin/llama-server "$payload/"
cp native/llama.cpp/build-btl3-cuda/bin/libggml*.so* "$payload/"
cp native/llama.cpp/build-btl3-cuda/bin/libllama*.so* "$payload/"
mv "$payload"/libggml-cuda.so* "$payload/$cuda_dir/"
"$payload/llama-server" --help
```

Do not package until `--help` exposes `--model`, `--host`, `--port`,
`--no-webui`, `--offline`, and `-np`. The packager enforces that contract.

### Hardware targets

- RTX 4090: build `linux-amd64-cuda12` or `cuda13` on that machine.
- RTX 5090: build on the 5090 so `native` includes its Blackwell target.
- Windows RTX 4090/5090: build `windows-amd64-cuda13` on Windows x64.
- DGX Spark: build `linux-arm64-cuda13` on Spark. Current DGX Spark software is
  ARM64 and ships CUDA 13; do not reuse an x86_64 payload. See NVIDIA's
  [hardware overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html) and
  [current release notes](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html).

## Build BTL-3 Patched Ollama

Build Ollama from the pinned source commit using its official Linux build, or
unpack a distribution built from that exact commit into `OLLAMA_DIST`. Then:

```bash
.venv/bin/python tools/build_patched_ollama.py \
  --ollama-dist "$OLLAMA_DIST" \
  --runner-payload /tmp/btl3-runner \
  --platform linux-amd64-cuda13 \
  --output dist/btl3-patched-ollama-linux-amd64-cuda13
```

The output contains:

```text
bin/ollama
bin/btl3-ollama
lib/ollama/llama-server
lib/ollama/cuda_v13/libggml-cuda.so
share/btl3/patched-ollama-manifest.json
```

`btl3-ollama --btl3-info` prints the patched label. All other arguments are
passed to the packaged Ollama binary.

Create the local model only with the patched distribution:

```bash
cp artifacts/release/BTL-3-Compact-AVQ2.gguf integrations/ollama/
dist/btl3-patched-ollama-linux-amd64-cuda13/bin/btl3-ollama serve
# in another terminal
dist/btl3-patched-ollama-linux-amd64-cuda13/bin/btl3-ollama \
  create btl3-compact -f integrations/ollama/Modelfile
dist/btl3-patched-ollama-linux-amd64-cuda13/bin/btl3-ollama \
  run btl3-compact
```

The GGUF stores BTL's packed representation in ordinary I8/F32/BF16 tensors
with custom names, so no new GGUF tensor-type enum is required. Actual model
creation and generation remain release gates on each CUDA target.

### Windows x64 CUDA 13

The pinned Ollama source has the same direct subprocess architecture on
Windows. Its
[`llama_binary.go`](https://github.com/ollama/ollama/blob/573386c35eac76124ffce571f4b0fefa0a7fe13c/llm/llama_binary.go)
adds `.exe` and searches beside a top-level `ollama.exe` at
`lib\ollama\llama-server.exe`; the pinned
[`build_windows.ps1`](https://github.com/ollama/ollama/blob/573386c35eac76124ffce571f4b0fefa0a7fe13c/scripts/build_windows.ps1)
installs CUDA payloads under that same directory.

Build the native runner from an x64 Visual Studio developer shell with CUDA 13:

```powershell
cmake -S native/llama.cpp -B native/llama.cpp/build-btl3-win-cuda13 `
  -A x64 `
  -DBUILD_SHARED_LIBS=ON `
  -DGGML_BACKEND_DL=ON `
  -DGGML_CUDA=ON `
  -DLLAMA_CURL=OFF
cmake --build native/llama.cpp/build-btl3-win-cuda13 `
  --config Release --target llama-server --parallel

$Bin = "native/llama.cpp/build-btl3-win-cuda13/bin/Release"
$Payload = "$env:TEMP/btl3-runner-win"
Remove-Item -Recurse -Force $Payload -ErrorAction SilentlyContinue
New-Item -ItemType Directory "$Payload/cuda_v13" | Out-Null
Copy-Item "$Bin/llama-server.exe" $Payload
Copy-Item "$Bin/*.dll" $Payload
Move-Item "$Payload/ggml-cuda.dll" "$Payload/cuda_v13/"
powershell -ExecutionPolicy Bypass -File tools/probe_windows_runner.ps1 `
  -Runner "$Payload/llama-server.exe" `
  -Output "$Payload/runner-cli-contract.json"
```

The target-side probe executes the exact runner and binds its required Ollama
CLI flags to its SHA-256. Cross-platform packaging refuses a missing, stale,
or wrong-platform contract. It also checks that `ollama.exe`,
`llama-server.exe`, and every staged DLL are PE x64.

Package an official pinned Windows x64 distribution:

```powershell
python tools/build_patched_ollama.py `
  --ollama-dist "$env:TEMP/ollama-windows-amd64" `
  --runner-payload "$env:TEMP/btl3-runner-win" `
  --platform windows-amd64-cuda13 `
  --output dist/btl3-patched-ollama-windows-amd64-cuda13

dist\btl3-patched-ollama-windows-amd64-cuda13\btl3-ollama.cmd --btl3-info
dist\btl3-patched-ollama-windows-amd64-cuda13\btl3-ollama.cmd serve
```

The Windows output uses top-level `ollama.exe` and `btl3-ollama.cmd`, with the
runner at `lib\ollama\llama-server.exe` and CUDA backend at
`lib\ollama\cuda_v13\ggml-cuda.dll`. This path is source-verified and
package-tested, but still requires generation conformance on real Windows
CUDA hardware before release.

## Required Ollama release gates

For each 4090, 5090, and DGX Spark artifact:

1. Verify runner and CUDA library architecture with `file`/`ldd` on Linux, or
   `dumpbin /headers` and `/dependents` on Windows.
2. Run `btl3-ollama --btl3-info` and verify the manifest hashes.
3. Create the model in a temporary `OLLAMA_MODELS` directory.
4. Confirm Ollama itself spawns the packaged runner PID.
5. Exercise streaming chat, cancellation, reasoning, single and parallel tool
   calls, JSON schema output, context reuse, and unload/reload.
6. Reverify the exact model SHA-256:
   `2ddf9527620a17a2a6739d184a7096c45712092e6589128792ec6254e94dc30c`.

Until those checks pass on a real CUDA executable, the output is a packaging
candidate—not an Ollama-compatible release.

## LM Studio generator

LM Studio's official plugin API treats generators as replacement token
sources. The BTL plugin therefore calls the native runner's OpenAI-compatible
endpoint; it does not ask LM Studio's stock engine to interpret AVQ2. Official
generator and tool-call APIs are documented in LM Studio's
[generator introduction](https://lmstudio.ai/docs/typescript/plugins/generator)
and
[tool-calling guide](https://lmstudio.ai/docs/typescript/plugins/generator/tool-calling-generators).

Install the BTL native runner and model into the platform-default BTL-3 data
directory (or configure their paths in the plugin), then:

```bash
cd integrations/lmstudio/btl3-native
./run-local
```

The script installs the locked dependencies and runs the official `lms dev`
workflow. The model appears in LM Studio's model picker as the generator
plugin. Its default endpoint is `http://127.0.0.1:8080/v1`. When that endpoint
is offline, the Node plugin starts the installed native runner directly with
the configured model; no separately launched compatibility server is required.

The plugin preserves:

- streamed content;
- reasoning fragments through `reasoningType: "reasoning"`;
- cancellation through `ctl.abortSignal`;
- full chat history and tool results;
- parallel tool-call IDs, names, and JSON argument fragments;
- explicit propagation of cancellation, connection, and API failures.

Content and reasoning stream live. LM Studio's generator event lifecycle
represents one active tool call at a time and its argument-fragment callback
does not include a call ID. To preserve BTL-3's parallel calls correctly, the
plugin buffers tool fragments by upstream call index and emits each complete
lifecycle sequentially when the server stream ends. It does not misattribute
interleaved arguments merely to make the tool UI look live.

The included `model.yaml` is catalog metadata, not a claim that the stock LM
Studio GGUF engine supports AVQ2. It uses the explicit compatibility label
`btl3-avq2-native`.
