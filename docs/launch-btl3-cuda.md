# BTL-3 consumer NVIDIA build and packaging

This repository contains reproducible native CUDA build and package definitions
for three consumer targets:

| Target | Intended hardware | Requested CUDA architectures |
|---|---|---|
| `linux-x86_64` | RTX 4090 and RTX 5090 | `89-real;120-real` |
| `windows-x86_64` | RTX 4090 and RTX 5090 | `89-real;120-real` |
| `linux-arm64` | NVIDIA DGX Spark | `121-real` |

The definitions pin CUDA Toolkit 13.0.2. NVIDIA lists compute capability 8.9
for RTX 4090, 12.0 for RTX 5090, and 12.1 for DGX Spark. NVIDIA's DGX Spark
porting guide specifically recommends `CMAKE_CUDA_ARCHITECTURES="121-real"`.
The current llama.cpp fork deliberately rewrites requested plain Blackwell
architectures to architecture-specific code: `120-real` becomes `120a-real`
and `121-real` becomes `121a-real`. This enables Blackwell-specific
instructions and is not forward-compatible with later architectures.

Sources:

- <https://developer.nvidia.com/cuda/gpus>
- <https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/compilation.html>
- <https://docs.nvidia.com/dgx/dgx-spark/system-overview.html>
- <https://docs.nvidia.com/dgx/dgx-spark/release-notes.html>

## Current support boundary

Native AVQ2, affine INT4, vocabulary projection, and embedding kernels have
passed exact-GGUF full-model execution on one Linux x86_64 RTX PRO 6000
Blackwell Server Edition (`sm_120a`). Three native benchmark repetitions
measured 84.70 prompt tokens/second and 43.16 generated tokens/second.

That measurement validates the implementation on that device. It does not
automatically validate a separately built archive, RTX 4090, RTX 5090,
Windows, or DGX Spark. The complete Linux arm64 runner cross-compiles under
CUDA 13.0.2 for DGX Spark `sm_121a`, but remains a preview until target-device
execution. Do not advertise unmeasured targets as supported.

## Linux x86_64 and DGX Spark builds

The Docker definition uses NVIDIA's multi-architecture
`nvidia/cuda:13.0.2-devel-ubuntu24.04` image. Its published manifest contains
both `linux/amd64` and `linux/arm64`. The Dockerfile pins manifest-list digest
`sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6`
so the tag cannot silently change the toolchain.

Build an RTX Linux bundle:

```bash
packaging/cuda/build-linux.sh \
  linux-x86_64 artifacts/runtime/BTL-3-Compact-linux-x86_64-cuda
```

Build natively on DGX Spark:

```bash
packaging/cuda/build-linux.sh \
  linux-arm64 artifacts/runtime/BTL-3-Compact-linux-arm64-cuda
```

Cross-building arm64 through Docker Buildx is useful as a compile/package
check, but it does not replace execution on DGX Spark.

## Windows x64 build

Run this in PowerShell with Visual Studio C++ tools, CMake, Python, and CUDA
Toolkit 13.0.x installed:

```powershell
.\packaging\cuda\build-windows.ps1 `
  -Output artifacts\runtime\BTL-3-Compact-windows-x86_64-cuda
```

The package contains `llama-server.exe`, `llama-cli.exe`, the produced llama
and GGML DLLs, and the required redistributable CUDA runtime DLLs. The NVIDIA
display driver remains a system prerequisite.

## Bundle contents

The model is deliberately external to every runtime bundle:

- `libexec/llama-server[.exe]`
- `libexec/llama-cli[.exe]`
- `lib/` runtime dependency closure
- `bin/btl3-server[.ps1]`
- `bundle-manifest.json`
- `model/` empty destination for the model

The manifest records checksums for packaged files and the expected external
model:

- filename: `BTL-3-Compact-AVQ2.gguf`
- bytes: `8,392,369,600`
- SHA-256:
  `2ddf9527620a17a2a6739d184a7096c45712092e6589128792ec6254e94dc30c`

## Safe context defaults

The launcher detects the first NVIDIA GPU's total memory. DGX Spark falls back
to system memory because CPU and GPU share its 128 GB unified pool.

| Detected memory | Default context |
|---:|---:|
| below 20,000 MiB | 16,384 |
| 20,000–27,999 MiB | 32,768 |
| 28,000–47,999 MiB | 65,536 |
| 48,000–95,999 MiB | 98,304 |
| at least 96,000 MiB | 131,072 |

These defaults reserve memory for the 8.39 GB model, runtime workspace, and
other processes. Override them with `BTL3_CTX_SIZE`. Override faulty hardware
detection with `BTL3_GPU_MEMORY_MIB`. On DGX Spark, the launcher enables
`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`; lower context if the system is under
memory pressure.

Place the GGUF in the bundle's `model` directory or set `BTL3_MODEL`, then run:

```bash
BTL3_PRINT_COMMAND=1 ./bin/btl3-server
./bin/btl3-server
```

For a verified per-user installation used automatically by the LM Studio
plugin, install the downloaded runtime and GGUF in one command:

```bash
python3 tools/install_consumer_bundle.py \
  --runtime artifacts/runtime/BTL-3-Compact-linux-x86_64-cuda \
  --model artifacts/release/BTL-3-Compact-AVQ2.gguf
```

The installer checks every runtime hash plus the complete 8.39 GB model hash
before atomically installing. Its defaults are `~/.local/share/btl3` on Linux
and `%LOCALAPPDATA%\BTL3` on Windows.

On Windows:

```powershell
$env:BTL3_PRINT_COMMAND = "1"
.\bin\btl3-server.ps1
```

The print mode validates model discovery, memory detection, and chosen context
without starting the server.

## Required NVIDIA conformance

Before changing the support label:

1. Build both Linux architectures and Windows x64 from clean environments.
2. Run `test-btl3-avq-cuda`, `test-btl3-int4-cuda`, and
   `test-btl3-vocab-cuda` on RTX 4090, RTX 5090, and DGX Spark.
3. Record numerical parity and packed-probe placement for all four custom
   operation families.
4. Load the exact external GGUF and execute prompt plus decode.
5. Run transport, reasoning, tool-call, cancellation, and behavior gates.
6. Record throughput, peak memory, driver, CUDA runtime, and artifact hashes.

No throughput or compatibility number should be inferred from a successful
cross-compile.
