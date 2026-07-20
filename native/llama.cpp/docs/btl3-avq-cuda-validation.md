# BTL-3 compact CUDA validation

## Implemented contract

The NVIDIA CUDA backend accepts four independently marked BTL-3 operations:

1. AVQ2 repeating-layer matmul, including signed normalized 128-wide
   Walsh-Hadamard transforms on both sides.
2. Affine INT4 matmul with packed nibbles and per-128-column FP16 scales and
   zero points.
3. Vocabulary get-rows for both AVQ2 rows and rescued packed 8-bit rows.
   Invalid token IDs explicitly produce zero rows.
4. Vocabulary-head matmul with the input sign and transform semantics used by
   the CPU and Metal implementations.

Packed weights are decoded in the kernels. No dense weight matrix is
materialized. Each representation has its own model-loader placement probe;
CUDA cannot claim one packed tensor merely because it supports another.
Unmarked custom operations remain unsupported.

The kernels use portable CUDA C++ operations and do not hardcode an SM
architecture. The build selects the target architecture.

## Local checks completed

The host-only contract test validates all operation markers, role-specific
placement probes, types, shapes and CPU reference behavior:

```sh
cmake --build build-btl3-sdk --target test-btl3-avq -j
./build-btl3-sdk/bin/test-btl3-avq
git diff --check
```

These checks pass locally. The CUDA test sources also pass host C++ syntax
checking, but this machine has neither NVCC nor an NVIDIA GPU. CUDA compilation,
device correctness and speed are therefore not claimed.

## Device correctness tests

The three CUDA tests execute marked GGML graphs on the first CUDA device and
compare every relevant output against deterministic CPU/reference values:

```sh
cmake --build BUILD_DIR --target \
  test-btl3-avq-cuda \
  test-btl3-int4-cuda \
  test-btl3-vocab-cuda -j
ctest --test-dir BUILD_DIR \
  -R '^test-btl3-(avq|int4|vocab)-cuda$' \
  --output-on-failure
```

The vocabulary test covers ordinary AVQ2 rows, rescued rows, invalid-token
zeroing and the vocabulary head. This is the required parity gate with the
CPU/Metal semantics.

## RTX 4090 (Ada, SM89)

Use CUDA 11.8 or newer:

```sh
cmake -S . -B build-cuda-sm89 \
  -DGGML_CUDA=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DGGML_NATIVE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda-sm89 --target \
  test-btl3-avq-cuda \
  test-btl3-int4-cuda \
  test-btl3-vocab-cuda -j
ctest --test-dir build-cuda-sm89 \
  -R '^test-btl3-(avq|int4|vocab)-cuda$' \
  --output-on-failure
```

## RTX 5090 (Blackwell, SM120)

Use CUDA 12.8 or newer. This tree normalizes plain `120` to the
architecture-specific `120a` target:

```sh
cmake -S . -B build-cuda-sm120 \
  -DGGML_CUDA=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DGGML_NATIVE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-cuda-sm120 --target \
  test-btl3-avq-cuda \
  test-btl3-int4-cuda \
  test-btl3-vocab-cuda -j
ctest --test-dir build-cuda-sm120 \
  -R '^test-btl3-(avq|int4|vocab)-cuda$' \
  --output-on-failure
```

## DGX Spark (Linux arm64, GB10, SM121)

NVIDIA lists GB10 as compute capability 12.1. Use CUDA 12.9 or newer and build
natively on the arm64 host:

```sh
test "$(uname -m)" = aarch64
cmake -S . -B build-cuda-sm121 \
  -DGGML_CUDA=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DGGML_NATIVE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build build-cuda-sm121 --target \
  test-btl3-avq-cuda \
  test-btl3-int4-cuda \
  test-btl3-vocab-cuda -j
ctest --test-dir build-cuda-sm121 \
  -R '^test-btl3-(avq|int4|vocab)-cuda$' \
  --output-on-failure
```

## Full-model placement gate

After unit correctness passes, load the compact GGUF with the desired GPU
offload and retain the loader/backend debug log. Confirm that:

- repeating AVQ2 tensors select CUDA through the AVQ2 probe;
- affine INT4 tensors select CUDA through the INT4 probe;
- input AVQ2 and rescue tensors select CUDA through the vocabulary-row probe;
- output AVQ2 tensors select CUDA through the vocabulary-head probe;
- no marked BTL-3 custom node falls back to CPU during prompt or decode.

Only then profile prompt and decode separately. Do not publish a throughput or
latency claim from source inspection or unit tests.
