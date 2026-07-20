# Third-party notices

## llama.cpp

The native runtime vendors a pinned llama.cpp source snapshot and includes
BTL-3-specific packed-model loaders and Metal/CUDA kernels. llama.cpp remains
available under its MIT license, included at `native/llama.cpp/LICENSE`.

## Qwen3.6-27B

BTL-3 is post-trained from `Qwen/Qwen3.6-27B`, revision
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`. Model weights are distributed
separately under Apache License 2.0 and are never committed to this source
repository.

## Platform libraries

Runtime bundles may link Apple Metal/Accelerate, OpenSSL, or NVIDIA CUDA
redistributable libraries. Those components retain their respective licenses
and notices.
