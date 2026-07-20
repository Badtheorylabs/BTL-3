# BTL-3 Product Integrations Contract

## Ownership boundary

This repository owns product-facing integration only:

- OpenAI-compatible local serving;
- streaming, cancellation, reasoning and tool-call transport;
- LM Studio generator integration;
- Ollama packaging and patched-runner distribution;
- installers, diagnostics and launch documentation;
- backend conformance and integration tests.

The packed model format, GGUF exporter, GGML tensor implementations and
AVQ2/INT4 kernels belong to the separate runtime repository maintained by the
runtime lane. Do not duplicate or silently fork those implementations here.

## Frozen model

The source model is the exact BTL-3 Compact package:

- package bytes: `8,572,102,407`;
- declared payload bytes: `8,572,070,080`;
- weight bytes: `8,551,772,952`;
- package manifest SHA-256:
  `a2b763323eed76d8f78fe5cbdf5a2349323b2c3d87dddc037714569946961116`.

Never modify, repack or requantize the source package from this repository.

## Engineering rules

1. No file may exceed 500 lines.
2. No slop code, placeholder behavior or fake compatibility claims.
3. Use tests before implementation for protocol and packaging behavior.
4. Integrations must preserve streaming, cancellation, tool calls and reasoning.
5. A backend is supported only after conformance tests pass against its real
   executable.
6. Stock Ollama compatibility and a patched Ollama distribution must be labeled
   separately.
7. No paid compute, deployment or publishing without explicit user approval.
8. Never require a persistent dense reconstruction of BTL-3 Compact.
