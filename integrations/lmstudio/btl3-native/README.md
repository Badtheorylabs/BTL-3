# BTL-3 Native for LM Studio

This generator exposes the native BTL-3 OpenAI-compatible server inside LM
Studio. It does not ask LM Studio's stock GGUF engine to load AVQ2.

## Local development install

1. Install the BTL-3 runtime and model in its platform-default location, or set
   `BTL3_RUNNER_PATH` and `BTL3_MODEL_PATH`.
2. Install the LM Studio CLI with `npx lmstudio install-cli` if needed.
3. In this directory run `npm ci`, then `lms dev`.
4. Select **badtheorylabs/btl3-native** in LM Studio's model picker. The plugin
   starts the native runner automatically when the local endpoint is offline.

The default endpoint is `http://127.0.0.1:8080/v1`. Change it in the plugin's
global settings when the native runner is elsewhere. Automatic startup can be
disabled when an independently managed server is preferred. The generator preserves
streamed content, reasoning fragments, cancellation, and parallel tool calls.
LM Studio's generator lifecycle has no per-fragment call ID, so parallel
tool-call fragments are buffered by call index and emitted sequentially after
the upstream stream finishes. This preserves the calls without misattributing
interleaved JSON fragments.

Thinking is disabled by default and a 2,048-token response cap is enforced.
The experimental thinking toggle is retained for diagnosis, but it is not
recommended in this release because it can repeat or fail to terminate.
The auto-start path explicitly requests full GPU offload and selects a safe
context from detected NVIDIA memory (4K on macOS, 32K on a detected 24 GB GPU).
Set `BTL3_CTX_SIZE` or `BTL3_GPU_LAYERS` before starting LM Studio to override
those launch values.

`model.yaml` is catalog metadata for the native-generator product. Its custom
`btl3-avq2-native` compatibility type deliberately avoids claiming that LM
Studio's stock GGUF engine can execute this model.

This plugin has not been published and remains a development preview. `lms
push` is intentionally outside this repository's local validation workflow.
