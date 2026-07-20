import { createConfigSchematics } from "@lmstudio/sdk";

export const globalConfigSchematics = createConfigSchematics()
  .field(
    "baseUrl",
    "string",
    {
      displayName: "BTL-3 server URL",
      subtitle: "OpenAI-compatible endpoint exposed by the native BTL-3 runner.",
      placeholder: "http://127.0.0.1:8080/v1",
    },
    "http://127.0.0.1:8080/v1",
  )
  .field(
    "apiKey",
    "string",
    {
      displayName: "Local API key",
      subtitle: "Must match BTL3_API_KEY when authentication is enabled.",
      isProtected: true,
    },
    "btl3-local",
  )
  .field(
    "autoStart",
    "boolean",
    {
      displayName: "Start the native runner automatically",
      subtitle: "Launch BTL-3 locally when the configured endpoint is offline.",
    },
    true,
  )
  .field(
    "runnerPath",
    "string",
    {
      displayName: "Native runner path",
      subtitle: "Optional. Defaults to BTL3_RUNNER_PATH or the installed BTL-3 runtime.",
      placeholder: "/home/user/.local/share/btl3/libexec/llama-server",
    },
    "",
  )
  .field(
    "modelPath",
    "string",
    {
      displayName: "BTL-3 model path",
      subtitle: "Optional. Defaults to BTL3_MODEL_PATH or the installed compact GGUF.",
      placeholder: "/home/user/.local/share/btl3/model/BTL-3-Compact-AVQ2.gguf",
    },
    "",
  )
  .build();
