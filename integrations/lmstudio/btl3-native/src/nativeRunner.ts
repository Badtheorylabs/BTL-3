import { execFileSync, spawn, type ChildProcess } from "node:child_process";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

type RunnerOptions = {
  baseUrl: string;
  runnerPath: string;
  modelPath: string;
};

let child: ChildProcess | undefined;
let startup: Promise<void> | undefined;
let stderrTail = "";
let childError: Error | undefined;

function installedPaths(): { runner: string; model: string } {
  if (process.platform === "win32") {
    const root = process.env.LOCALAPPDATA ?? join(homedir(), "AppData", "Local");
    return {
      runner: join(root, "BTL3", "libexec", "llama-server.exe"),
      model: join(root, "BTL3", "model", "BTL-3-Compact-AVQ2.gguf"),
    };
  }
  const root = process.env.XDG_DATA_HOME ?? join(homedir(), ".local", "share");
  return {
    runner: join(root, "btl3", "libexec", "llama-server"),
    model: join(root, "btl3", "model", "BTL-3-Compact-AVQ2.gguf"),
  };
}

function endpoint(baseUrl: string, path: string): URL {
  const url = new URL(baseUrl);
  url.pathname = path;
  url.search = "";
  return url;
}

async function healthy(baseUrl: string): Promise<boolean> {
  try {
    const response = await fetch(endpoint(baseUrl, "/health"), {
      signal: AbortSignal.timeout(1_500),
    });
    return response.ok;
  } catch {
    return false;
  }
}

function detectGpuMemoryMiB(): number | undefined {
  const configured = Number(process.env.BTL3_GPU_MEMORY_MIB ?? "");
  if (Number.isFinite(configured) && configured > 0) return configured;
  if (process.platform === "darwin") return undefined;
  try {
    const output = execFileSync(
      "nvidia-smi",
      ["--query-gpu=memory.total", "--format=csv,noheader,nounits"],
      { encoding: "utf8", timeout: 3_000, windowsHide: true },
    );
    const detected = Number(output.split(/\r?\n/, 1)[0]?.replace(/[^0-9]/g, ""));
    return Number.isFinite(detected) && detected > 0 ? detected : undefined;
  } catch {
    return undefined;
  }
}

function chooseContext(memoryMiB: number | undefined): string {
  if (process.env.BTL3_CTX_SIZE) return process.env.BTL3_CTX_SIZE;
  if (process.platform === "darwin") return "4096";
  if (!memoryMiB || memoryMiB < 20_000) return "16384";
  if (memoryMiB < 28_000) return "32768";
  if (memoryMiB < 48_000) return "65536";
  if (memoryMiB < 96_000) return "98304";
  return "131072";
}

function runnerArguments(baseUrl: string, modelPath: string): string[] {
  const url = new URL(baseUrl);
  const port = url.port || "8080";
  const contextSize = chooseContext(detectGpuMemoryMiB());
  const gpuLayers = process.env.BTL3_GPU_LAYERS || "99";
  const maxTokens = process.env.BTL3_MAX_TOKENS || "2048";
  const repeatPenalty = process.env.BTL3_REPEAT_PENALTY || "1.10";
  const repeatLastN = process.env.BTL3_REPEAT_LAST_N || "512";
  return [
    "--model", modelPath,
    "--alias", "BTL-3",
    "--host", url.hostname,
    "--port", port,
    "--ctx-size", contextSize,
    "--parallel", "1",
    "--n-gpu-layers", gpuLayers,
    "--jinja",
    "--reasoning", "auto",
    "--reasoning-format", "deepseek",
    "--chat-template-kwargs", '{"enable_thinking":false}',
    "--cont-batching",
    "--n-predict", maxTokens,
    "--repeat-penalty", repeatPenalty,
    "--repeat-last-n", repeatLastN,
    "--cache-ram", "0",
    "--no-warmup",
    "--no-ui",
    "--offline",
  ];
}

async function waitUntilReady(baseUrl: string): Promise<void> {
  const deadline = Date.now() + 60_000;
  while (Date.now() < deadline) {
    if (await healthy(baseUrl)) return;
    if (childError) {
      throw new Error(`BTL-3 runner failed to start: ${childError.message}`);
    }
    if (child?.exitCode !== null && child?.exitCode !== undefined) {
      throw new Error(`BTL-3 runner exited (${child.exitCode}): ${stderrTail}`);
    }
    await new Promise(resolve => setTimeout(resolve, 250));
  }
  throw new Error(`BTL-3 runner did not become ready: ${stderrTail}`);
}

async function start(options: RunnerOptions): Promise<void> {
  if (await healthy(options.baseUrl)) return;
  const defaults = installedPaths();
  const runner = options.runnerPath || process.env.BTL3_RUNNER_PATH || defaults.runner;
  const model = options.modelPath || process.env.BTL3_MODEL_PATH || defaults.model;
  if (!existsSync(runner)) {
    throw new Error(`BTL-3 runner not found at ${runner}. Install the native runtime or configure Native runner path.`);
  }
  if (!existsSync(model)) {
    throw new Error(`BTL-3 model not found at ${model}. Install the verified GGUF or configure BTL-3 model path.`);
  }
  const root = dirname(dirname(runner));
  const libraryPath = join(root, "lib");
  const env = { ...process.env };
  if (process.platform === "win32") {
    env.PATH = `${libraryPath};${env.PATH ?? ""}`;
    env.GGML_BACKEND_PATH = join(libraryPath, "ggml-cuda.dll");
  } else if (process.platform === "linux") {
    env.LD_LIBRARY_PATH =
      `${libraryPath}${env.LD_LIBRARY_PATH ? `:${env.LD_LIBRARY_PATH}` : ""}`;
    env.GGML_BACKEND_PATH = join(libraryPath, "libggml-cuda.so");
  } else {
    env.DYLD_LIBRARY_PATH =
      `${libraryPath}${env.DYLD_LIBRARY_PATH ? `:${env.DYLD_LIBRARY_PATH}` : ""}`;
  }
  stderrTail = "";
  childError = undefined;
  child = spawn(runner, runnerArguments(options.baseUrl, model), {
    windowsHide: true,
    stdio: ["ignore", "ignore", "pipe"],
    env,
  });
  child.stderr?.on("data", data => {
    stderrTail = (stderrTail + String(data)).slice(-4_096);
  });
  child.once("error", error => {
    childError = error;
    stderrTail = (stderrTail + error.message).slice(-4_096);
  });
  await waitUntilReady(options.baseUrl);
}

export async function ensureNativeRunner(options: RunnerOptions): Promise<void> {
  if (await healthy(options.baseUrl)) return;
  startup ??= start(options).finally(() => {
    startup = undefined;
  });
  await startup;
}

process.once("exit", () => {
  child?.kill();
});
