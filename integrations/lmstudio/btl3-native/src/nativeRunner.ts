import { spawn, type ChildProcess } from "node:child_process";
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

function runnerArguments(baseUrl: string, modelPath: string): string[] {
  const url = new URL(baseUrl);
  const port = url.port || "8080";
  return [
    "--model", modelPath,
    "--host", url.hostname,
    "--port", port,
    "--no-webui",
    "--offline",
    "-np", "1",
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
  const root = dirname(dirname(runner));
  const libraryPath = join(root, "lib");
  const env = { ...process.env };
  if (process.platform === "win32") {
    env.PATH = `${libraryPath};${env.PATH ?? ""}`;
    env.GGML_BACKEND_PATH = join(libraryPath, "ggml-cuda.dll");
  } else {
    env.LD_LIBRARY_PATH =
      `${libraryPath}${env.LD_LIBRARY_PATH ? `:${env.LD_LIBRARY_PATH}` : ""}`;
    env.GGML_BACKEND_PATH = join(libraryPath, "libggml-cuda.so");
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
