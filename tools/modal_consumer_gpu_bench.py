"""Measure BTL-3 Compact on Modal's RTX PRO 6000 without idle GPU time."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time

import modal


ROOT = Path(__file__).resolve().parents[1]
MODEL = "/vol/release/compact/BTL-3-Compact-AVQ2.gguf"
volume = modal.Volume.from_name("btl-vector2-vol")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("build-essential", "cmake", "ninja-build", "pkg-config")
    .add_local_dir(
        ROOT / "native/llama.cpp",
        "/src",
        copy=True,
        ignore=[
            ".git",
            ".git/**",
            "build-btl3",
            "build-btl3/**",
            "build-btl3-sdk",
            "build-btl3-sdk/**",
            "tools/ui",
            "tools/ui/**",
            "docs",
            "docs/**",
            "media",
            "media/**",
            "tests",
            "tests/**",
            ".github",
            ".github/**",
            "**/__pycache__/**",
            "*.pyc",
        ],
    )
    .run_commands(
        "cmake -S /src -B /src/build -G Ninja "
        "-DCMAKE_BUILD_TYPE=Release "
        "-DCMAKE_CUDA_ARCHITECTURES=120-real "
        "-DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF "
        "-DBUILD_SHARED_LIBS=ON -DLLAMA_OPENSSL=OFF "
        "-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON "
        "-DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=ON "
        "-DLLAMA_BUILD_UI=OFF",
        "cmake --build /src/build --target llama-bench --parallel",
    )
)

app = modal.App("btl3-rtx-pro-6000-speed")


def run(command: list[str], *, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=True,
        timeout=timeout,
    )


@app.function(
    image=image,
    gpu="RTX-PRO-6000",
    volumes={"/vol": volume},
    timeout=900,
    startup_timeout=300,
    scaledown_window=2,
    max_containers=1,
)
def benchmark() -> dict:
    if not Path(MODEL).is_file():
        raise FileNotFoundError(MODEL)
    environment = {
        "GGML_BACKEND_PATH": "/src/build/bin/libggml-cuda.so",
    }
    gpu = run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,clocks.max.memory",
        "--format=csv,noheader,nounits",
    ]).stdout.strip()
    command = [
        "/src/build/bin/llama-bench",
        "-m", MODEL,
        "-ngl", "99",
        "-p", "512",
        "-n", "128",
        "-r", "3",
        "-o", "json",
    ]
    started = time.perf_counter()
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env={**__import__("os").environ, **environment},
        check=True,
        timeout=840,
    )
    elapsed = time.perf_counter() - started
    return {
        "gpu": gpu,
        "model_bytes": Path(MODEL).stat().st_size,
        "elapsed_seconds": elapsed,
        "bench": json.loads(result.stdout),
        "stderr_tail": result.stderr[-4_000:],
    }


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(benchmark.remote(), indent=2))
