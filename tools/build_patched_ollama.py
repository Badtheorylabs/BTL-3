"""Package Ollama with the BTL-3 native llama-server runtime payload."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import tempfile


OLLAMA_COMMIT = "573386c35eac76124ffce571f4b0fefa0a7fe13c"
MODEL_NAME = "BTL-3-Compact-AVQ2.gguf"
MODEL_BYTES = 8_392_369_600
MODEL_SHA256 = "2ddf9527620a17a2a6739d184a7096c45712092e6589128792ec6254e94dc30c"
REQUIRED_RUNNER_FLAGS = (
    "--model",
    "--port",
    "--host",
    "--no-webui",
    "--offline",
    "-np",
)
PLATFORMS = {
    "linux-amd64-cuda12": {"cuda_dir": "cuda_v12", "windows": False},
    "linux-amd64-cuda13": {"cuda_dir": "cuda_v13", "windows": False},
    "linux-arm64-cuda12": {"cuda_dir": "cuda_v12", "windows": False},
    "linux-arm64-cuda13": {"cuda_dir": "cuda_v13", "windows": False},
    "windows-amd64-cuda13": {"cuda_dir": "cuda_v13", "windows": True},
}


class PackageError(ValueError):
    """The input cannot produce an honestly labeled patched distribution."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def executable(path: Path) -> bool:
    return path.is_file() and bool(path.stat().st_mode & stat.S_IXUSR)


def validate_runner_cli(path: Path) -> None:
    try:
        result = subprocess.run(
            [path, "--help"],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PackageError(f"runner CLI contract probe failed: {error}") from error
    help_text = result.stdout + result.stderr
    missing = [flag for flag in REQUIRED_RUNNER_FLAGS if flag not in help_text]
    if result.returncode != 0 or missing:
        raise PackageError(
            f"runner CLI contract missing flags {missing}; exit={result.returncode}"
        )


def validate_pe_x64(path: Path, label: str) -> None:
    try:
        with path.open("rb") as stream:
            header = stream.read(64)
            if len(header) < 64 or header[:2] != b"MZ":
                raise PackageError(f"{label} is not a PE x64 binary: {path}")
            pe_offset = struct.unpack_from("<I", header, 0x3C)[0]
            stream.seek(pe_offset)
            pe_header = stream.read(6)
    except OSError as error:
        raise PackageError(f"cannot read {label}: {path}: {error}") from error
    if len(pe_header) != 6 or pe_header[:4] != b"PE\0\0":
        raise PackageError(f"{label} is not a PE x64 binary: {path}")
    machine = struct.unpack_from("<H", pe_header, 4)[0]
    if machine != 0x8664:
        raise PackageError(
            f"{label} is not PE x64 (machine=0x{machine:04x}): {path}"
        )


def validate_windows_contract(payload: Path, runner: Path, platform: str) -> None:
    contract_path = payload / "runner-cli-contract.json"
    if not contract_path.is_file():
        raise PackageError(
            "Windows runner lacks runner-cli-contract.json from target probe"
        )
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise PackageError(f"invalid Windows runner CLI contract: {error}") from error
    if contract.get("schema_version") != 1 or contract.get("platform") != platform:
        raise PackageError("Windows runner CLI contract platform/schema mismatch")
    if contract.get("runner_sha256") != sha256(runner):
        raise PackageError("Windows runner SHA-256 does not match CLI contract")
    flags = contract.get("required_flags")
    missing = [flag for flag in REQUIRED_RUNNER_FLAGS if flag not in (flags or [])]
    if missing:
        raise PackageError(f"Windows runner CLI contract lacks flags: {missing}")


def validate_inputs(
    ollama_dist: Path, runner_payload: Path, output: Path, platform: str
) -> dict[str, str | bool]:
    if platform not in PLATFORMS:
        raise PackageError(f"unsupported platform: {platform}")
    config = PLATFORMS[platform]
    if output.exists():
        raise PackageError(f"output already exists: {output}")
    cuda_dir = str(config["cuda_dir"])
    if config["windows"]:
        ollama = ollama_dist / "ollama.exe"
        runner = runner_payload / "llama-server.exe"
        cuda_backend = runner_payload / cuda_dir / "ggml-cuda.dll"
        validate_pe_x64(ollama, "Ollama executable")
        validate_pe_x64(runner, "runner executable")
        validate_pe_x64(cuda_backend, f"{cuda_dir} CUDA backend")
        for candidate in runner_payload.rglob("*"):
            if candidate.is_file() and candidate.suffix.lower() == ".dll":
                validate_pe_x64(candidate, "runner DLL")
        validate_windows_contract(runner_payload, runner, platform)
    else:
        if not executable(ollama_dist / "bin/ollama"):
            raise PackageError("Ollama distribution lacks executable bin/ollama")
        runner = runner_payload / "llama-server"
        if not executable(runner):
            raise PackageError("runner payload lacks executable llama-server")
        validate_runner_cli(runner)
        if not (runner_payload / cuda_dir / "libggml-cuda.so").is_file():
            raise PackageError(f"runner payload lacks {cuda_dir} CUDA backend")
    for path in runner_payload.rglob("*"):
        if "bridge" in path.name.lower():
            raise PackageError("direct-runner payload must not contain an HTTP bridge")
    return config


def write_unix_wrapper(path: Path) -> None:
    path.write_text(
        """#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
if [ "${1:-}" = "--btl3-info" ]; then
  echo "BTL-3 Patched Ollama (direct native runner; not stock Ollama)"
  echo "runner=$root/lib/ollama/llama-server"
  exit 0
fi
exec "$root/bin/ollama" "$@"
"""
    )
    path.chmod(0o755)


def write_windows_wrapper(path: Path) -> None:
    path.write_text(
        """@echo off
setlocal
if /I "%~1"=="--btl3-info" (
  echo BTL-3 Patched Ollama ^(direct native runner; not stock Ollama^)
  echo runner=%~dp0lib\\ollama\\llama-server.exe
  exit /b 0
)
"%~dp0ollama.exe" %*
exit /b %ERRORLEVEL%
""",
        newline="\r\n",
    )


def file_manifest(root: Path) -> dict[str, dict[str, int | str]]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            result[path.relative_to(root).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        elif path.is_symlink():
            result[path.relative_to(root).as_posix()] = {
                "symlink": os.readlink(path)
            }
    return result


def build_package(
    *,
    ollama_dist: Path,
    runner_payload: Path,
    output: Path,
    platform: str,
) -> Path:
    ollama_dist = ollama_dist.resolve()
    runner_payload = runner_payload.resolve()
    output = output.resolve()
    config = validate_inputs(ollama_dist, runner_payload, output, platform)
    cuda_dir = str(config["cuda_dir"])
    windows = bool(config["windows"])
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        shutil.copytree(ollama_dist, staging, dirs_exist_ok=True, symlinks=True)
        runtime = staging / "lib/ollama"
        shutil.rmtree(runtime, ignore_errors=True)
        shutil.copytree(runner_payload, runtime, symlinks=True)
        if windows:
            write_windows_wrapper(staging / "btl3-ollama.cmd")
            runner_path = "lib/ollama/llama-server.exe"
        else:
            write_unix_wrapper(staging / "bin/btl3-ollama")
            runner_path = "lib/ollama/llama-server"
        metadata = staging / "share/btl3"
        metadata.mkdir(parents=True)
        manifest = {
            "schema_version": 1,
            "product": "BTL-3 Patched Ollama",
            "label": "patched; not stock Ollama",
            "integration": "direct-subprocess",
            "stock_ollama_compatible": False,
            "ollama": {"commit": OLLAMA_COMMIT},
            "platform": platform,
            "runner": {
                "path": runner_path,
                "cuda_backend": cuda_dir,
            },
            "external_model": {
                "filename": MODEL_NAME,
                "bytes": MODEL_BYTES,
                "sha256": MODEL_SHA256,
            },
            "runtime_files": file_manifest(runtime),
        }
        (metadata / "patched-ollama-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama-dist", type=Path, required=True)
    parser.add_argument("--runner-payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", choices=sorted(PLATFORMS), required=True)
    args = parser.parse_args()
    print(
        build_package(
            ollama_dist=args.ollama_dist,
            runner_payload=args.runner_payload,
            output=args.output,
            platform=args.platform,
        )
    )


if __name__ == "__main__":
    main()
