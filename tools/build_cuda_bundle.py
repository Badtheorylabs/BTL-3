"""Package a prebuilt BTL-3 CUDA runtime without embedding model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "BTL-3-Compact-AVQ2.gguf"
MODEL_BYTES = 8_392_369_600
MODEL_SHA256 = "2ddf9527620a17a2a6739d184a7096c45712092e6589128792ec6254e94dc30c"
PLATFORM_SPECS = {
    "linux-x86_64": {
        "label": "Linux x86_64 CUDA (RTX 4090/5090)",
        "format": "elf",
        "machine": 62,
        "cuda_architectures": "89-real;120-real",
        "server": "llama-server",
        "cli": "llama-cli",
        "launcher": "btl3-cuda-server",
    },
    "linux-arm64": {
        "label": "Linux arm64 CUDA (DGX Spark)",
        "format": "elf",
        "machine": 183,
        "cuda_architectures": "121-real",
        "server": "llama-server",
        "cli": "llama-cli",
        "launcher": "btl3-cuda-server",
    },
    "windows-x86_64": {
        "label": "Windows x64 CUDA (RTX 4090/5090)",
        "format": "pe",
        "machine": 0x8664,
        "cuda_architectures": "89-real;120-real",
        "server": "llama-server.exe",
        "cli": "llama-cli.exe",
        "launcher": "btl3-cuda-server.ps1",
    },
}


def choose_context(memory_mib: int) -> int:
    """Choose a conservative context while reserving room for weights/workspace."""
    if memory_mib >= 96_000:
        return 131_072
    if memory_mib >= 48_000:
        return 98_304
    if memory_mib >= 28_000:
        return 65_536
    if memory_mib >= 20_000:
        return 32_768
    return 16_384


def executable_machine(path: Path, file_format: str) -> int:
    data = path.read_bytes()
    if file_format == "elf":
        if len(data) < 20 or data[:4] != b"\x7fELF" or data[4:6] != b"\x02\x01":
            raise ValueError(f"{path.name} is not a 64-bit little-endian ELF")
        return struct.unpack_from("<H", data, 18)[0]
    if len(data) < 64 or data[:2] != b"MZ":
        raise ValueError(f"{path.name} is not a PE executable")
    offset = struct.unpack_from("<I", data, 0x3C)[0]
    if len(data) < offset + 6 or data[offset : offset + 4] != b"PE\0\0":
        raise ValueError(f"{path.name} has an invalid PE header")
    return struct.unpack_from("<H", data, offset + 4)[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_runtime_files(source: Path, output: Path, target: str) -> None:
    spec = PLATFORM_SPECS[target]
    expected = spec["machine"]
    machine_name = "x86_64" if expected in (62, 0x8664) else "arm64"
    for name in (spec["server"], spec["cli"]):
        path = source / name
        if not path.is_file():
            raise ValueError(f"missing required executable: {path}")
        actual = executable_machine(path, spec["format"])
        if actual != expected:
            raise ValueError(
                f"{name} is machine {actual}, expected {machine_name} ({expected})"
            )
        destination = output / "libexec" / name
        shutil.copy2(path, destination)
        if spec["format"] == "elf":
            destination.chmod(0o755)

    library_patterns = ("*.so", "*.so.*") if spec["format"] == "elf" else ("*.dll",)
    seen: set[str] = set()
    for pattern in library_patterns:
        for path in sorted(source.glob(pattern)):
            if path.name in seen or not path.is_file():
                continue
            shutil.copy2(path, output / "lib" / path.name)
            seen.add(path.name)
    cuda_library = (
        any(name.startswith("libggml-cuda.so") for name in seen)
        if spec["format"] == "elf"
        else any(name.lower().startswith("ggml-cuda") for name in seen)
    )
    if not cuda_library:
        raise ValueError(f"{source} does not contain the CUDA backend library")
    for path in sorted(source.glob("LICENSE.*")):
        if path.name != "LICENSE.llama.cpp":
            shutil.copy2(path, output / path.name)


def write_manifest(output: Path, target: str) -> None:
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file():
            files[path.relative_to(output).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    manifest = {
        "schema_version": 1,
        "bundle": f"BTL-3 Compact {PLATFORM_SPECS[target]['label']}",
        "target": target,
        "cuda": {
            "toolkit": "13.0.2",
            "architectures": PLATFORM_SPECS[target]["cuda_architectures"],
        },
        "status": "cross-compiled; NVIDIA runtime conformance pending",
        "external_model": {
            "filename": MODEL_NAME,
            "bytes": MODEL_BYTES,
            "sha256": MODEL_SHA256,
        },
        "files": files,
    }
    (output / "bundle-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def package_bundle(target: str, source: Path, output: Path) -> Path:
    if target not in PLATFORM_SPECS:
        raise ValueError(f"unknown target: {target}")
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for name in ("bin", "lib", "libexec", "model"):
            (staging / name).mkdir(parents=True, exist_ok=True)
        copy_runtime_files(source, staging, target)
        launcher_name = PLATFORM_SPECS[target]["launcher"]
        launcher_source = ROOT / "launch" / launcher_name
        launcher_target = staging / "bin" / launcher_name.replace("-cuda", "")
        shutil.copy2(launcher_source, launcher_target)
        if target != "windows-x86_64":
            launcher_target.chmod(0o755)
        shutil.copy2(
            ROOT / "native/llama.cpp/LICENSE", staging / "LICENSE.llama.cpp"
        )
        write_manifest(staging, target)
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=sorted(PLATFORM_SPECS), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(package_bundle(args.target, args.source, args.output))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
