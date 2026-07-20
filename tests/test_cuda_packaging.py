from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from build_cuda_bundle import PLATFORM_SPECS, choose_context, package_bundle


def write_elf(path: Path, machine: int) -> None:
    header = bytearray(64)
    header[:4] = b"\x7fELF"
    header[4] = 2
    header[5] = 1
    struct.pack_into("<H", header, 18, machine)
    path.write_bytes(header)
    path.chmod(0o755)


def write_pe(path: Path, machine: int = 0x8664) -> None:
    header = bytearray(256)
    header[:2] = b"MZ"
    struct.pack_into("<I", header, 0x3C, 128)
    header[128:132] = b"PE\0\0"
    struct.pack_into("<H", header, 132, machine)
    path.write_bytes(header)


def test_cuda_platforms_are_pinned_to_consumer_targets() -> None:
    assert PLATFORM_SPECS["linux-x86_64"]["cuda_architectures"] == (
        "89-real;120-real"
    )
    assert PLATFORM_SPECS["windows-x86_64"]["cuda_architectures"] == (
        "89-real;120-real"
    )
    assert PLATFORM_SPECS["linux-arm64"]["cuda_architectures"] == "121-real"


def test_build_definitions_pin_cuda_and_forward_architectures() -> None:
    dockerfile = (ROOT / "packaging/cuda/Dockerfile").read_text()
    linux = (ROOT / "packaging/cuda/build-linux.sh").read_text()
    windows = (ROOT / "packaging/cuda/build-windows.ps1").read_text()
    assert "nvidia/cuda:13.0.2-devel-ubuntu24.04" in dockerfile
    assert "CMAKE_CUDA_ARCHITECTURES" in dockerfile
    assert "GGML_BACKEND_DL=ON" in dockerfile
    # Cross-build containers contain the CUDA toolkit but no host driver.
    # Device tests are built and executed by the native hardware gates.
    assert "LLAMA_BUILD_TESTS=OFF" in dockerfile
    assert "89-real;120-real" in linux
    assert "121-real" in linux
    assert "89-real;120-real" in windows
    assert "GGML_BACKEND_DL=ON" in windows


@pytest.mark.parametrize(
    ("memory_mib", "expected"),
    [
        (16_000, 16_384),
        (24_000, 32_768),
        (32_000, 65_536),
        (64_000, 98_304),
        (120_000, 131_072),
    ],
)
def test_context_selection_reserves_memory(memory_mib: int, expected: int) -> None:
    assert choose_context(memory_mib) == expected


def test_linux_bundle_rejects_wrong_machine(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_elf(source / "llama-server", 183)
    write_elf(source / "llama-cli", 183)
    with pytest.raises(ValueError, match="x86_64"):
        package_bundle("linux-x86_64", source, tmp_path / "bundle")


def test_linux_bundle_rejects_missing_cuda_backend(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_elf(source / "llama-server", 62)
    write_elf(source / "llama-cli", 62)
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="CUDA backend"):
        package_bundle("linux-x86_64", source, output)
    assert not output.exists()


def test_linux_bundle_is_relocatable_and_keeps_model_external(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_elf(source / "llama-server", 62)
    write_elf(source / "llama-cli", 62)
    (source / "libggml-cuda.so").write_bytes(b"cuda")
    output = tmp_path / "bundle"
    package_bundle("linux-x86_64", source, output)

    assert (output / "bin/btl3-server").stat().st_mode & 0o111
    assert (output / "libexec/llama-server").exists()
    assert (output / "libexec/llama-cli").exists()
    assert (output / "lib/libggml-cuda.so").exists()
    assert not list(output.rglob("*.gguf"))
    manifest = json.loads((output / "bundle-manifest.json").read_text())
    assert manifest["target"] == "linux-x86_64"
    assert manifest["external_model"]["bytes"] == 8_392_369_600


def test_windows_bundle_includes_native_launcher(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_pe(source / "llama-server.exe")
    write_pe(source / "llama-cli.exe")
    (source / "ggml-cuda.dll").write_bytes(b"cuda backend")
    (source / "cudart64_13.dll").write_bytes(b"cuda")
    output = tmp_path / "bundle"
    package_bundle("windows-x86_64", source, output)

    assert (output / "bin/btl3-server.ps1").exists()
    assert (output / "libexec/llama-server.exe").exists()
    assert (output / "lib/cudart64_13.dll").exists()


def test_posix_launcher_prints_safe_detected_context(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    (bundle / "bin").mkdir(parents=True)
    (bundle / "libexec").mkdir()
    (bundle / "model").mkdir()
    server = bundle / "libexec/llama-server"
    server.write_text("#!/bin/sh\nexit 0\n")
    server.chmod(0o755)
    model = bundle / "model/BTL-3-Compact-AVQ2.gguf"
    model.write_bytes(b"x")
    launcher = bundle / "bin/btl3-server"
    launcher.write_bytes((ROOT / "launch/btl3-cuda-server").read_bytes())
    launcher.chmod(0o755)

    result = subprocess.run(
        [str(launcher)],
        check=True,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "BTL3_GPU_MEMORY_MIB": "24576",
            "BTL3_PRINT_COMMAND": "1",
        },
    )
    assert "ctx_size=32768" in result.stdout
    assert "gpu_memory_mib=24576" in result.stdout


def test_launchers_point_dynamic_loader_at_packaged_cuda_backend() -> None:
    linux = (ROOT / "launch/btl3-cuda-server").read_text()
    windows = (ROOT / "launch/btl3-cuda-server.ps1").read_text()
    assert 'GGML_BACKEND_PATH="$bundle_root/lib/libggml-cuda.so"' in linux
    assert '"lib\\ggml-cuda.dll"' in windows
