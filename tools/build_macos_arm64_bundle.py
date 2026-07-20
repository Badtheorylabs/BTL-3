"""Build a relocatable local macOS arm64 BTL-3 runtime bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "native/llama.cpp/build-btl3-sdk/bin"
DEFAULT_OUTPUT = ROOT / "artifacts/runtime/BTL-3-Compact-macos-arm64"
MODEL_NAME = "BTL-3-Compact-AVQ2.gguf"
MODEL_BYTES = 8_392_369_600
MODEL_SHA256 = "2ddf9527620a17a2a6739d184a7096c45712092e6589128792ec6254e94dc30c"


def run(*command: str | Path, capture: bool = False) -> str:
    result = subprocess.run(
        [str(part) for part in command],
        check=True,
        text=True,
        capture_output=capture,
    )
    return result.stdout if capture else ""


def dependencies(path: Path) -> list[str]:
    lines = run("otool", "-L", path, capture=True).splitlines()[1:]
    return [line.strip().split(" (", 1)[0] for line in lines if line.strip()]


def rpaths(path: Path) -> list[str]:
    lines = run("otool", "-l", path, capture=True).splitlines()
    found = []
    for index, line in enumerate(lines):
        if line.strip() == "cmd LC_RPATH":
            value = lines[index + 2].strip()
            found.append(value.removeprefix("path ").split(" (offset", 1)[0])
    return found


def resolve_dependency(source: Path, dependency: str) -> Path | None:
    if dependency.startswith("@rpath/"):
        candidate = source / Path(dependency).name
        return candidate if candidate.exists() else None
    if dependency.startswith("/opt/homebrew/") and Path(dependency).name in {
        "libssl.3.dylib",
        "libcrypto.3.dylib",
    }:
        return Path(dependency)
    return None


def copy_dependencies(source: Path, executable: Path, lib_dir: Path) -> list[Path]:
    queue = [executable]
    copied: dict[str, Path] = {}
    while queue:
        current = queue.pop()
        for dependency in dependencies(current):
            dependency_path = resolve_dependency(source, dependency)
            if dependency_path is None:
                continue
            alias = Path(dependency).name
            real_source = dependency_path.resolve()
            real_name = real_source.name
            if real_name not in copied:
                target = lib_dir / real_name
                shutil.copy2(real_source, target)
                copied[real_name] = target
                queue.append(real_source)
            if alias != real_name:
                alias_path = lib_dir / alias
                if not alias_path.exists():
                    alias_path.symlink_to(real_name)
    return list(copied.values())


def make_relocatable(executable: Path, libraries: list[Path]) -> None:
    files = [executable, *libraries]
    for path in files:
        for dependency in dependencies(path):
            if dependency.startswith("/opt/homebrew/"):
                run(
                    "install_name_tool",
                    "-change",
                    dependency,
                    f"@rpath/{Path(dependency).name}",
                    path,
                )
        for old_rpath in rpaths(path):
            if old_rpath.startswith("/"):
                run("install_name_tool", "-delete_rpath", old_rpath, path)
        desired = "@executable_path/../lib" if path == executable else "@loader_path"
        if desired not in rpaths(path):
            run("install_name_tool", "-add_rpath", desired, path)
        if path != executable:
            run("install_name_tool", "-id", f"@rpath/{path.name}", path)
        run("codesign", "--force", "--sign", "-", path)


def copy_license(source: Path, output: Path) -> None:
    shutil.copy2(ROOT / "native/llama.cpp/LICENSE", output / "LICENSE.llama.cpp")
    openssl = Path("/opt/homebrew/opt/openssl@3")
    license_candidates = [openssl / "LICENSE.txt", openssl / "LICENSE"]
    license_path = next((path for path in license_candidates if path.exists()), None)
    if license_path:
        shutil.copy2(license_path, output / "LICENSE.OpenSSL")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(output: Path) -> None:
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and not path.is_symlink():
            relative = path.relative_to(output).as_posix()
            files[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        elif path.is_symlink():
            files[path.relative_to(output).as_posix()] = {
                "symlink": os.readlink(path)
            }
    manifest = {
        "schema_version": 1,
        "bundle": "BTL-3 Compact macOS arm64",
        "platform": "macOS arm64",
        "llama_cpp": {"build": 9596, "commit": "9fcaed763"},
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    if os.uname().machine != "arm64":
        parser.error("bundle must be built on macOS arm64")

    (output / "bin").mkdir(parents=True)
    (output / "lib").mkdir()
    (output / "libexec").mkdir()
    (output / "model").mkdir()
    (output / "share/btl3").mkdir(parents=True)
    executable = output / "libexec/llama-server"
    shutil.copy2(source / "llama-server", executable)
    libraries = copy_dependencies(source, source / "llama-server", output / "lib")
    make_relocatable(executable, libraries)

    shutil.copy2(ROOT / "launch/btl3-server", output / "bin/btl3-server")
    shutil.copy2(
        ROOT / "launch/btl3-ollama-bridge",
        output / "bin/btl3-ollama-bridge",
    )
    shutil.copy2(
        ROOT / "tools/ollama_bridge.py",
        output / "share/btl3/ollama_bridge.py",
    )
    for path in (output / "bin").iterdir():
        path.chmod(0o755)
    copy_license(source, output)
    write_manifest(output)
    print(output)


if __name__ == "__main__":
    main()
