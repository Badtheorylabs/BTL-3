"""Install a verified BTL-3 consumer runtime and its external model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile


MODEL_NAME = "BTL-3-Compact-AVQ2.gguf"
MODEL_BYTES = 8_392_369_600
MODEL_SHA256 = "2ddf9527620a17a2a6739d184a7096c45712092e6589128792ec6254e94dc30c"


class InstallError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_prefix() -> Path:
    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA")
        if not root:
            raise InstallError("LOCALAPPDATA is unavailable; pass --prefix")
        return Path(root) / "BTL3"
    root = os.environ.get("XDG_DATA_HOME")
    return Path(root) / "btl3" if root else Path.home() / ".local/share/btl3"


def read_manifest(runtime: Path) -> dict:
    path = runtime / "bundle-manifest.json"
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise InstallError(f"invalid runtime manifest: {error}") from error
    external = manifest.get("external_model", {})
    expected = {
        "filename": MODEL_NAME,
        "bytes": MODEL_BYTES,
        "sha256": MODEL_SHA256,
    }
    if any(external.get(key) != value for key, value in expected.items()):
        raise InstallError("runtime expects a different BTL-3 model")
    return manifest


def verify_runtime(runtime: Path, manifest: dict) -> None:
    for name, expected in manifest.get("files", {}).items():
        path = runtime / name
        if "symlink" in expected:
            if not path.is_symlink() or os.readlink(path) != expected["symlink"]:
                raise InstallError(f"runtime symlink mismatch: {name}")
            continue
        if not path.is_file():
            raise InstallError(f"runtime file is missing: {name}")
        if path.stat().st_size != expected["bytes"]:
            raise InstallError(f"runtime file size mismatch: {name}")
        if sha256(path) != expected["sha256"]:
            raise InstallError(f"runtime file checksum mismatch: {name}")


def verify_model(model: Path) -> None:
    if not model.is_file():
        raise InstallError(f"model is missing: {model}")
    if model.stat().st_size != MODEL_BYTES:
        raise InstallError(f"model size mismatch: {model.stat().st_size}")
    if sha256(model) != MODEL_SHA256:
        raise InstallError("model SHA-256 mismatch")


def install(runtime: Path, model: Path, prefix: Path, replace: bool) -> Path:
    runtime, model, prefix = runtime.resolve(), model.resolve(), prefix.resolve()
    manifest = read_manifest(runtime)
    verify_runtime(runtime, manifest)
    verify_model(model)
    if prefix.exists() and not replace:
        raise InstallError(f"install already exists (use --replace): {prefix}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{prefix.name}-", dir=prefix.parent))
    try:
        shutil.copytree(runtime, staging, dirs_exist_ok=True, symlinks=True)
        destination = staging / "model" / MODEL_NAME
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model, destination)
        receipt = {
            "schema_version": 1,
            "installed_by": "BTL-3 consumer installer",
            "platform": platform.platform(),
            "runtime_target": manifest.get("target") or manifest.get("platform"),
            "model": {
                "path": f"model/{MODEL_NAME}",
                "bytes": MODEL_BYTES,
                "sha256": MODEL_SHA256,
            },
        }
        (staging / "install-receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        if prefix.exists():
            backup = prefix.with_name(f".{prefix.name}.previous")
            shutil.rmtree(backup, ignore_errors=True)
            prefix.rename(backup)
            try:
                staging.rename(prefix)
            except Exception:
                backup.rename(prefix)
                raise
            shutil.rmtree(backup)
        else:
            staging.rename(prefix)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return prefix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, default=None)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    try:
        print(install(
            args.runtime,
            args.model,
            args.prefix or default_prefix(),
            args.replace,
        ))
    except InstallError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
