from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import install_consumer_bundle as installer


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    binary = runtime / "libexec/llama-server"
    binary.parent.mkdir()
    binary.write_bytes(b"runner")
    library = runtime / "lib/libggml.so.0.13.1"
    library.parent.mkdir()
    library.write_bytes(b"library")
    (runtime / "lib/libggml.so.0").symlink_to(library.name)
    model = tmp_path / installer.MODEL_NAME
    model.write_bytes(b"model")
    monkeypatch.setattr(installer, "MODEL_BYTES", 5)
    monkeypatch.setattr(installer, "MODEL_SHA256", digest(b"model"))
    manifest = {
        "target": "linux-x86_64",
        "external_model": {
            "filename": installer.MODEL_NAME,
            "bytes": 5,
            "sha256": digest(b"model"),
        },
        "files": {
            "libexec/llama-server": {
                "bytes": 6,
                "sha256": digest(b"runner"),
            },
            "lib/libggml.so.0.13.1": {
                "bytes": 7,
                "sha256": digest(b"library"),
            },
            "lib/libggml.so.0": {"symlink": "libggml.so.0.13.1"},
        },
    }
    (runtime / "bundle-manifest.json").write_text(json.dumps(manifest))
    return runtime, model


def test_installs_verified_runtime_and_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, model = fixture(tmp_path, monkeypatch)
    output = installer.install(runtime, model, tmp_path / "installed", False)
    assert (output / "libexec/llama-server").read_bytes() == b"runner"
    assert (output / "lib/libggml.so.0").is_symlink()
    assert (output / "model" / installer.MODEL_NAME).read_bytes() == b"model"
    receipt = json.loads((output / "install-receipt.json").read_text())
    assert receipt["runtime_target"] == "linux-x86_64"


def test_rejects_modified_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, model = fixture(tmp_path, monkeypatch)
    (runtime / "libexec/llama-server").write_bytes(b"tampered")
    with pytest.raises(installer.InstallError, match="size mismatch"):
        installer.install(runtime, model, tmp_path / "installed", False)


def test_requires_replace_for_existing_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, model = fixture(tmp_path, monkeypatch)
    output = tmp_path / "installed"
    output.mkdir()
    with pytest.raises(installer.InstallError, match="already exists"):
        installer.install(runtime, model, output, False)
