from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import build_launch_release as release


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_bundle(root: Path, name: str, status: str | None = None) -> Path:
    bundle = root / name
    runner = bundle / "bin/btl3-server"
    runner.parent.mkdir(parents=True)
    runner.write_bytes(b"runner")
    manifest = {
        "schema_version": 1,
        "bundle": name,
        "external_model": {
            "filename": release.MODEL_NAME,
            "bytes": 5,
            "sha256": digest(b"model"),
        },
        "files": {
            "bin/btl3-server": {
                "bytes": 6,
                "sha256": digest(b"runner"),
            }
        },
    }
    if status:
        manifest["status"] = status
    (bundle / "bundle-manifest.json").write_text(json.dumps(manifest))
    return bundle


def fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, Path]:
    monkeypatch.setattr(release, "MODEL_BYTES", 5)
    monkeypatch.setattr(release, "MODEL_SHA256", digest(b"model"))
    model = tmp_path / release.MODEL_NAME
    model.write_bytes(b"model")
    supported = write_bundle(tmp_path, "macos")
    preview = write_bundle(tmp_path, "spark", "conformance pending")
    integration = tmp_path / "integration"
    (integration / "src").mkdir(parents=True)
    (integration / "src/index.ts").write_text("export {};\n")
    (integration / "node_modules/pkg").mkdir(parents=True)
    (integration / "node_modules/pkg/index.js").write_text("ignored")
    return model, supported, preview, integration


def test_builds_verified_release_with_clear_support_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, supported, preview, integration = fixture(tmp_path, monkeypatch)
    output = release.build_release(
        output=tmp_path / "release",
        model=model,
        supported_runtimes=[supported],
        preview_runtimes=[preview],
        integration_sources=[integration],
        license_files=[],
    )
    manifest = json.loads((output / "RELEASE_MANIFEST.json").read_text())
    assert manifest["model"]["sha256"] == digest(b"model")
    assert manifest["runtimes"]["supported"][0]["name"] == "macos"
    assert manifest["runtimes"]["preview"][0]["name"] == "spark"
    assert not (output / "integrations/integration/node_modules").exists()
    assert (output / "model" / release.MODEL_NAME).stat().st_ino == model.stat().st_ino
    assert output.stat().st_mode & 0o777 == 0o755
    assert (output / "SHA256SUMS").is_file()


def test_rejects_wrong_model_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, supported, _, _ = fixture(tmp_path, monkeypatch)
    model.write_bytes(b"wrong")
    with pytest.raises(release.ReleaseError, match="model SHA-256"):
        release.build_release(
            output=tmp_path / "release",
            model=model,
            supported_runtimes=[supported],
            preview_runtimes=[],
            integration_sources=[],
            license_files=[],
        )


def test_rejects_preview_bundle_marked_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, _, preview, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(release.ReleaseError, match="conformance pending"):
        release.build_release(
            output=tmp_path / "release",
            model=model,
            supported_runtimes=[preview],
            preview_runtimes=[],
            integration_sources=[],
            license_files=[],
        )
