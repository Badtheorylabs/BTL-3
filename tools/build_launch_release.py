"""Build a verified, self-describing BTL-3 Compact launch directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable


MODEL_NAME = "BTL-3-Compact-AVQ2.gguf"
MODEL_BYTES = 8_392_369_600
MODEL_SHA256 = "2ddf9527620a17a2a6739d184a7096c45712092e6589128792ec6254e94dc30c"
EXCLUDED_PARTS = {"node_modules", "__pycache__", ".git", ".pytest_cache", ".venv"}


class ReleaseError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(model: Path) -> None:
    if not model.is_file():
        raise ReleaseError(f"model is missing: {model}")
    if model.stat().st_size != MODEL_BYTES:
        raise ReleaseError(
            f"model size mismatch: expected {MODEL_BYTES}, got {model.stat().st_size}"
        )
    if sha256(model) != MODEL_SHA256:
        raise ReleaseError("model SHA-256 mismatch")


def read_bundle_manifest(bundle: Path) -> dict:
    manifest_path = bundle / "bundle-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseError(f"invalid bundle manifest in {bundle}: {error}") from error
    external = manifest.get("external_model", {})
    expected = {
        "filename": MODEL_NAME,
        "bytes": MODEL_BYTES,
        "sha256": MODEL_SHA256,
    }
    if any(external.get(key) != value for key, value in expected.items()):
        raise ReleaseError(f"runtime expects a different model: {bundle}")
    verify_bundle_files(bundle, manifest)
    return manifest


def verify_bundle_files(bundle: Path, manifest: dict) -> None:
    for relative, expected in manifest.get("files", {}).items():
        path = bundle / relative
        if "symlink" in expected:
            if not path.is_symlink() or os.readlink(path) != expected["symlink"]:
                raise ReleaseError(f"runtime symlink mismatch: {path}")
            continue
        if not path.is_file():
            raise ReleaseError(f"runtime file is missing: {path}")
        if path.stat().st_size != expected["bytes"]:
            raise ReleaseError(f"runtime file size mismatch: {path}")
        if sha256(path) != expected["sha256"]:
            raise ReleaseError(f"runtime file checksum mismatch: {path}")


def is_pending(manifest: dict) -> bool:
    status = str(manifest.get("status", "")).lower()
    return any(word in status for word in ("pending", "preview", "cross-compiled"))


def copy_tree_clean(source: Path, destination: Path) -> None:
    def ignored(_: str, names: list[str]) -> set[str]:
        return {
            name
            for name in names
            if name in EXCLUDED_PARTS or name == ".DS_Store"
        }

    shutil.copytree(source, destination, symlinks=True, ignore=ignored)


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def safe_name(path: Path) -> str:
    name = path.name
    if not name or name in {".", ".."}:
        raise ReleaseError(f"invalid release input name: {path}")
    return name


def add_runtimes(
    staging: Path, bundles: Iterable[Path], tier: str
) -> list[dict]:
    records = []
    for bundle in bundles:
        manifest = read_bundle_manifest(bundle)
        if tier == "supported" and is_pending(manifest):
            raise ReleaseError(
                f"conformance pending runtime cannot be marked supported: {bundle}"
            )
        name = safe_name(bundle)
        destination = staging / "runtimes" / tier / name
        copy_tree_clean(bundle, destination)
        records.append(
            {
                "name": name,
                "path": str(destination.relative_to(staging)),
                "bundle": manifest.get("bundle", name),
                "status": manifest.get(
                    "status",
                    "verified native runtime" if tier == "supported" else "preview",
                ),
            }
        )
    return records


def add_named_sources(
    staging: Path, sources: Iterable[Path], destination_root: str
) -> list[str]:
    copied = []
    for source in sources:
        if not source.exists():
            raise ReleaseError(f"release input is missing: {source}")
        destination = staging / destination_root / safe_name(source)
        if source.is_dir():
            copy_tree_clean(source, destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        copied.append(str(destination.relative_to(staging)))
    return copied


def write_checksums(staging: Path) -> None:
    output = staging / "SHA256SUMS"
    files = sorted(
        path
        for path in staging.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path != output
        and ".DS_Store" not in path.parts
    )
    output.write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(staging)}\n" for path in files
        )
    )


def install_staging(staging: Path, output: Path, replace: bool) -> Path:
    if output.exists() and not replace:
        raise ReleaseError(f"output already exists (use --replace): {output}")
    if output.exists():
        backup = output.with_name(f".{output.name}.previous")
        shutil.rmtree(backup, ignore_errors=True)
        output.rename(backup)
        try:
            staging.rename(output)
        except Exception:
            backup.rename(output)
            raise
        shutil.rmtree(backup)
    else:
        staging.rename(output)
    return output


def build_release(
    *,
    output: Path,
    model: Path,
    supported_runtimes: list[Path],
    preview_runtimes: list[Path],
    integration_sources: list[Path],
    license_files: list[Path],
    documentation_sources: list[Path] | None = None,
    evidence_sources: list[Path] | None = None,
    tool_sources: list[Path] | None = None,
    readme: Path | None = None,
    replace: bool = False,
) -> Path:
    output = output.resolve()
    model = model.resolve()
    verify_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        link_or_copy(model, staging / "model" / MODEL_NAME)
        supported = add_runtimes(staging, supported_runtimes, "supported")
        preview = add_runtimes(staging, preview_runtimes, "preview")
        integrations = add_named_sources(
            staging, integration_sources, "integrations"
        )
        licenses = add_named_sources(staging, license_files, "licenses")
        documentation = add_named_sources(
            staging, documentation_sources or [], "docs"
        )
        evidence = add_named_sources(staging, evidence_sources or [], "evidence")
        tools = add_named_sources(staging, tool_sources or [], "tools")
        if readme:
            shutil.copy2(readme, staging / "README.md")
        manifest = {
            "schema_version": 1,
            "release": "BTL-3 Compact",
            "checkpoint": "RL-0013",
            "architecture": "Qwen3.6-27B",
            "model": {
                "path": f"model/{MODEL_NAME}",
                "bytes": MODEL_BYTES,
                "sha256": MODEL_SHA256,
            },
            "runtimes": {"supported": supported, "preview": preview},
            "integrations": integrations,
            "documentation": documentation,
            "evidence": evidence,
            "tools": tools,
            "licenses": licenses,
            "stock_ollama_compatible": False,
            "stock_lm_studio_engine_compatible": False,
        }
        (staging / "RELEASE_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        write_checksums(staging)
        staging.chmod(0o755)
        return install_staging(staging, output, replace)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--supported-runtime", type=Path, action="append", default=[])
    parser.add_argument("--preview-runtime", type=Path, action="append", default=[])
    parser.add_argument("--integration", type=Path, action="append", default=[])
    parser.add_argument("--license", type=Path, action="append", default=[])
    parser.add_argument("--documentation", type=Path, action="append", default=[])
    parser.add_argument("--evidence", type=Path, action="append", default=[])
    parser.add_argument("--tool", type=Path, action="append", default=[])
    parser.add_argument("--readme", type=Path)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    try:
        path = build_release(
            output=args.output,
            model=args.model,
            supported_runtimes=args.supported_runtime,
            preview_runtimes=args.preview_runtime,
            integration_sources=args.integration,
            license_files=args.license,
            documentation_sources=args.documentation,
            evidence_sources=args.evidence,
            tool_sources=args.tool,
            readme=args.readme,
            replace=args.replace,
        )
        print(path)
    except ReleaseError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
