"""Streaming custom GGUF exporter for the frozen BTL-3 Compact package."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
from typing import Callable

import numpy as np

from btl3_gguf.metadata import add_metadata
from btl3_gguf.qwen35 import (
    convert_small_state,
    gguf_name,
    projection_name,
)
from btl3_gguf.safetensor_view import SafeTensorFile, TensorView
from btl3_gguf.signs import seeded_signs


EXPECTED_MANIFEST = "a2b763323eed76d8f78fe5cbdf5a2349323b2c3d87dddc037714569946961116"
LORA = re.compile(
    r"^base_model\.model\.model\.language_model\.layers\.(\d+)\.(.+)"
    r"\.lora_([AB])\.weight$"
)
ITEM_SIZES = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1, "I32": 4}


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    source: str
    component: str
    load: Callable[[], np.ndarray]

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * ITEM_SIZES[self.dtype]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_gguf(llama_cpp: Path):
    location = str(llama_cpp / "gguf-py")
    if location not in sys.path:
        sys.path.insert(0, location)
    return importlib.import_module("gguf")


def validate_source(source: Path) -> dict:
    manifest_path = source / "package.json"
    if sha256(manifest_path) != EXPECTED_MANIFEST:
        raise ValueError("source package manifest does not match frozen BTL-3 Compact")
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["files"]:
        path = source / entry["path"]
        if not path.is_file() or path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"missing or size-mismatched source file: {entry['path']}")
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"source checksum mismatch: {entry['path']}")
    return manifest


def _view_spec(
    name: str,
    view: TensorView,
    *,
    dtype: str | None = None,
    shape: tuple[int, ...] | None = None,
    component: str,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> TensorSpec:
    output_dtype = dtype or view.dtype
    output_shape = shape or view.shape

    def load() -> np.ndarray:
        value = view.array()
        if transform is not None:
            value = transform(value)
        return value.reshape(output_shape)

    return TensorSpec(
        name=name,
        shape=output_shape,
        dtype=output_dtype,
        source=f"{view.path}:{view.name}",
        component=component,
        load=load,
    )


def _suffix(base: str, value: str) -> str:
    if not base.endswith(".weight"):
        raise ValueError(f"projection base name lacks .weight: {base}")
    return base.removesuffix(".weight") + f".{value}"


def add_avq_specs(
    specs: list[TensorSpec],
    gguf,
    root: Path,
    layer: int,
    module: str,
    item: dict,
) -> None:
    source = SafeTensorFile(root / item["tensor_file"])
    prefix = item["prefix"]
    rows, columns = (int(value) for value in item["shape"])
    base = projection_name(gguf, layer, module)
    specs.extend(
        (
            _view_spec(
                _suffix(base, "avq2_codes"),
                source[f"{prefix}.codes"],
                dtype="I8",
                component="avq2",
                transform=lambda value: value.view(np.int8),
            ),
            _view_spec(
                _suffix(base, "avq2_affine_weight"),
                source[f"{prefix}.affine_weight"],
                dtype="F32",
                component="avq2",
                transform=lambda value: value.astype(np.float32),
            ),
            _view_spec(
                _suffix(base, "avq2_affine_bias"),
                source[f"{prefix}.affine_bias"],
                dtype="F32",
                component="avq2",
                transform=lambda value: value.astype(np.float32),
            ),
            TensorSpec(
                _suffix(base, "avq2_input_signs"),
                (columns,),
                "I8",
                f"torch-seed:{item['input_seed']}",
                "avq2",
                lambda: seeded_signs(columns, int(item["input_seed"])),
            ),
            TensorSpec(
                _suffix(base, "avq2_output_signs"),
                (rows,),
                "I8",
                f"torch-seed:{item['output_seed']}",
                "avq2",
                lambda: seeded_signs(rows, int(item["output_seed"])),
            ),
        )
    )


def add_int4_specs(
    specs: list[TensorSpec],
    gguf,
    root: Path,
    layer: int,
    module: str,
    item: dict,
    *,
    component: str = "affine-int4",
) -> None:
    source = SafeTensorFile(root / item["tensor_file"])
    prefix = item.get("prefix")

    def tensor(suffix: str) -> TensorView:
        candidates = [f"{prefix}.{suffix}"] if prefix else []
        candidates.extend((suffix, f"packed_{suffix}"))
        for name in candidates:
            if name in source.tensors:
                return source[name]
        raise KeyError(
            f"missing INT4 {suffix} tensor in {source.path}; "
            f"found {sorted(source.tensors)}"
        )

    rows, columns = (int(value) for value in item["shape"])
    base = projection_name(gguf, layer, module)
    specs.extend(
        (
            _view_spec(
                _suffix(base, "btl_int4_codes"),
                tensor("codes"),
                dtype="I8",
                shape=(rows, columns // 2),
                component=component,
                transform=lambda value: value.view(np.int8),
            ),
            _view_spec(
                _suffix(base, "btl_int4_scales"),
                tensor("scales"),
                shape=(rows, columns // 128),
                component=component,
            ),
            _view_spec(
                _suffix(base, "btl_int4_zeros"),
                tensor("zeros"),
                dtype="I8",
                shape=(rows, columns // 128),
                component=component,
                transform=lambda value: value.view(np.int8),
            ),
        )
    )


def add_small_state(specs: list[TensorSpec], gguf, source: Path, layer: int | None) -> None:
    state = SafeTensorFile(source / "small_state.safetensors")
    prefix = None if layer is None else f"layers.{layer}."
    for name, view in state.tensors.items():
        if prefix is not None and not name.startswith(prefix):
            continue
        target = gguf_name(gguf, name)
        is_conv = ".conv1d." in name
        force_f32 = (
            name.endswith("norm.weight")
            or name.endswith(".A_log")
            or name.endswith(".dt_bias")
            or is_conv
        )
        shape = tuple(value for value in view.shape if value != 1) if is_conv else view.shape
        specs.append(
            TensorSpec(
                target,
                shape,
                "F32" if force_f32 else "BF16",
                f"{view.path}:{name}",
                "small-state",
                lambda view=view: convert_small_state(view)[0],
            )
        )


def add_decoder(specs: list[TensorSpec], gguf, source: Path, layer: int | None) -> None:
    decoder = source / "decoder"
    manifest = json.loads((decoder / "manifest.json").read_text())
    islands = {
        (int(item["layer"]), item["module"]): item
        for item in manifest["fp16_islands"]
    }
    demotions = {
        (int(item["layer"]), item["module"]): item
        for item in json.loads((source / "package.json").read_text())["contract"]["demotions"]
    }
    layers = [layer] if layer is not None else range(64)
    for index in layers:
        layer_meta = json.loads((decoder / f"layers/{index:02d}.json").read_text())
        tensor_root = decoder / "layers"
        for module, item in sorted(layer_meta["matrices"].items()):
            identity = (index, module)
            if identity in islands or identity in demotions:
                continue
            item = {**item, "tensor_file": layer_meta["tensor_file"]}
            if item["kind"] == "vector2":
                add_avq_specs(specs, gguf, tensor_root, index, module, item)
            elif item["kind"] == "int4":
                add_int4_specs(specs, gguf, tensor_root, index, module, item)
            else:
                raise ValueError(f"unsupported decoder representation: {item['kind']}")
    for (index, module), item in sorted(islands.items()):
        if layer is not None and index != layer or (index, module) in demotions:
            continue
        view = SafeTensorFile(decoder / item["tensor_file"])["weight"]
        specs.append(
            _view_spec(
                projection_name(gguf, index, module),
                view,
                dtype="BF16",
                component="bf16-island",
            )
        )
    for (index, module), contract in sorted(demotions.items()):
        if layer is not None and index != layer:
            continue
        item = json.loads(
            (source / "demotions/tensors" / contract["artifact"]["name"]).with_suffix(".json").read_text()
        )
        item["tensor_file"] = item["tensor_file"]
        add_int4_specs(
            specs,
            gguf,
            source / "demotions/tensors",
            index,
            module,
            item,
            component="affine-int4-demotion",
        )


def add_lora(specs: list[TensorSpec], gguf, source: Path, layer: int | None) -> None:
    config = json.loads((source / "behavior_adapter/adapter_config.json").read_text())
    if int(config["r"]) != 8 or float(config["lora_alpha"]) / int(config["r"]) != 2.0:
        raise ValueError("native LoRA contract requires rank 8 and scale 2.0")
    state = SafeTensorFile(source / "behavior_adapter/adapter_model.safetensors")
    for name, view in state.tensors.items():
        match = LORA.fullmatch(name)
        if not match:
            raise ValueError(f"unexpected behavior-adapter tensor: {name}")
        index = int(match.group(1))
        if layer is not None and index != layer:
            continue
        base = projection_name(gguf, index, match.group(2))
        suffix = "btl_lora_a" if match.group(3) == "A" else "btl_lora_b"
        specs.append(_view_spec(_suffix(base, suffix), view, component="behavior-lora"))


def add_vocabulary(specs: list[TensorSpec], source: Path) -> None:
    for label, base, seed_key in (
        ("embedding", "token_embd", "input_seed"),
        ("lm_head", "output", "input_seed"),
    ):
        meta = json.loads((source / f"vocabulary/{label}.json").read_text())
        state = SafeTensorFile(source / f"vocabulary/{label}.safetensors")
        rows, columns = (int(value) for value in meta["shape"])
        specs.extend(
            (
                _view_spec(f"{base}.avq2_codes", state["codes"], dtype="I8", component="vocabulary", transform=lambda value: value.view(np.int8)),
                _view_spec(f"{base}.avq2_affine_weight", state["affine_weight"], dtype="F32", component="vocabulary", transform=lambda value: value.astype(np.float32)),
                _view_spec(f"{base}.avq2_affine_bias", state["affine_bias"], dtype="F32", component="vocabulary", transform=lambda value: value.astype(np.float32)),
                TensorSpec(f"{base}.avq2_input_signs", (columns,), "I8", f"torch-seed:{meta[seed_key]}", "vocabulary", lambda columns=columns, seed=int(meta[seed_key]): seeded_signs(columns, seed)),
            )
        )
        if label == "embedding":
            specs.extend(
                (
                    _view_spec(f"{base}.btl_rescued_row_ids", state["rescued_row_ids"], component="vocabulary"),
                    _view_spec(f"{base}.btl_rescued_upper_six", state["rescued_upper_six"], dtype="I8", component="vocabulary", transform=lambda value: value.view(np.int8)),
                    _view_spec(f"{base}.btl_rescued_scales", state["rescued_scales"], component="vocabulary"),
                )
            )
    residual = SafeTensorFile(source / "vocabulary/lm_head_residual.safetensors")
    specs.append(_view_spec("output.btl_residual_left", residual["left"], component="vocabulary"))
    specs.append(_view_spec("output.btl_residual_right", residual["right"], component="vocabulary"))


def build_specs(source: Path, gguf, layer: int | None) -> list[TensorSpec]:
    specs: list[TensorSpec] = []
    add_small_state(specs, gguf, source, layer)
    add_decoder(specs, gguf, source, layer)
    add_lora(specs, gguf, source, layer)
    if layer is None:
        add_vocabulary(specs, source)
    names = [spec.name for spec in specs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate GGUF tensors: {duplicates}")
    return specs


def runtime_gaps(llama_cpp: Path) -> list[str]:
    implementation = "\n".join(
        path.read_text()
        for path in sorted((llama_cpp / "src").glob("btl3-*.cpp"))
    )
    gaps = []
    for function in ("build_vocab_get_rows", "build_vocab_head"):
        if f"ggml_tensor * {function}(" not in implementation:
            gaps.append(
                f"native runtime declares/calls btl3::{function} but has no definition"
            )
    return gaps


def report_for(specs: list[TensorSpec], manifest: dict, llama_cpp: Path, mode: str) -> dict:
    by_component: dict[str, dict[str, int]] = {}
    for spec in specs:
        entry = by_component.setdefault(spec.component, {"tensors": 0, "bytes": 0})
        entry["tensors"] += 1
        entry["bytes"] += spec.nbytes
    return {
        "schema_version": 1,
        "model": "BTL-3 Compact",
        "mode": mode,
        "source_manifest_sha256": EXPECTED_MANIFEST,
        "source_checksums_verified": len(manifest["files"]),
        "source_weight_bytes": manifest["weight_bytes"],
        "tensor_count": len(specs),
        "projected_tensor_bytes": sum(spec.nbytes for spec in specs),
        "components": by_component,
        "exporter_unsupported": [],
        "native_runtime_gaps": runtime_gaps(llama_cpp),
    }


def write_gguf(output: Path, source: Path, gguf, specs: list[TensorSpec], full: bool) -> None:
    writer = gguf.GGUFWriter(output, "qwen35")
    add_metadata(writer, gguf, source, include_tokenizer=full)
    dtype_map = {
        "BF16": gguf.GGMLQuantizationType.BF16,
    }
    for spec in specs:
        value = spec.load()
        if tuple(value.shape) != spec.shape or value.nbytes != spec.nbytes:
            raise ValueError(f"materialized tensor mismatch: {spec.name}")
        raw_dtype = dtype_map.get(spec.dtype)
        writer.add_tensor(
            spec.name,
            value,
            raw_shape=spec.shape if raw_dtype is not None else None,
            raw_dtype=raw_dtype,
        )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()


def verify_gguf(output: Path, gguf, specs: list[TensorSpec]) -> dict:
    reader = gguf.GGUFReader(output)
    actual = {tensor.name: tensor for tensor in reader.tensors}
    expected = {spec.name: spec for spec in specs}
    tensor_types = {
        name: getattr(gguf.GGMLQuantizationType, name)
        for name in ITEM_SIZES
    }
    if set(actual) != set(expected):
        raise ValueError("GGUF tensor-name set does not match the export plan")
    for name, spec in expected.items():
        tensor = actual[name]
        if tuple(reversed(tensor.shape.tolist())) != spec.shape:
            raise ValueError(f"GGUF shape mismatch: {name}")
        if tensor.tensor_type != tensor_types[spec.dtype]:
            raise ValueError(f"GGUF dtype mismatch: {name}")
        expected_value = np.ascontiguousarray(spec.load())
        expected_digest = hashlib.sha256(
            memoryview(expected_value).cast("B")
        ).digest()
        actual_digest = hashlib.sha256(
            memoryview(np.ascontiguousarray(tensor.data)).cast("B")
        ).digest()
        if expected_value.nbytes != tensor.data.nbytes or expected_digest != actual_digest:
            raise ValueError(f"GGUF payload mismatch: {name}")
    return {
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256(output),
        "verified_tensor_count": len(actual),
        "verified_payload_count": len(actual),
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--llama-cpp", type=Path, default=root / "native/llama.cpp")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--conformance-layer", type=int)
    args = parser.parse_args()
    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run is used")
    if args.conformance_layer is not None and not 0 <= args.conformance_layer < 64:
        parser.error("--conformance-layer must be between 0 and 63")
    return args


def main() -> None:
    args = parse_args()
    gguf = load_gguf(args.llama_cpp)
    manifest = validate_source(args.source)
    specs = build_specs(args.source, gguf, args.conformance_layer)
    mode = (
        f"conformance-layer-{args.conformance_layer}"
        if args.conformance_layer is not None
        else "full"
    )
    report = report_for(specs, manifest, args.llama_cpp, mode)
    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_gguf(args.output, args.source, gguf, specs, args.conformance_layer is None)
        report.update(verify_gguf(args.output, gguf, specs))
    rendered = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
