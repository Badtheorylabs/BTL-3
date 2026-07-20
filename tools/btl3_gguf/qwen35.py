"""Qwen3.5/3.6 GGUF naming and small-state conversion semantics."""

from __future__ import annotations

import re

import numpy as np

from btl3_gguf.safetensor_view import TensorView, bfloat16_to_float32


LAYER = re.compile(r"^layers\.(\d+)\.(.+)$")


def hf_name(package_name: str) -> str:
    if package_name == "norm.weight":
        return "model.norm.weight"
    match = LAYER.fullmatch(package_name)
    if not match:
        raise ValueError(f"unsupported small-state name: {package_name}")
    return f"model.layers.{match.group(1)}.{match.group(2)}"


def gguf_name(gguf, package_name: str, block_count: int = 64) -> str:
    mapping = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN35, block_count)
    source = hf_name(package_name)
    if source.endswith(".dt_bias"):
        source = source.removesuffix(".dt_bias") + ".dt_proj.bias"
    target = mapping.get_name(source, try_suffixes=(".weight", ".bias"))
    if target is None:
        raise ValueError(f"Qwen3.5 converter cannot map {source!r}")
    return target


def projection_name(gguf, layer: int, module: str, block_count: int = 64) -> str:
    mapping = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN35, block_count)
    source = f"model.layers.{layer}.{module}.weight"
    target = mapping.get_name(source, try_suffixes=(".weight", ".bias"))
    if target is None:
        raise ValueError(f"Qwen3.5 converter cannot map {source!r}")
    return target


def reorder_v_heads(
    array: np.ndarray,
    axis: int,
    *,
    key_heads: int = 16,
    values_per_key: int = 3,
    head_dim: int = 128,
) -> np.ndarray:
    axis %= array.ndim
    shape = list(array.shape)
    expected = key_heads * values_per_key * head_dim
    if shape[axis] != expected:
        raise ValueError(f"reorder axis is {shape[axis]}, expected {expected}")
    expanded = shape[:axis] + [key_heads, values_per_key, head_dim] + shape[axis + 1 :]
    value = array.reshape(expanded)
    axes = list(range(value.ndim))
    axes[axis], axes[axis + 1] = axes[axis + 1], axes[axis]
    return value.transpose(axes).copy().reshape(shape)


def convert_small_state(view: TensorView) -> tuple[np.ndarray, object]:
    """Apply the official Qwen3.5 converter's tensor transforms."""

    name = hf_name(view.name)
    source = view.array()
    raw_dtype = None
    if view.dtype == "BF16":
        if any(
            marker in name
            for marker in (".A_log", ".dt_bias", ".conv1d")
        ) or (name.endswith("norm.weight") and "linear_attn.norm.weight" not in name):
            value = bfloat16_to_float32(source)
        elif "linear_attn.norm.weight" in name:
            value = bfloat16_to_float32(source)
        else:
            value = source
            raw_dtype = "BF16"
    else:
        value = source

    if name.endswith(".A_log"):
        value = -np.exp(value, dtype=np.float32)
    if name.endswith(".A_log") or name.endswith(".dt_bias"):
        value = reorder_v_heads(value, 0, head_dim=1)
    elif ".in_proj_a." in name or ".in_proj_b." in name:
        value = reorder_v_heads(value, 0, head_dim=1)
    elif ".conv1d" in name:
        value = value.squeeze()
        qk_channels = 2 * 16 * 128
        value = np.concatenate(
            (value[:qk_channels], reorder_v_heads(value[qk_channels:], 0)),
            axis=0,
        )
    if name.endswith("norm.weight") and "linear_attn.norm.weight" not in name:
        value = value + np.float32(1.0)
    return value, raw_dtype
