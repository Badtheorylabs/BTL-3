"""Read-only, mmap-backed safetensors access without importing Torch."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct

import numpy as np


DTYPES = {
    "BF16": np.dtype("<u2"),
    "F16": np.dtype("<f2"),
    "F32": np.dtype("<f4"),
    "I8": np.dtype("i1"),
    "I32": np.dtype("<i4"),
    "U8": np.dtype("u1"),
}


@dataclass(frozen=True)
class TensorView:
    path: Path
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int

    def array(self) -> np.ndarray:
        dtype = DTYPES.get(self.dtype)
        if dtype is None:
            raise ValueError(f"unsupported safetensors dtype: {self.dtype}")
        if self.nbytes == 0:
            return np.empty(self.shape, dtype=dtype)
        return np.memmap(
            self.path,
            mode="r",
            dtype=dtype,
            offset=self.offset,
            shape=self.shape,
            order="C",
        )


class SafeTensorFile:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        with self.path.open("rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                raise ValueError(f"truncated safetensors header: {self.path}")
            header_bytes = struct.unpack("<Q", raw)[0]
            header = json.loads(handle.read(header_bytes))
        self.data_offset = 8 + header_bytes
        self.tensors: dict[str, TensorView] = {}
        file_bytes = self.path.stat().st_size
        for name, item in header.items():
            if name == "__metadata__":
                continue
            dtype = str(item["dtype"])
            shape = tuple(int(value) for value in item["shape"])
            start, end = (int(value) for value in item["data_offsets"])
            expected = int(np.prod(shape, dtype=np.int64)) * DTYPES[dtype].itemsize
            if end - start != expected:
                raise ValueError(f"byte count mismatch for {self.path}:{name}")
            if self.data_offset + end > file_bytes:
                raise ValueError(f"tensor exceeds file boundary: {self.path}:{name}")
            self.tensors[name] = TensorView(
                path=self.path,
                name=name,
                dtype=dtype,
                shape=shape,
                offset=self.data_offset + start,
                nbytes=end - start,
            )

    def __getitem__(self, name: str) -> TensorView:
        try:
            return self.tensors[name]
        except KeyError as error:
            raise KeyError(f"missing tensor {self.path}:{name}") from error


def bfloat16_to_float32(array: np.ndarray) -> np.ndarray:
    if array.dtype != np.dtype("<u2"):
        raise ValueError("BF16 storage must be represented as little-endian uint16")
    words = array.astype(np.uint32) << np.uint32(16)
    return words.view(np.float32)
