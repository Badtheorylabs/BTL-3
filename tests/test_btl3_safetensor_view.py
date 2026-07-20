from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from btl3_gguf.safetensor_view import SafeTensorFile, bfloat16_to_float32


class SafeTensorViewTests(unittest.TestCase):
    def test_memmaps_tensor_payload(self) -> None:
        values = np.array([1.5, -2.0], dtype="<f4")
        header = json.dumps(
            {"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}},
            separators=(",", ":"),
        ).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.safetensors"
            path.write_bytes(struct.pack("<Q", len(header)) + header + values.tobytes())
            mapped = SafeTensorFile(path)["x"].array()
            np.testing.assert_array_equal(mapped, values)
            self.assertIsInstance(mapped, np.memmap)

    def test_decodes_bfloat16_exactly(self) -> None:
        words = np.array([0x3F80, 0xC000], dtype="<u2")
        np.testing.assert_array_equal(
            bfloat16_to_float32(words),
            np.array([1.0, -2.0], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
