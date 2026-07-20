from __future__ import annotations

import unittest

import numpy as np

from btl3_gguf.qwen35 import reorder_v_heads


class Qwen35TransformTests(unittest.TestCase):
    def test_reorders_grouped_heads_to_tiled_heads(self) -> None:
        grouped = np.arange(2 * 3 * 2).reshape(2 * 3 * 2)
        actual = reorder_v_heads(
            grouped,
            0,
            key_heads=2,
            values_per_key=3,
            head_dim=2,
        )
        expected = grouped.reshape(2, 3, 2).transpose(1, 0, 2).reshape(-1)
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
