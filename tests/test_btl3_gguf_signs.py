from __future__ import annotations

import hashlib
import unittest

import numpy as np

from btl3_gguf.signs import seeded_signs


class SignTests(unittest.TestCase):
    def test_matches_real_torch_seed_digest(self) -> None:
        signs = seeded_signs(17_408, 40_521_442)
        packed = np.packbits(signs == 1, bitorder="little")
        self.assertEqual(
            hashlib.sha256(packed.tobytes()).hexdigest(),
            "5a80ab3cab9e3b38b1f85108937b49c7d97afb06a19d006e73cce78ac64d5e3a",
        )

    def test_matches_short_torch_vector(self) -> None:
        expected = [-1, -1, -1, -1, 1, -1, -1, 1]
        self.assertEqual(seeded_signs(8, 40_521_435).tolist(), expected)


if __name__ == "__main__":
    unittest.main()
