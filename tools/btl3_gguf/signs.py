"""Portable reproduction of Torch CPU seeded sign tensors."""

from __future__ import annotations


class SignError(ValueError):
    """Raised when a sign request cannot match the frozen package contract."""


class _MT19937:
    def __init__(self, seed: int) -> None:
        self.state = [seed]
        for index in range(1, 624):
            previous = self.state[-1]
            self.state.append(
                (1812433253 * (previous ^ (previous >> 30)) + index) & 0xFFFFFFFF
            )
        self.index = 624

    def _twist(self) -> None:
        for index in range(624):
            value = (self.state[index] & 0x80000000) | (
                self.state[(index + 1) % 624] & 0x7FFFFFFF
            )
            twisted = self.state[(index + 397) % 624] ^ (value >> 1)
            if value & 1:
                twisted ^= 0x9908B0DF
            self.state[index] = twisted
        self.index = 0

    def next_u32(self) -> int:
        if self.index == 624:
            self._twist()
        value = self.state[self.index]
        self.index += 1
        value ^= value >> 11
        value ^= (value << 7) & 0x9D2C5680
        value ^= (value << 15) & 0xEFC60000
        value ^= value >> 18
        return value & 0xFFFFFFFF


def seeded_signs(count: int, seed: int):
    """Return NumPy int8 -1/+1 signs matching torch.randint on CPU."""

    if type(count) is not int or count < 0:
        raise SignError("count must be a non-negative integer")
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
        raise SignError("seed must be an unsigned 32-bit integer")
    import numpy as np

    generator = _MT19937(seed)
    values = np.empty(count, dtype=np.int8)
    for index in range(count):
        values[index] = 1 if generator.next_u32() & 1 else -1
    return values
