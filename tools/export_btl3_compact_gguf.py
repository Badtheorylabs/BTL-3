#!/usr/bin/env python3
"""Command-line entrypoint for the BTL-3 Compact custom GGUF exporter."""

from __future__ import annotations

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent))

from btl3_gguf.exporter import main  # noqa: E402


if __name__ == "__main__":
    main()
