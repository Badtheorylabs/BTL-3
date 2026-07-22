#!/usr/bin/env python3
"""Build the canonical academic BTL-3 compression paper with Tectonic."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf"
TARGET = OUT / "btl-3-behavior-before-perplexity.pdf"


def main() -> None:
    tectonic = shutil.which("tectonic")
    if tectonic is None:
        raise RuntimeError("tectonic is required to render PAPER.tex")
    OUT.mkdir(parents=True, exist_ok=True)
    command = [
        tectonic,
        "--outdir",
        str(OUT),
        str(ROOT / "PAPER.tex"),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    generated = OUT / "PAPER.pdf"
    if not generated.is_file():
        raise RuntimeError("Tectonic did not produce PAPER.pdf")
    generated.replace(TARGET)
    print(f"wrote {TARGET}")


if __name__ == "__main__":
    main()
