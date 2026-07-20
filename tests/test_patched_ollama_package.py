from __future__ import annotations

import json
import stat
import struct
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.build_patched_ollama import (
    PackageError,
    REQUIRED_RUNNER_FLAGS,
    build_package,
    sha256,
)


def executable(path: Path, content: str = "#!/bin/sh\nexit 0\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def pe_x64(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = bytearray(512)
    payload[:2] = b"MZ"
    struct.pack_into("<I", payload, 0x3C, 0x80)
    payload[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<H", payload, 0x84, 0x8664)
    path.write_bytes(payload)


def windows_contract(root: Path) -> None:
    runner = root / "llama-server.exe"
    (root / "runner-cli-contract.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "platform": "windows-amd64-cuda13",
                "runner_sha256": sha256(runner),
                "required_flags": list(REQUIRED_RUNNER_FLAGS),
            }
        )
    )


class PatchedOllamaPackageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.dist = self.root / "ollama-dist"
        self.runner = self.root / "runner"
        executable(self.dist / "bin/ollama")
        executable(self.dist / "lib/ollama/llama-server", "# stock\n")
        (self.dist / "lib/ollama/stock.txt").write_text("stock")
        executable(
            self.runner / "llama-server",
            "#!/bin/sh\n"
            "if [ \"${1:-}\" = --help ]; then\n"
            "  echo '--model --port --host --no-webui --offline -c -np'\n"
            "fi\n",
        )
        (self.runner / "libggml-base.so").write_text("base")
        (self.runner / "cuda_v12").mkdir()
        (self.runner / "cuda_v12/libggml-cuda.so").write_text("cuda")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_builds_direct_runner_distribution(self) -> None:
        output = self.root / "out"
        build_package(
            ollama_dist=self.dist,
            runner_payload=self.runner,
            output=output,
            platform="linux-amd64-cuda12",
        )
        self.assertIn(
            "--no-webui", (output / "lib/ollama/llama-server").read_text()
        )
        self.assertFalse((output / "lib/ollama/stock.txt").exists())
        self.assertTrue((output / "lib/ollama/cuda_v12/libggml-cuda.so").is_file())
        self.assertTrue((output / "bin/ollama").is_file())
        self.assertTrue((output / "bin/btl3-ollama").stat().st_mode & stat.S_IXUSR)
        manifest = json.loads(
            (output / "share/btl3/patched-ollama-manifest.json").read_text()
        )
        self.assertEqual(manifest["integration"], "direct-subprocess")
        self.assertEqual(manifest["stock_ollama_compatible"], False)
        self.assertEqual(manifest["runner"]["cuda_backend"], "cuda_v12")

    def test_rejects_payload_without_cuda_backend(self) -> None:
        (self.runner / "cuda_v12/libggml-cuda.so").unlink()
        with self.assertRaisesRegex(PackageError, "CUDA backend"):
            build_package(
                ollama_dist=self.dist,
                runner_payload=self.runner,
                output=self.root / "out",
                platform="linux-amd64-cuda12",
            )

    def test_refuses_to_overwrite_output(self) -> None:
        output = self.root / "out"
        output.mkdir()
        with self.assertRaisesRegex(PackageError, "already exists"):
            build_package(
                ollama_dist=self.dist,
                runner_payload=self.runner,
                output=output,
                platform="linux-amd64-cuda12",
            )

    def test_rejects_runner_without_ollama_cli_contract(self) -> None:
        executable(self.runner / "llama-server", "#!/bin/sh\nexit 0\n")
        with self.assertRaisesRegex(PackageError, "runner CLI contract"):
            build_package(
                ollama_dist=self.dist,
                runner_payload=self.runner,
                output=self.root / "out",
                platform="linux-amd64-cuda12",
            )

    def test_builds_windows_x64_cuda13_distribution(self) -> None:
        windows_dist = self.root / "ollama-windows"
        windows_runner = self.root / "runner-windows"
        pe_x64(windows_dist / "ollama.exe")
        pe_x64(windows_dist / "lib/ollama/llama-server.exe")
        (windows_dist / "lib/ollama/stock.dll").write_text("stock")
        pe_x64(windows_runner / "llama-server.exe")
        pe_x64(windows_runner / "ggml-base.dll")
        pe_x64(windows_runner / "cuda_v13/ggml-cuda.dll")
        windows_contract(windows_runner)

        output = self.root / "windows-out"
        build_package(
            ollama_dist=windows_dist,
            runner_payload=windows_runner,
            output=output,
            platform="windows-amd64-cuda13",
        )

        self.assertTrue((output / "ollama.exe").is_file())
        self.assertTrue((output / "btl3-ollama.cmd").is_file())
        wrapper = (output / "btl3-ollama.cmd").read_text()
        self.assertIn("%~dp0ollama.exe", wrapper)
        self.assertIn("not stock Ollama", wrapper)
        self.assertTrue((output / "lib/ollama/llama-server.exe").is_file())
        self.assertTrue((output / "lib/ollama/cuda_v13/ggml-cuda.dll").is_file())
        self.assertFalse((output / "lib/ollama/stock.dll").exists())
        manifest = json.loads(
            (output / "share/btl3/patched-ollama-manifest.json").read_text()
        )
        self.assertEqual(manifest["platform"], "windows-amd64-cuda13")
        self.assertEqual(
            manifest["runner"]["path"], "lib/ollama/llama-server.exe"
        )

    def test_windows_rejects_non_pe_cuda_backend(self) -> None:
        windows_dist = self.root / "ollama-windows"
        windows_runner = self.root / "runner-windows"
        pe_x64(windows_dist / "ollama.exe")
        pe_x64(windows_runner / "llama-server.exe")
        (windows_runner / "cuda_v13").mkdir(parents=True)
        (windows_runner / "cuda_v13/ggml-cuda.dll").write_text("not PE")
        windows_contract(windows_runner)

        with self.assertRaisesRegex(PackageError, "PE x64"):
            build_package(
                ollama_dist=windows_dist,
                runner_payload=windows_runner,
                output=self.root / "windows-out",
                platform="windows-amd64-cuda13",
            )

    def test_windows_rejects_contract_for_different_runner(self) -> None:
        windows_dist = self.root / "ollama-windows"
        windows_runner = self.root / "runner-windows"
        pe_x64(windows_dist / "ollama.exe")
        pe_x64(windows_runner / "llama-server.exe")
        pe_x64(windows_runner / "cuda_v13/ggml-cuda.dll")
        windows_contract(windows_runner)
        data = bytearray((windows_runner / "llama-server.exe").read_bytes())
        data[-1] = 1
        (windows_runner / "llama-server.exe").write_bytes(data)

        with self.assertRaisesRegex(PackageError, "runner SHA-256"):
            build_package(
                ollama_dist=windows_dist,
                runner_payload=windows_runner,
                output=self.root / "windows-out",
                platform="windows-amd64-cuda13",
            )


if __name__ == "__main__":
    unittest.main()
