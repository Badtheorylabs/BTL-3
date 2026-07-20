from __future__ import annotations

import json
from pathlib import Path
import unittest

from tools.validate_lmstudio_package import validate_package


ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "integrations/lmstudio/btl3-native"


class LMStudioPackageTest(unittest.TestCase):
    def test_generator_package_uses_official_surfaces(self) -> None:
        report = validate_package(PLUGIN)
        self.assertEqual(report["model"], "BTL-3 Compact")
        self.assertTrue(report["streaming"])
        self.assertTrue(report["cancellation"])
        self.assertTrue(report["tool_calls"])
        self.assertTrue(report["parallel_tool_calls"])
        self.assertTrue(report["tool_call_fragments_buffered"])
        self.assertTrue(report["api_errors_propagated"])
        self.assertTrue(report["auto_start"])
        self.assertEqual(report["default_base_url"], "http://127.0.0.1:8080/v1")

    def test_manifest_is_explicitly_local_generator(self) -> None:
        manifest = json.loads((PLUGIN / "manifest.json").read_text())
        self.assertEqual(manifest["type"], "plugin")
        self.assertEqual(manifest["runner"], "node")
        self.assertEqual(manifest["name"], "btl3-native")

    def test_model_yaml_is_catalog_metadata_not_stock_engine_claim(self) -> None:
        text = (PLUGIN / "model.yaml").read_text()
        self.assertIn("model: badtheorylabs/btl-3-compact", text)
        self.assertIn("compatibilityTypes:", text)
        self.assertIn("- btl3-avq2-native", text)
        self.assertNotIn("- gguf", text)


if __name__ == "__main__":
    unittest.main()
