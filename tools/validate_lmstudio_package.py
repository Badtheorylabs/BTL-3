"""Static contract checks for the BTL-3 LM Studio generator package."""

from __future__ import annotations

import json
from pathlib import Path


class ValidationError(ValueError):
    pass


def require(source: str, token: str, meaning: str) -> None:
    if token not in source:
        raise ValidationError(f"generator lacks {meaning}: {token}")


def validate_tool_flow(source: str) -> None:
    finish_start = source.find("function finishTool(")
    start = source.find("function bufferToolDelta(")
    end = source.find("\nasync function generate(", start)
    if finish_start < 0 or start < 0 or end < 0:
        raise ValidationError("parallel tool-call buffer is not isolated")
    finish = source[finish_start:start]
    handler = source[start:end]
    require(handler, "pending.get(call.index)", "per-index parallel call buffer")
    require(handler, "state.argumentFragments.push", "tool argument fragment buffer")
    require(source, "parallel_tool_calls: true", "parallel tool-call preservation")
    lifecycle = [
        "ctl.toolCallGenerationStarted",
        "ctl.toolCallGenerationNameReceived",
        "ctl.toolCallGenerationArgumentFragmentGenerated",
        "ctl.toolCallGenerationEnded",
    ]
    positions = [finish.find(token) for token in lifecycle]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        raise ValidationError("buffered tool lifecycle is not emitted in SDK order")
    if "parallel_tool_calls: false" in source:
        raise ValidationError("parallel tool calls are disabled")


def validate_package(root: Path) -> dict:
    required = [
        "manifest.json",
        "model.yaml",
        "package.json",
        "src/config.ts",
        "src/generator.ts",
        "src/index.ts",
        "src/nativeRunner.ts",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValidationError(f"missing LM Studio files: {missing}")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("type") != "plugin" or manifest.get("runner") != "node":
        raise ValidationError("manifest is not an LM Studio Node plugin")
    source = (root / "src/generator.ts").read_text()
    config = (root / "src/config.ts").read_text()
    runner = (root / "src/nativeRunner.ts").read_text()
    require(source, "context.withGenerator", "official generator registration")
    require(source, "ctl.abortSignal", "cancellation")
    require(source, "reasoningType: \"reasoning\"", "reasoning fragments")
    require(source, "ctl.toolCallGenerationStarted", "tool-call lifecycle")
    require(source, "throw asError(error)", "API and abort error propagation")
    validate_tool_flow(source)
    require(source, "ctl.fragmentGenerated", "streaming fragments")
    require(source, "stream: true", "upstream streaming")
    default_url = "http://127.0.0.1:8080/v1"
    require(config, default_url, "local BTL-3 endpoint default")
    require(config, "\"autoStart\"", "automatic native-runner option")
    require(source, "ensureNativeRunner", "automatic native-runner startup")
    require(runner, "spawn(runner, runnerArguments", "native subprocess launch")
    require(runner, "\"--model\", modelPath", "explicit model argument")
    require(runner, "\"--offline\"", "offline runner mode")
    if "shell: true" in runner:
        raise ValidationError("native runner must not use shell execution")
    model_yaml = (root / "model.yaml").read_text()
    require(model_yaml, "- btl3-avq2-native", "native-only compatibility label")
    if "\n    - gguf" in model_yaml or "\n      - gguf" in model_yaml:
        raise ValidationError("model.yaml falsely claims stock GGUF compatibility")
    return {
        "model": "BTL-3 Compact",
        "streaming": True,
        "cancellation": True,
        "tool_calls": True,
        "parallel_tool_calls": True,
        "tool_call_fragments_buffered": True,
        "api_errors_propagated": True,
        "reasoning": True,
        "auto_start": True,
        "default_base_url": default_url,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_package(args.root), indent=2))


if __name__ == "__main__":
    main()
