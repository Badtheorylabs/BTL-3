"""Small Ollama API bridge for the BTL-3 OpenAI-compatible local server."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BRIDGE_VERSION = "btl3-bridge-0.1.0"

@dataclass(frozen=True)
class BridgeConfig:
    upstream: str
    listen_host: str
    listen_port: int
    ollama_model: str
    openai_model: str
    gguf_path: Path
    gguf_sha256: str
    context_length: int
    api_key: str = ""

def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _arguments(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {"_raw": str(value)}
    return parsed if isinstance(parsed, dict) else {"_value": parsed}

def _ollama_tools(calls: list[dict]) -> list[dict]:
    result = []
    for index, call in enumerate(calls):
        function = call.get("function", {})
        result.append(
            {
                "type": "function",
                "function": {
                    "index": index,
                    "name": function.get("name", ""),
                    "arguments": _arguments(function.get("arguments")),
                },
            }
        )
    return result

def _openai_messages(messages: list[dict]) -> list[dict]:
    result: list[dict] = []
    pending_ids: list[str] = []
    for message in messages:
        role = message.get("role")
        converted = {"role": role, "content": message.get("content", "")}
        if role == "assistant":
            reasoning = message.get("thinking")
            if reasoning:
                converted["reasoning_content"] = reasoning
            calls = []
            pending_ids = []
            for index, call in enumerate(message.get("tool_calls") or []):
                function = call.get("function", {})
                call_id = call.get("id") or f"call_{index}"
                pending_ids.append(call_id)
                calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": function.get("name", ""),
                            "arguments": json.dumps(
                                function.get("arguments") or {},
                                separators=(",", ":"),
                            ),
                        },
                    }
                )
            if calls:
                converted["tool_calls"] = calls
        elif role == "tool":
            converted["tool_call_id"] = (
                message.get("tool_call_id")
                or (pending_ids.pop(0) if pending_ids else "call_0")
            )
        result.append(converted)
    return result

def _sampling(body: dict) -> dict:
    options = body.get("options") or {}
    mappings = {
        "temperature": "temperature",
        "top_p": "top_p",
        "seed": "seed",
        "stop": "stop",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
        "num_predict": "max_tokens",
    }
    return {
        target: options[source]
        for source, target in mappings.items()
        if source in options
    }

def _chat_payload(config: BridgeConfig, body: dict, messages: list[dict]) -> dict:
    payload = {
        "model": config.openai_model,
        "messages": _openai_messages(messages),
        "stream": bool(body.get("stream", True)),
        **_sampling(body),
    }
    if body.get("tools"):
        payload["tools"] = body["tools"]
    think = body.get("think")
    if think is not None:
        payload["reasoning_effort"] = (
            think if isinstance(think, str) else ("medium" if think else "none")
        )
    response_format = body.get("format")
    if response_format == "json":
        payload["response_format"] = {"type": "json_object"}
    elif isinstance(response_format, dict):
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": response_format},
        }
    return payload

def _metrics(usage: dict, started: int) -> dict:
    duration = time.monotonic_ns() - started
    return {
        "total_duration": duration,
        "load_duration": 0,
        "prompt_eval_count": usage.get("prompt_tokens", 0),
        "prompt_eval_duration": 0,
        "eval_count": usage.get("completion_tokens", 0),
        "eval_duration": duration,
    }

def _model_details() -> dict:
    return {
        "parent_model": "",
        "format": "gguf",
        "family": "qwen3.5",
        "families": ["qwen3.5"],
        "parameter_size": "27B",
        "quantization_level": "BTL3_AVQ2_MIXED",
    }

def _handler(config: BridgeConfig):
    class OllamaBridgeHandler(BaseHTTPRequestHandler):
        server_version = BRIDGE_VERSION

        def log_message(self, fmt: str, *args) -> None:
            print(f"[ollama-bridge] {self.address_string()} {fmt % args}")

        def do_HEAD(self) -> None:
            self.send_response(200 if self.path == "/" else 404)
            self.end_headers()

        def do_GET(self) -> None:
            if self.path == "/":
                self._text("Ollama is running")
            elif self.path == "/api/version":
                self._json(200, {"version": BRIDGE_VERSION})
            elif self.path == "/api/tags":
                self._json(200, {"models": [self._model_entry()]})
            elif self.path == "/api/ps":
                entry = self._model_entry()
                entry.update(
                    {
                        "expires_at": "9999-12-31T23:59:59Z",
                        "size_vram": config.gguf_path.stat().st_size,
                        "context_length": config.context_length,
                    }
                )
                self._json(200, {"models": [entry]})
            else:
                self._json(404, {"error": f"unsupported endpoint: {self.path}"})

        def do_POST(self) -> None:
            try:
                body = self._read_json()
                if self.path == "/api/show":
                    self._show(body)
                elif self.path == "/api/chat":
                    self._generate(body, kind="chat")
                elif self.path == "/api/generate":
                    self._generate(body, kind="generate")
                else:
                    self._json(
                        501,
                        {
                            "error": (
                                f"{self.path} is not implemented by the BTL-3 "
                                "local API bridge"
                            )
                        },
                    )
            except ValueError as exc:
                self._json(400, {"error": str(exc)})

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 16 * 1024 * 1024:
                raise ValueError("request body must be 1 byte to 16 MiB")
            try:
                body = json.loads(self.rfile.read(length))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON: {exc.msg}") from exc
            if not isinstance(body, dict):
                raise ValueError("request body must be a JSON object")
            return body

        def _validate_model(self, body: dict) -> None:
            requested = str(body.get("model") or body.get("name") or "")
            accepted = {
                config.ollama_model,
                config.ollama_model.removesuffix(":latest"),
                config.openai_model,
            }
            if requested not in accepted:
                raise ValueError(
                    f"model {requested!r} not found; use {config.ollama_model!r}"
                )

        def _show(self, body: dict) -> None:
            self._validate_model(body)
            self._json(
                200,
                {
                    "parameters": f"num_ctx {config.context_length}",
                    "license": "See the BTL-3 release license.",
                    "capabilities": ["completion", "tools", "thinking"],
                    "modified_at": _now(),
                    "details": _model_details(),
                    "model_info": {
                        "general.architecture": "qwen3.5",
                        "general.parameter_count": 27_000_000_000,
                        "general.quantization_type": "BTL3_AVQ2_MIXED",
                        "qwen3_5.context_length": config.context_length,
                    },
                },
            )

        def _generate(self, body: dict, *, kind: str) -> None:
            self._validate_model(body)
            if kind == "chat":
                messages = body.get("messages")
                if not isinstance(messages, list):
                    raise ValueError("messages must be an array")
            else:
                prompt = body.get("prompt")
                if not isinstance(prompt, str):
                    raise ValueError("prompt must be a string")
                messages = []
                if body.get("system"):
                    messages.append({"role": "system", "content": body["system"]})
                messages.append({"role": "user", "content": prompt})
            payload = _chat_payload(config, body, messages)
            started = time.monotonic_ns()
            try:
                upstream = self._upstream(payload)
                if payload["stream"]:
                    self._stream(upstream, kind, started)
                else:
                    self._complete(json.load(upstream), kind, started)
            except HTTPError as exc:
                detail = exc.read().decode(errors="replace")
                self._json(exc.code, {"error": f"upstream error: {detail}"})
            except URLError as exc:
                self._json(502, {"error": f"BTL-3 server unavailable: {exc.reason}"})

        def _upstream(self, payload: dict):
            headers = {"Content-Type": "application/json"}
            if config.api_key:
                headers["Authorization"] = f"Bearer {config.api_key}"
            request = Request(
                config.upstream.rstrip("/") + "/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers=headers,
                method="POST",
            )
            return urlopen(request, timeout=3600)

        def _complete(self, response: dict, kind: str, started: int) -> None:
            choice = response["choices"][0]
            message = choice.get("message", {})
            content = message.get("content") or ""
            thinking = (
                message.get("reasoning_content")
                or message.get("reasoning")
                or ""
            )
            calls = _ollama_tools(message.get("tool_calls") or [])
            result = {
                "model": config.ollama_model,
                "created_at": _now(),
                "done": True,
                "done_reason": choice.get("finish_reason") or "stop",
                **_metrics(response.get("usage") or {}, started),
            }
            if kind == "chat":
                result["message"] = {
                    "role": "assistant",
                    "content": content,
                    **({"thinking": thinking} if thinking else {}),
                    **({"tool_calls": calls} if calls else {}),
                }
            else:
                result["response"] = content
                if thinking:
                    result["thinking"] = thinking
            self._json(200, result)

        def _stream(self, upstream, kind: str, started: int) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            calls: dict[int, dict] = {}
            usage: dict = {}
            finish_reason = "stop"
            try:
                for raw_line in upstream:
                    line = raw_line.decode(errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    usage = chunk.get("usage") or usage
                    choice = (chunk.get("choices") or [{}])[0]
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta") or {}
                    thinking = (
                        delta.get("reasoning_content")
                        or delta.get("reasoning")
                        or ""
                    )
                    content = delta.get("content") or ""
                    for call in delta.get("tool_calls") or []:
                        index = int(call.get("index", 0))
                        state = calls.setdefault(
                            index,
                            {"id": call.get("id", ""), "name": "", "arguments": ""},
                        )
                        state["id"] = call.get("id") or state["id"]
                        function = call.get("function") or {}
                        state["name"] += function.get("name") or ""
                        state["arguments"] += function.get("arguments") or ""
                    if thinking:
                        self._stream_chunk(kind, thinking=thinking)
                    if content:
                        self._stream_chunk(kind, content=content)
                tools = _ollama_tools(
                    [
                        {
                            "id": state["id"],
                            "function": {
                                "name": state["name"],
                                "arguments": state["arguments"],
                            },
                        }
                        for _, state in sorted(calls.items())
                    ]
                )
                self._stream_chunk(
                    kind,
                    done=True,
                    tools=tools,
                    finish_reason=finish_reason,
                    metrics=_metrics(usage, started),
                )
            except (BrokenPipeError, ConnectionResetError):
                upstream.close()

        def _stream_chunk(
            self,
            kind: str,
            *,
            content: str = "",
            thinking: str = "",
            done: bool = False,
            tools: list[dict] | None = None,
            finish_reason: str = "",
            metrics: dict | None = None,
        ) -> None:
            payload = {
                "model": config.ollama_model,
                "created_at": _now(),
                "done": done,
            }
            if kind == "chat":
                payload["message"] = {
                    "role": "assistant",
                    "content": content,
                    **({"thinking": thinking} if thinking else {}),
                    **({"tool_calls": tools} if tools else {}),
                }
            else:
                payload["response"] = content
                if thinking:
                    payload["thinking"] = thinking
            if done:
                payload["done_reason"] = finish_reason
                payload.update(metrics or {})
            self.wfile.write(json.dumps(payload, separators=(",", ":")).encode())
            self.wfile.write(b"\n")
            self.wfile.flush()

        def _model_entry(self) -> dict:
            return {
                "name": config.ollama_model,
                "model": config.ollama_model,
                "modified_at": datetime.fromtimestamp(
                    config.gguf_path.stat().st_mtime, timezone.utc
                )
                .isoformat()
                .replace("+00:00", "Z"),
                "size": config.gguf_path.stat().st_size,
                "digest": config.gguf_sha256,
                "details": _model_details(),
            }

        def _json(self, status: int, body: dict) -> None:
            payload = json.dumps(body, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _text(self, body: str) -> None:
            payload = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return OllamaBridgeHandler

def create_server(config: BridgeConfig) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(
        (config.listen_host, config.listen_port),
        _handler(config),
    )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", default="http://127.0.0.1:8080")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--model", default="btl3-compact:latest")
    parser.add_argument("--openai-model", default="BTL-3")
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--gguf-sha256", required=True)
    parser.add_argument("--ctx-size", type=int, default=32768)
    parser.add_argument("--api-key", default="")
    args = parser.parse_args()
    config = BridgeConfig(
        upstream=args.upstream,
        listen_host=args.host,
        listen_port=args.port,
        ollama_model=args.model,
        openai_model=args.openai_model,
        gguf_path=args.gguf.resolve(),
        gguf_sha256=args.gguf_sha256,
        context_length=args.ctx_size,
        api_key=args.api_key,
    )
    if not config.gguf_path.is_file():
        parser.error(f"GGUF does not exist: {config.gguf_path}")
    server = create_server(config)
    print(
        f"BTL-3 Ollama API bridge listening on "
        f"http://{config.listen_host}:{server.server_port}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
