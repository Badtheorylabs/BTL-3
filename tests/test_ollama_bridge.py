from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

from tools.ollama_bridge import BridgeConfig, create_server


class FakeOpenAIHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def log_message(self, *_args) -> None:
        pass

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json({"status": "ok"})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        self.__class__.requests.append({"path": self.path, "body": body})
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [
                {"choices": [{"delta": {"reasoning_content": "check "}}]},
                {"choices": [{"delta": {"content": "hello"}}]},
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "function": {
                                            "name": "lookup",
                                            "arguments": "{\"q\":\"x\"}",
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 4},
                },
            ]
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            return
        self._json(
            {
                "model": "BTL-3",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "hello",
                            "reasoning_content": "check",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": "{\"q\":\"x\"}",
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4},
            }
        )

    def _json(self, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class BridgeTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeOpenAIHandler.requests = []
        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAIHandler)
        self.upstream_thread = threading.Thread(
            target=self.upstream.serve_forever, daemon=True
        )
        self.upstream_thread.start()
        self.tempdir = TemporaryDirectory()
        model = Path(self.tempdir.name) / "model.gguf"
        model.write_bytes(b"gguf")
        config = BridgeConfig(
            upstream=f"http://127.0.0.1:{self.upstream.server_port}",
            listen_host="127.0.0.1",
            listen_port=0,
            ollama_model="btl3-compact:latest",
            openai_model="BTL-3",
            gguf_path=model,
            gguf_sha256="abcd" * 16,
            context_length=32768,
        )
        self.bridge = create_server(config)
        self.bridge_thread = threading.Thread(
            target=self.bridge.serve_forever, daemon=True
        )
        self.bridge_thread.start()
        self.base = f"http://127.0.0.1:{self.bridge.server_port}"

    def tearDown(self) -> None:
        self.bridge.shutdown()
        self.bridge.server_close()
        self.upstream.shutdown()
        self.upstream.server_close()
        self.tempdir.cleanup()

    def request(self, method: str, path: str, body: dict | None = None):
        data = None if body is None else json.dumps(body).encode()
        request = Request(
            self.base + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=5) as response:
            return response.headers, response.read()

    def test_discovery_endpoints(self) -> None:
        _, version = self.request("GET", "/api/version")
        self.assertIn("btl3-bridge", json.loads(version)["version"])
        _, tags = self.request("GET", "/api/tags")
        model = json.loads(tags)["models"][0]
        self.assertEqual(model["name"], "btl3-compact:latest")
        self.assertEqual(model["size"], 4)
        _, show = self.request(
            "POST", "/api/show", {"model": "btl3-compact:latest"}
        )
        self.assertIn("tools", json.loads(show)["capabilities"])
        _, ps = self.request("GET", "/api/ps")
        self.assertEqual(json.loads(ps)["models"][0]["context_length"], 32768)

    def test_non_streaming_chat_preserves_reasoning_and_tools(self) -> None:
        _, payload = self.request(
            "POST",
            "/api/chat",
            {
                "model": "btl3-compact",
                "stream": False,
                "think": True,
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            },
        )
        result = json.loads(payload)
        self.assertEqual(result["message"]["thinking"], "check")
        self.assertEqual(
            result["message"]["tool_calls"][0]["function"]["arguments"], {"q": "x"}
        )
        upstream = FakeOpenAIHandler.requests[-1]["body"]
        self.assertEqual(upstream["model"], "BTL-3")
        self.assertEqual(upstream["reasoning_effort"], "medium")

    def test_streaming_chat_is_ollama_ndjson(self) -> None:
        headers, payload = self.request(
            "POST",
            "/api/chat",
            {
                "model": "btl3-compact",
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        self.assertEqual(headers.get_content_type(), "application/x-ndjson")
        chunks = [json.loads(line) for line in payload.splitlines()]
        self.assertEqual(chunks[0]["message"]["thinking"], "check ")
        self.assertEqual(chunks[1]["message"]["content"], "hello")
        self.assertTrue(chunks[-1]["done"])
        self.assertEqual(
            chunks[-1]["message"]["tool_calls"][0]["function"]["name"], "lookup"
        )

    def test_generate_maps_to_chat(self) -> None:
        _, payload = self.request(
            "POST",
            "/api/generate",
            {
                "model": "btl3-compact",
                "prompt": "hello",
                "system": "be concise",
                "stream": False,
            },
        )
        result = json.loads(payload)
        self.assertEqual(result["response"], "hello")
        messages = FakeOpenAIHandler.requests[-1]["body"]["messages"]
        self.assertEqual(messages[0], {"role": "system", "content": "be concise"})

    @unittest.skipUnless(shutil.which("ollama"), "ollama CLI is not installed")
    def test_stock_ollama_cli_discovers_bridge_model(self) -> None:
        environment = {**os.environ, "OLLAMA_HOST": self.base}
        result = subprocess.run(
            ["ollama", "list"],
            env=environment,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("btl3-compact:latest", result.stdout)
        run = subprocess.run(
            ["ollama", "run", "btl3-compact:latest", "hello"],
            env=environment,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn("hello", run.stdout)


if __name__ == "__main__":
    unittest.main()
