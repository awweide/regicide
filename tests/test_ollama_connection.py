import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest

from regicide.agent import Ollama


class _OllamaHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path != "/api/tags":
            self.send_error(404)
            return
        self._send_json({"models": [{"name": "test-model"}]})

    def do_POST(self):
        if self.path != "/api/generate":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode())
        self.server.seen_payload = payload
        self._send_json({"response": "pong"})

    def _send_json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def fake_ollama_server():
    server = HTTPServer(("127.0.0.1", 0), _OllamaHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_ollama_connection_check_prints_steps_and_calls_tags_and_generate(fake_ollama_server, capsys):
    url = f"http://127.0.0.1:{fake_ollama_server.server_port}"
    result = Ollama("test-model", url=url, timeout=2, retries=0).check_connection()

    output = capsys.readouterr().out
    assert "[1/4] Checking Ollama server URL" in output
    assert "[2/4] Requesting model list" in output
    assert "[3/4] Sending a minimal non-streaming /api/generate prompt" in output
    assert "[4/4] Ollama communication check completed successfully." in output
    assert result["response"] == "pong"
    assert result["available_models"] == ["test-model"]
    assert fake_ollama_server.seen_payload == {
        "model": "test-model",
        "prompt": "Reply with exactly: pong",
        "stream": False,
    }


@pytest.mark.skipif(
    os.environ.get("OLLAMA_INTEGRATION") != "1",
    reason="set OLLAMA_INTEGRATION=1 to run against a real Ollama server",
)
def test_real_ollama_server_communication(capsys):
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "llama3")
    result = Ollama(model, url=url, timeout=30, retries=0).check_connection()

    output = capsys.readouterr().out
    assert "[4/4] Ollama communication check completed successfully." in output
    assert isinstance(result["response"], str)
    assert result["response"].strip()
