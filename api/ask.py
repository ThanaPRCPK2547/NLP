from __future__ import annotations

import importlib.util
import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
AGENT_PATH = ROOT_DIR / "model" / "20260425_ai_agent_v2.py"
MAX_REQUEST_BYTES = 24_000


def _load_agent_module() -> Any:
    module_name = "data_engineering_rag_agent"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, AGENT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load agent module from {AGENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_module = _load_agent_module()
_agent = _module.DataEngineeringRAGAgent()


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
        self._send_json({"ok": True})

    def do_GET(self) -> None:
        self._send_json(
            {
                "ok": True,
                "service": "Data Engineering RAG Agent",
                "endpoint": "/api/ask",
                "llm_enabled": bool(_agent.llm.available),
                "retrieval_mode": _agent.retriever.mode,
            }
        )

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json({"error": "Invalid Content-Length"}, status=400)
            return

        if length <= 0 or length > MAX_REQUEST_BYTES:
            self._send_json({"error": "Request body is empty or too large"}, status=413)
            return

        try:
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body or "{}")
        except UnicodeDecodeError:
            self._send_json({"error": "Request body must be UTF-8"}, status=400)
            return
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON payload"}, status=400)
            return

        if not isinstance(payload, dict):
            self._send_json({"error": "JSON payload must be an object"}, status=400)
            return

        question = payload.get("question", "")
        if not isinstance(question, str) or not question.strip():
            self._send_json({"error": "Question is required"}, status=400)
            return

        try:
            result = _agent.ask(
                question,
                name=payload.get("name", "Guest"),
                role_level=payload.get("role_level", "mid"),
                project_size=payload.get("project_size", "medium"),
                records_per_day=payload.get("records_per_day", 1_000_000),
                preferred_stack=payload.get("preferred_stack", "cloud-agnostic"),
                answer_style=payload.get("answer_style", "practical"),
            )
        except Exception:
            self._send_json({"error": "Agent failed to process the request"}, status=500)
            return

        self._send_json(
            {
                "markdown": result.markdown,
                "structured": result.structured,
                "debug": {
                    k: v
                    for k, v in result.debug.items()
                    if k not in {"expanded_queries", "tool_trace", "output_schema_keys"}
                },
            }
        )

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._security_headers()
        self.end_headers()
        self.wfile.write(body)

    def _security_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; frame-ancestors 'none'",
        )
