#!/usr/bin/env python3

"""Local HTTP chat UI (HTML/CSS from eliza_experiments/burmese_chat_ui.py) + modular BiLSTM + eliza (scripts/chat.py)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# reuse full page from experiments (same HTML/JS/CSS)
sys.path.insert(0, str(ROOT_DIR / "eliza_experiments"))
import burmese_chat_ui as _burmese_ui  # noqa: E402

render_page = _burmese_ui.render_page

from scripts.chat import (  # noqa: E402
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_STOPWORDS_PATH,
    chat_turn,
    load_chat_context,
    resolve_project_path,
)


class ModularWebBackend:
    # function to wire checkpoint + eliza to the same json api the original burmese_chat_ui expects
    def __init__(self, checkpoint_path: str, stopwords_path: str, lang: str = "mm"):
        self.lang = lang if lang in ("mm", "en") else "mm"
        self.stopwords_path = stopwords_path
        self.ctx = load_chat_context(checkpoint_path, language=self.lang)
        self.eliza = self.ctx["eliza"]

    @property
    def model_loaded(self) -> bool:
        return self.ctx["model"] is not None

    def status_text(self) -> str:
        if self.model_loaded:
            return "LSTM စိတ်ခံစားမှုခန့်မှန်းမှု ပါဝင်ပြီး အလုပ်လုပ်နေပါတယ်။"
        return "စိတ်ခံစားမှုမော်ဒယ်ဖိုင် မတွေ့သေးပါ။ Rule-based chatbot အဖြစ်ပဲ အလုပ်လုပ်နေပါတယ်။"

    def greeting_payload(self) -> dict[str, Any]:
        return {
            "reply": random.choice(self.eliza.script["initials"]),
            "emotion": None,
            "score": None,
            "quit": False,
            "model_loaded": self.model_loaded,
            "status_text": self.status_text(),
        }

    def chat(self, message: str) -> dict[str, Any]:
        text = str(message or "").strip()
        if not text:
            return {
                "reply": "တစ်ခုခု ရိုက်ပြီး ပြောပြပါ။",
                "emotion": None,
                "score": None,
                "quit": False,
                "model_loaded": self.model_loaded,
                "status_text": self.status_text(),
            }

        out = chat_turn(self.ctx, text, self.stopwords_path)
        if out["kind"] == "empty":
            return {
                "reply": "တစ်ခုခု ရိုက်ပြီး ပြောပြပါ။",
                "emotion": None,
                "score": None,
                "quit": False,
                "model_loaded": self.model_loaded,
                "status_text": self.status_text(),
            }
        if out["kind"] == "quit":
            return {
                "reply": out["final"],
                "emotion": None,
                "score": None,
                "quit": True,
                "model_loaded": self.model_loaded,
                "status_text": self.status_text(),
            }
        return {
            "reply": out["eliza_reply"],
            "emotion": out["emotion_label"],
            "score": out["emotion_score"],
            "quit": False,
            "model_loaded": self.model_loaded,
            "status_text": self.status_text(),
        }


class ChatHandler(BaseHTTPRequestHandler):
    backend: ModularWebBackend

    def _send_html(self, html: str, status: HTTPStatus = HTTPStatus.OK):
        payload = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, data: dict[str, Any], status: HTTPStatus = HTTPStatus.OK):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path in {"/", "/index.html"}:
            self._send_html(render_page())
            return

        if self.path == "/api/reset":
            self._send_json(self.backend.greeting_payload())
            return

        if self.path == "/api/health":
            self._send_json(
                {
                    "ok": True,
                    "lang": self.backend.lang,
                    "model_loaded": self.backend.model_loaded,
                    "status_text": self.backend.status_text(),
                }
            )
            return

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if self.path != "/api/chat":
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length).decode("utf-8") if content_length else "{}"
        try:
            body = json.loads(raw_body or "{}")
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body"}, status=HTTPStatus.BAD_REQUEST)
            return

        payload = self.backend.chat(body.get("message", ""))
        self._send_json(payload)

    def log_message(self, format: str, *args):
        return


def parse_args():
    parser = argparse.ArgumentParser(description="Burmese hybrid ELIZA web UI (modular model + eliza)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--lang", default="mm", choices=["en", "mm"])
    parser.add_argument("--checkpoint_path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--stopwords_path", default=DEFAULT_STOPWORDS_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    backend = ModularWebBackend(
        checkpoint_path=resolve_project_path(args.checkpoint_path),
        stopwords_path=resolve_project_path(args.stopwords_path),
        lang=args.lang,
    )
    ChatHandler.backend = backend
    server = ThreadingHTTPServer((args.host, args.port), ChatHandler)

    print(f"Hybrid ELIZA web UI (modular) at http://{args.host}:{args.port}")
    print(backend.status_text())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
