#!/usr/bin/env python3

"""Standalone local web UI for Burmese Hybrid ELIZA chat."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import random
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
print(BASE_DIR)
PREFERRED_MODULE_PATH = BASE_DIR / "hybrid-eliza-improve-ver1.py"
FALLBACK_MODULE_PATH = BASE_DIR / "hybrid-eliza-mm-lstm.py"
MODULE_PATH = PREFERRED_MODULE_PATH if PREFERRED_MODULE_PATH.exists() else FALLBACK_MODULE_PATH
MYANMAR_TOKEN_RE = re.compile(r"[\u1000-\u109F\uAA60-\uAA7F]+|[a-zA-Z0-9]+")


def load_hybrid_module():
    spec = importlib.util.spec_from_file_location("hybrid_eliza_runtime", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load chatbot module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_scripts_from_source(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "SCRIPTS":
                    return ast.literal_eval(node.value)
    raise RuntimeError(f"Could not find SCRIPTS in {path}")


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[၊။!?,;:\"'()\[\]{}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_char_ngrams(text: str, min_n: int = 2, max_n: int = 3) -> list[str]:
    compact = text.replace(" ", "")
    ngrams: list[str] = []
    for n in range(min_n, max_n + 1):
        if len(compact) < n:
            continue
        for index in range(len(compact) - n + 1):
            ngrams.append(compact[index : index + n])
    return ngrams


def tokenize_text(text: str, lang: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    if lang == "mm":
        if " " in text:
            base_tokens = [token for token in text.split() if token]
        else:
            base_tokens = MYANMAR_TOKEN_RE.findall(text)
        return base_tokens + build_char_ngrams(text)
    return text.split()


class ChatBackend:
    def __init__(self, lang: str = "mm", model_path: str | None = None):
        self.lang = lang
        self.module = None
        self.bot = None
        self.import_error: str | None = None
        self.scripts = load_scripts_from_source(MODULE_PATH)

        default_model = BASE_DIR / ("eliza_eq_mm_lstm.pth" if lang == "mm" else "eliza_eq_lstm.pth")
        resolved_model_path = str(Path(model_path).expanduser()) if model_path else str(default_model)

        try:
            self.module = load_hybrid_module()
            self.scripts = self.module.SCRIPTS
            self.bot = self.module.HybridEliza(lang=lang, model_path=resolved_model_path)
            self.bot.load_model()
        except Exception as exc:
            self.import_error = str(exc)

    @property
    def model_loaded(self) -> bool:
        return self.bot is not None and self.bot.model is not None

    def status_text(self) -> str:
        if self.model_loaded:
            return "LSTM စိတ်ခံစားမှုခန့်မှန်းမှု ပါဝင်ပြီး အလုပ်လုပ်နေပါတယ်။"
        if self.import_error:
            return "PyTorch/LSTM မော်ဒယ်မရှိသေးပါ။ Burmese rule-based chatbot အဖြစ်ပဲ အလုပ်လုပ်နေပါတယ်။"
        return "စိတ်ခံစားမှုမော်ဒယ်ဖိုင် မတွေ့သေးပါ။ Rule-based chatbot အဖြစ်ပဲ အလုပ်လုပ်နေပါတယ်။"

    def greeting_payload(self) -> dict[str, Any]:
        return {
            "reply": random.choice(self.scripts[self.lang]["initials"]),
            "emotion": None,
            "score": None,
            "quit": False,
            "model_loaded": self.model_loaded,
            "status_text": self.status_text(),
        }

    def reflect(self, fragment: str) -> str:
        if self.bot is not None:
            return self.bot.reflect(fragment)
        tokens = tokenize_text(fragment, self.lang)
        reflected = [self.scripts[self.lang]["posts"].get(token, token) for token in tokens]
        return " ".join(reflected).strip()

    def rule_respond(self, text: str) -> str:
        if self.bot is not None:
            return self.bot.rule_respond(text)

        normalized = normalize_text(text)
        for src, dst in self.scripts[self.lang]["pres"].items():
            normalized = normalized.replace(src, dst)

        for pattern, responses, _rank in self.scripts[self.lang]["keywords"]:
            match = re.search(pattern, normalized)
            if match:
                response = random.choice(responses)
                fragments = [self.reflect(group) for group in match.groups() if group and group.strip()]
                return response.format(*fragments) if fragments else response
        return "ဆက်ပြောပြပါ။" if self.lang == "mm" else "Please continue."

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

        normalized = self.module.normalize_text(text) if self.module is not None else normalize_text(text)
        if normalized in self.scripts[self.lang]["quits"]:
            return {
                "reply": random.choice(self.scripts[self.lang]["finals"]),
                "emotion": None,
                "score": None,
                "quit": True,
                "model_loaded": self.model_loaded,
                "status_text": self.status_text(),
            }

        reply = self.rule_respond(text)
        emotion, score = (self.bot.get_eq(text) if self.bot is not None else (None, None))
        return {
            "reply": reply,
            "emotion": emotion if self.model_loaded else None,
            "score": score if self.model_loaded else None,
            "quit": False,
            "model_loaded": self.model_loaded,
            "status_text": self.status_text(),
        }


def render_page() -> str:
    return """<!DOCTYPE html>
<html lang="my">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Burmese Hybrid ELIZA</title>
  <style>
    :root {
      --bg-top: #f8d08a;
      --bg-bottom: #f6efe5;
      --panel: rgba(255, 250, 242, 0.92);
      --panel-border: rgba(110, 65, 24, 0.12);
      --ink: #2f241b;
      --muted: #6f5a47;
      --accent: #b45f06;
      --accent-soft: #ffe0a8;
      --user-bubble: #fff3d7;
      --bot-bubble: #fff9ef;
      --shadow: 0 24px 70px rgba(83, 49, 16, 0.14);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at top left, rgba(255, 249, 220, 0.9), transparent 30%),
        radial-gradient(circle at bottom right, rgba(233, 170, 82, 0.35), transparent 28%),
        linear-gradient(180deg, var(--bg-top), var(--bg-bottom));
      color: var(--ink);
      font-family: "Padauk", "Noto Sans Myanmar", "Myanmar Text", "Segoe UI", sans-serif;
    }

    .page {
      min-height: 100vh;
      padding: 24px;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .shell {
      width: min(960px, 100%);
      min-height: 82vh;
      display: grid;
      grid-template-rows: auto auto 1fr auto auto;
      gap: 18px;
      padding: 22px;
      border-radius: 28px;
      background: var(--panel);
      border: 1px solid var(--panel-border);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }

    .hero {
      display: flex;
      flex-wrap: wrap;
      align-items: end;
      justify-content: space-between;
      gap: 12px;
    }

    .title {
      margin: 0;
      font-size: clamp(1.8rem, 4vw, 3rem);
      line-height: 1.05;
      letter-spacing: -0.03em;
    }

    .subtitle {
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 0.98rem;
    }

    .pill-row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
    }

    .status {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(255, 246, 227, 0.95);
      color: var(--muted);
      border: 1px solid rgba(180, 95, 6, 0.15);
      font-size: 0.92rem;
    }

    .status-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 0 6px rgba(180, 95, 6, 0.12);
    }

    .ghost-button {
      border: 0;
      border-radius: 999px;
      background: #fff;
      color: var(--ink);
      padding: 10px 16px;
      font: inherit;
      cursor: pointer;
      box-shadow: 0 10px 24px rgba(83, 49, 16, 0.08);
    }

    .messages {
      overflow-y: auto;
      padding: 8px 4px 8px 0;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .message {
      max-width: min(82%, 680px);
      padding: 14px 16px;
      border-radius: 20px;
      line-height: 1.55;
      white-space: pre-wrap;
      word-break: break-word;
      box-shadow: 0 10px 26px rgba(83, 49, 16, 0.08);
    }

    .message.user {
      align-self: flex-end;
      background: var(--user-bubble);
      border-bottom-right-radius: 6px;
    }

    .message.bot {
      align-self: flex-start;
      background: var(--bot-bubble);
      border-bottom-left-radius: 6px;
    }

    .meta {
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.88rem;
    }

    .chips {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }

    .chip {
      border: 0;
      border-radius: 999px;
      padding: 10px 14px;
      font: inherit;
      cursor: pointer;
      background: var(--accent-soft);
      color: var(--ink);
    }

    .composer {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: end;
    }

    .composer textarea {
      min-height: 72px;
      max-height: 180px;
      resize: vertical;
      border: 1px solid rgba(110, 65, 24, 0.12);
      border-radius: 22px;
      padding: 16px 18px;
      font: inherit;
      color: var(--ink);
      background: rgba(255, 255, 255, 0.92);
      outline: none;
    }

    .composer textarea:focus {
      border-color: rgba(180, 95, 6, 0.4);
      box-shadow: 0 0 0 4px rgba(180, 95, 6, 0.08);
    }

    .send-button {
      min-width: 124px;
      min-height: 56px;
      border: 0;
      border-radius: 20px;
      padding: 0 20px;
      font: inherit;
      cursor: pointer;
      color: #fff;
      background: linear-gradient(135deg, #c96c08, #8c4304);
      box-shadow: 0 16px 32px rgba(140, 67, 4, 0.25);
    }

    .footnote {
      margin: 0;
      color: var(--muted);
      font-size: 0.9rem;
      text-align: center;
    }

    @media (max-width: 720px) {
      .page {
        padding: 10px;
      }

      .shell {
        min-height: 100vh;
        border-radius: 0;
      }

      .composer {
        grid-template-columns: 1fr;
      }

      .send-button {
        width: 100%;
      }

      .message {
        max-width: 92%;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <main class="shell">
      <header class="hero">
        <div>
          <h1 class="title">Burmese Hybrid ELIZA</h1>
          <p class="subtitle">မြန်မာလို စာရိုက်ပြီး စကားပြောနိုင်တဲ့ local chat UI</p>
        </div>
        <button id="resetButton" class="ghost-button" type="button">စကားဝိုင်းအသစ်</button>
      </header>

      <div class="pill-row">
        <div class="status">
          <span class="status-dot"></span>
          <span id="statusText">ချိတ်ဆက်နေပါတယ်...</span>
        </div>
      </div>

      <section id="messages" class="messages" aria-live="polite"></section>

      <div class="chips">
        <button class="chip" type="button" data-text="အခုနောက်ပိုင်း စိတ်ဖိစီးနေတယ်။">အခုနောက်ပိုင်း စိတ်ဖိစီးနေတယ်။</button>
        <button class="chip" type="button" data-text="အိပ်မပျော်ဘူး။">အိပ်မပျော်ဘူး။</button>
        <button class="chip" type="button" data-text="ကျွန်တော် အခက်အခဲရှိနေတယ်။">ကျွန်တော် အခက်အခဲရှိနေတယ်။</button>
        <button class="chip" type="button" data-text="ဘာကြောင့် ဒီလိုခံစားရတာလဲ မသိဘူး။">ဘာကြောင့် ဒီလိုခံစားရတာလဲ မသိဘူး။</button>
      </div>

      <form id="composer" class="composer">
        <textarea id="messageInput" lang="my" placeholder="ဒီမှာ မြန်မာလို ရေးပြီး စကားပြောပါ..." required></textarea>
        <button class="send-button" type="submit">ပို့မယ်</button>
      </form>

      <p class="footnote">Enter နဲ့ပို့ပါ။ Shift + Enter နဲ့ စာကြောင်းအသစ်ဆင်းနိုင်ပါတယ်။</p>
    </main>
  </div>

  <script>
    const messages = document.getElementById("messages");
    const composer = document.getElementById("composer");
    const input = document.getElementById("messageInput");
    const statusText = document.getElementById("statusText");
    const resetButton = document.getElementById("resetButton");
    const state = { locked: false };

    function formatScore(score) {
      return `${(score * 100).toFixed(1)}%`;
    }

    function scrollToBottom() {
      messages.scrollTop = messages.scrollHeight;
    }

    function appendEmotionMeta(bubble, payload = {}) {
      if (!payload.emotion) {
        return;
      }

      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = `Emotion score: ${payload.emotion} (${formatScore(payload.score)})`;
      bubble.appendChild(meta);
    }

    function appendMessage(role, text, payload = {}) {
      const bubble = document.createElement("article");
      bubble.className = `message ${role}`;
      bubble.textContent = text;

      if (false && role === "bot" && payload.emotion) {
        const meta = document.createElement("div");
        meta.className = "meta";
        meta.textContent = `စိတ်ခံစားမှု ခန့်မှန်းချက်: ${payload.emotion} (${formatScore(payload.score)})`;
        bubble.appendChild(meta);
      }

      messages.appendChild(bubble);
      scrollToBottom();
      return bubble;
    }

    async function requestJson(url, options = {}) {
      const response = await fetch(url, options);
      if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`);
      }
      return response.json();
    }

    function setLocked(locked) {
      state.locked = locked;
      input.disabled = locked;
      composer.querySelector("button").disabled = locked;
    }

    async function resetChat() {
      const payload = await requestJson("/api/reset");
      messages.innerHTML = "";
      statusText.textContent = payload.status_text;
      setLocked(false);
      appendMessage("bot", payload.reply, payload);
      input.value = "";
      input.focus();
    }

    async function sendMessage(prefill = null) {
      if (state.locked) {
        return;
      }

      const text = (prefill ?? input.value).trim();
      if (!text) {
        return;
      }

      const userBubble = appendMessage("user", text);
      input.value = "";

      const payload = await requestJson("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });

      statusText.textContent = payload.status_text;
      appendEmotionMeta(userBubble, payload);
      appendMessage("bot", payload.reply, payload);

      if (payload.quit) {
        setLocked(true);
      } else {
        input.focus();
      }
    }

    composer.addEventListener("submit", async (event) => {
      event.preventDefault();
      await sendMessage();
    });

    input.addEventListener("keydown", async (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        await sendMessage();
      }
    });

    resetButton.addEventListener("click", resetChat);

    document.querySelectorAll(".chip").forEach((chip) => {
      chip.addEventListener("click", async () => {
        input.value = chip.dataset.text;
        await sendMessage(chip.dataset.text);
      });
    });

    resetChat().catch((error) => {
      statusText.textContent = "စနစ်ချိတ်ဆက်ရာမှာ ပြဿနာရှိနေပါတယ်။";
      appendMessage("bot", `Error: ${error.message}`);
    });
  </script>
</body>
</html>
"""


class ChatHandler(BaseHTTPRequestHandler):
    backend: ChatBackend

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
    parser = argparse.ArgumentParser(description="Standalone Burmese chat UI for Hybrid ELIZA")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--lang", default="mm", choices=["en", "mm"])
    parser.add_argument("--model_path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    backend = ChatBackend(lang=args.lang, model_path=args.model_path)
    ChatHandler.backend = backend
    server = ThreadingHTTPServer((args.host, args.port), ChatHandler)

    print(f"Hybrid ELIZA chat UI running at http://{args.host}:{args.port}")
    print(backend.status_text())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
