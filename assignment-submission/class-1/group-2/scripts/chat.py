#!/usr/bin/env python3

"""This module provides interactive chat.

This module depends on:
- scripts/eval.py
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, TypedDict

from scripts.eval import load_inference_bundle, predict_texts
from src.eliza import Eliza


# single defaults for train/eval/chat entry points (project root = parent of scripts/)
DEFAULT_CHECKPOINT_PATH = "./checkpoints/bilstm_smaller_params.pth"
DEFAULT_STOPWORDS_PATH = "./data/stopwords.txt"


# function to resolve a path relative to project root so terminal/streamlit/custom_ui load the same files
def resolve_project_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p.resolve())
    root = Path(__file__).resolve().parent.parent
    return str((root / p).resolve())


# condensed palette/fonts from eliza_experiments/burmese_chat_ui.py for streamlit injection
STREAMLIT_CHAT_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Padauk&display=swap');
    html, body, [class*="css"]  {
        font-family: "Padauk", "Noto Sans Myanmar", "Myanmar Text", "Segoe UI", sans-serif;
    }
    .stApp {
        background: linear-gradient(180deg, #f8d08a 0%, #f6efe5 100%);
    }
    [data-testid="stChatMessage"] {
        background: rgba(255, 250, 242, 0.95);
        border-radius: 12px;
        border: 1px solid rgba(110, 65, 24, 0.12);
    }
</style>
"""


class ChatTurnReply(TypedDict):
    kind: str
    emotion_label: str
    emotion_score: float
    eliza_reply: str


class ChatTurnQuit(TypedDict):
    kind: str
    final: str


class ChatTurnEmpty(TypedDict):
    kind: str


ChatTurnResult = ChatTurnReply | ChatTurnQuit | ChatTurnEmpty


# function to load model bundle and eliza for one session
def load_chat_context(
    checkpoint_path: str,
    language: str = "mm",
) -> dict[str, Any]:
    if language not in ("mm", "en"):
        language = "mm"
    (
        model,
        word2id,
        id2label,
        max_len,
        use_char_ngrams,
        ngram_min,
        ngram_max,
    ) = load_inference_bundle(checkpoint_path)
    eliza = Eliza(language=language)
    return {
        "model": model,
        "word2id": word2id,
        "id2label": id2label,
        "max_len": max_len,
        "use_char_ngrams": use_char_ngrams,
        "ngram_min": ngram_min,
        "ngram_max": ngram_max,
        "eliza": eliza,
    }


# function to run one user turn (terminal and streamlit both use this)
def chat_turn(
    ctx: dict[str, Any],
    text: str,
    stopwords_path: str,
) -> ChatTurnResult:
    text = str(text).strip()
    if not text:
        return ChatTurnEmpty(kind="empty")

    eliza: Eliza = ctx["eliza"]
    if eliza.is_quit(text):
        return ChatTurnQuit(
            kind="quit",
            final=random.choice(eliza.script["finals"]),
        )

    pred_ids, pred_labels, pred_scores = predict_texts(
        model=ctx["model"],
        word2id=ctx["word2id"],
        id2label=ctx["id2label"],
        max_len=ctx["max_len"],
        texts=[text],
        stopwords_path=stopwords_path,
        use_char_ngrams=ctx["use_char_ngrams"],
        ngram_min=ctx["ngram_min"],
        ngram_max=ctx["ngram_max"],
    )

    return ChatTurnReply(
        kind="reply",
        emotion_label=pred_labels[0],
        emotion_score=float(pred_scores[0]),
        eliza_reply=eliza.respond(text),
    )


# function to run interactive chat inference (see group2-hybrid-eliza.py --mode chat)
def run_chat(
    checkpoint_path: str,
    stopwords_path: str = DEFAULT_STOPWORDS_PATH,
    language: str = "mm",
):
    checkpoint_path = resolve_project_path(checkpoint_path)
    stopwords_path = resolve_project_path(stopwords_path)
    ctx = load_chat_context(checkpoint_path, language=language)

    while True:
        try:
            text = input("You: ").strip()
        except KeyboardInterrupt:
            break

        out = chat_turn(ctx, text, stopwords_path)
        if out["kind"] == "empty":
            continue
        if out["kind"] == "quit":
            print(f"Eliza: {out['final']}")
            break

        print(f"Predicted emotion: {out['emotion_label']} ({out['emotion_score']:.2%})")
        print(f"Eliza: {out['eliza_reply']}")


# function to start streamlit app with env pointing at checkpoint (blocks until streamlit exits)
def launch_streamlit_ui(
    checkpoint_path: str,
    stopwords_path: str = DEFAULT_STOPWORDS_PATH,
    language: str = "mm",
) -> None:
    root = Path(__file__).resolve().parent.parent
    app = root / "scripts" / "streamlit_chatter.py"
    env = os.environ.copy()
    env["CHAT_CHECKPOINT"] = resolve_project_path(checkpoint_path)
    env["CHAT_STOPWORDS"] = resolve_project_path(stopwords_path)
    env["CHAT_LANGUAGE"] = language
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app), "--server.address", "localhost"],
        cwd=str(root),
        env=env,
        check=False,
    )


# function to start custom html/http ui (eliza_experiments/burmese_chat_ui.py page + modular backend)
def launch_custom_ui(
    checkpoint_path: str,
    stopwords_path: str = DEFAULT_STOPWORDS_PATH,
    language: str = "mm",
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    root = Path(__file__).resolve().parent.parent
    web = root / "scripts" / "custom_ui_chatter.py"
    subprocess.run(
        [
            sys.executable,
            str(web),
            "--host",
            host,
            "--port",
            str(port),
            "--lang",
            language,
            "--checkpoint_path",
            resolve_project_path(checkpoint_path),
            "--stopwords_path",
            resolve_project_path(stopwords_path),
        ],
        cwd=str(root),
        check=False,
    )


# function to run chat with default values
def main():
    run_chat(checkpoint_path=DEFAULT_CHECKPOINT_PATH, language="mm")

# run the script
if __name__ == "__main__":
    main()