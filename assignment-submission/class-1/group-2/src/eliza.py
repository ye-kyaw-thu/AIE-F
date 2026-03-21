"""ELIZA-style rule engine: normalization, reflection, and keyword responses."""

import copy
import random
import re

from src.eliza_rules import SCRIPTS
from src.preprocessing import MYANMAR_TOKEN_RE


class Eliza:
    # function to initialize script and sort keyword rules by priority (highest first)
    def __init__(self, language="mm"):
        if language not in SCRIPTS:
            raise ValueError(f"Unsupported language: {language}")
        self.script = copy.deepcopy(SCRIPTS[language])
        self.language = language
        self.script["keywords"] = sorted(
            self.script["keywords"], key=lambda x: x[2], reverse=True
        )

    # function to normalize text for rule matching (lowercase + punctuation cleanup)
    def normalize_text(self, text):
        text = str(text).strip().lower()
        text = re.sub(r"[၊။!?,;:\"'()\[\]{}]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # function to tokenize for reflection only (no char n-grams; base tokens for posts lookup)
    def tokenize_for_rules(self, text: str):
        text = self.normalize_text(text)
        if not text:
            return []

        if self.language == "mm":
            if " " in text:
                return [tok for tok in text.split() if tok]
            return MYANMAR_TOKEN_RE.findall(text)

        return text.split()

    # function to apply posts substitutions for captured fragments
    def reflect(self, fragment):
        tokens = self.tokenize_for_rules(fragment)
        reflected = [self.script["posts"].get(tok, tok) for tok in tokens]
        return " ".join(reflected).strip()

    # function to match keywords, apply pres, and return a response
    def rule_respond(self, text):
        text = self.normalize_text(text)
        for src, dst in self.script["pres"].items():
            text = text.replace(src, dst)

        for pattern, responses, _rank in self.script["keywords"]:
            match = re.search(pattern, text)
            if match:
                response = random.choice(responses)
                fragments = [
                    self.reflect(group) for group in match.groups() if group and group.strip()
                ]
                return response.format(*fragments) if fragments else response

        return "ဆက်ပြောပြပါ။" if self.language == "mm" else "Please continue."

    # function to run rule-based reply (alias for chat UIs)
    def respond(self, text):
        return self.rule_respond(text)

    # function to detect quit phrases after normalization
    def is_quit(self, text: str) -> bool:
        return self.normalize_text(text) in self.script["quits"]
