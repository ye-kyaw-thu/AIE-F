"""This module provides Burmese text normalization, tokenization, and stopword removal.

This module depends on:
- src/rabbit.py
"""

import re
import myanmartools
from mmdt_tokenizer import MyanmarTokenizer
from src.rabbit import Rabbit


# splits mixed Burmese + Latin/alphanumeric text into spans before Burmese tokenization
## u1000-u109F: Myanmar unicode block
## uAA60-uAA7F: Myanmar extended unicode block
MYANMAR_TOKEN_RE = re.compile(r"[\u1000-\u109F\uAA60-\uAA7F]+|[a-zA-Z0-9]+")


# function to build character n-gram tokens
def build_char_ngrams(text: str, min_n: int = 2, max_n: int = 3):
    compact = text.replace(" ", "")
    ngrams = []
    for n in range(min_n, max_n + 1):
        if len(compact) < n:
            continue
        for i in range(len(compact) - n + 1):
            ngrams.append(compact[i : i + n])
    return ngrams


# function to load stopwords from file
def load_stopwords(file_path: str):
    """
    Load stopwords from a file (one word per line).
    Returns a set for fast lookup.
    """
    stopwords = set()

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word:
                stopwords.add(word)

    return stopwords


# function to clean punctuation and normalize spaces
def clean_punctuation(text: str) -> str:
    text = re.sub(r"[၊။()!?,.:;\"'“”‘’\-_/\\\[\]{}<>@#$%^&*+=|`~…]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# class to handle text preprocessing
class TextProcessor:
    """
    Handles:
    - Zawgyi to Unicode conversion
    - Lowercasing (for consistent Latin digits/letters in mixed text)
    - Sentence tokenization
    - Word tokenization
    - Char n-grams appended to word tokens
    - Stopword removal
    """

    # function to initialize processor components
    def __init__(
        self,
        stopwords=None,
        use_char_ngrams: bool = False,
        ngram_min: int = 2,
        ngram_max: int = 3,
    ):
        self.stopwords = stopwords if stopwords else set()
        self.use_char_ngrams = use_char_ngrams
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max

        # initialize tokenizer once
        self.word_tokenizer = MyanmarTokenizer()

        # initialize zawgyi detector
        self.detector = myanmartools.ZawgyiDetector()


    # function to detect and convert Zawgyi encoding to Unicode
    def normalize_text(self, text: str) -> str:
        prob = self.detector.get_zawgyi_probability(text)

        if prob > 0.5:
            return Rabbit.zg2uni(text)

        return text


    # function to tell if a regex chunk is Burmese or Latin/alphanumeric
    def _is_myanmar_chunk(self, chunk: str) -> bool:
        if not chunk:
            return False
        o = ord(chunk[0])
        return (0x1000 <= o <= 0x109F) or (0xAA60 <= o <= 0xAA7F)


    # function to tokenize Burmese text into words
    def tokenize(self, text: str):
        chunks = MYANMAR_TOKEN_RE.findall(text)
        if not chunks and text.strip():
            return self.word_tokenizer.word_tokenize(text)[0]

        tokens = []
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            if self._is_myanmar_chunk(chunk):
                tokens.extend(self.word_tokenizer.word_tokenize(chunk)[0])
            else:
                tokens.append(chunk.lower())

        return tokens


    # function to remove stopwords from tokens
    def remove_stopwords(self, tokens):
        if not self.stopwords:
            return tokens

        return [token for token in tokens if token not in self.stopwords]


    # function to run full preprocessing pipeline
    def process(self, text: str, remove_stopwords_flag: bool = True):
        """
        Full preprocessing pipeline:
        1. Normalize encoding
        2. Lowercase
        3. Clean punctuation
        4. Tokenize
        5. Char n-grams appended to token list (optional)
        6. Remove stopwords (optional)
        """
        # step 1: normalize
        text = self.normalize_text(text)

        # step 2: lowercase
        text = text.strip().lower()

        # step 3: clean punctuation
        text = clean_punctuation(text)

        # step 4: tokenize
        tokens = self.tokenize(text)

        # step 5: char n-grams (optional)
        if self.use_char_ngrams:
            tokens = tokens + build_char_ngrams(
                text, min_n=self.ngram_min, max_n=self.ngram_max
            )

        # step 6: stopword removal (optional)
        if remove_stopwords_flag:
            tokens = self.remove_stopwords(tokens)

        return tokens