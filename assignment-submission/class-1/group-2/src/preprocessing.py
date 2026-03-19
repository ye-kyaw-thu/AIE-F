"""This module provides Burmese text normalization, tokenization, and stopword removal.

This module depends on:
- src/rabbit.py
"""

import re
import myanmartools
from mmdt_tokenizer import MyanmarTokenizer
from src.rabbit import Rabbit


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
    - Sentence tokenization
    - Word tokenization
    - Stopword removal
    """

    # function to initialize processor components
    def __init__(self, stopwords=None):
        self.stopwords = stopwords if stopwords else set()

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


    # function to tokenize text into words
    def tokenize(self, text: str):
        return self.word_tokenizer.word_tokenize(text)[0]


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
        2. Clean punctuation
        3. Tokenize
        4. Remove stopwords
        """
        # step 1: normalize
        text = self.normalize_text(text)

        # step 2: clean punctuation
        text = clean_punctuation(text)

        # step 3: tokenize
        tokens = self.tokenize(text)

        # step 4: stopword removal (optional)
        if remove_stopwords_flag:
            tokens = self.remove_stopwords(tokens)

        return tokens