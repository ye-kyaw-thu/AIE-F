'''
Syllable Grapheme to Phoneme Conversion
Author: Ye Kyaw Thu, Thura Aung

from grapheme2phonemes import G2PConverter

# Initialize converter for G2P (grapheme to phoneme) or G2IPA (grapheme to IPA)
converter = G2PConverter(model="G2P")  # or "G2IPA"

# Example sentence (Myanmar script)
sentence = "မြန်မာနိုင်ငံ"

# Convert to phonemes
output = converter.convert(sentence)

# Print output
print("Input sentence:", sentence)
print("G2P Output:")
for syl, phoneme in output:
    print(f"{syl}|{phoneme}", end=" ")
'''
import os
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import List, Tuple
from cached_path import cached_path
import re 

tf.get_logger().setLevel('ERROR')

# Huggingface repo URL base for neural G2P models
HF_REPO = "hf://LULab/myNLP-Transliteration-models/Models"

# Cached paths for G2P models
G2P_MODEL_PATH = cached_path(f"{HF_REPO}/G2P/myG2P-bilstm.h5")
G2P_TAG_PATH = cached_path(f"{HF_REPO}/G2P/tag_map_g2p.json")
G2P_VOCAB_PATH = cached_path(f"{HF_REPO}/G2P/vocab_g2p.json")

# Cached paths for G2IPA models
G2IPA_MODEL_PATH = cached_path(f"{HF_REPO}/G2IPA/myG2IPA-bilstm.h5")
G2IPA_TAG_PATH = cached_path(f"{HF_REPO}/G2IPA/tag_map_g2ipa.json")
G2IPA_VOCAB_PATH = cached_path(f"{HF_REPO}/G2IPA/vocab_g2ipa.json")

class SyllableTokenizer:
    """
    Syllable Tokenizer using Sylbreak for Myanmar language.
    Author: Ye Kyaw Thu
    Link: https://github.com/ye-kyaw-thu/sylbreak
    tokenizer = SyllableTokenizer()
    syllables = tokenizer.tokenize("မြန်မာနိုင်ငံ။")
    print(syllables)
    # ['မြန်', 'မာ', 'နိုင်', 'ငံ', '။']
    """

    def __init__(self) -> None:
        self._my_consonant = r"က-အ"
        self._en_char = r"a-zA-Z0-9"
        self._other_char = r"ဣဤဥဦဧဩဪဿ၌၍၏၀-၉၊။!-/:-@\[-`{-~\s"
        self._ss_symbol = "္"
        self._a_that = "်"
        pattern = (
            rf"((?<!.{self._ss_symbol})["  # negative‑lookbehind for stacked conso.
            rf"{self._my_consonant}"          # any Burmese consonant
            rf"](?![{self._a_that}{self._ss_symbol}])"  # not followed by virama
            rf"|[{self._en_char}{self._other_char}])"
        )
        self._break_pattern: re.Pattern[str] = re.compile(pattern)

    # ------------------------------------------------------------------
    def tokenize(self, raw_text: str) -> List[str]:
        """Return a list of syllables for *raw_text*."""
        lined_text = re.sub(self._break_pattern, r" \1", raw_text)
        return lined_text.split()

class G2PModel:
    def __init__(self, model_type: str, model_path: str, tag_path: str, vocab_path: str) -> None:
        self.model_type = model_type
        self.model_path = model_path
        self.tag_path = tag_path
        self.vocab_path = vocab_path
        self.vocab = None
        self.tag_map = None
        self.model = None
        self.load_model()

    def load_model(self) -> None:
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

        with open(self.tag_path, 'r', encoding='utf-8') as f:
            self.tag_map = json.load(f)

        self.model = load_model(str(self.model_path))

    def convert(self) -> None:
        pass

class G2PConverter:
    def __init__(self, model: str = "G2P") -> None:
        model = model.upper()
        if model == "G2P":
            self.g2p_model = G2PModel(model, G2P_MODEL_PATH, G2P_TAG_PATH, G2P_VOCAB_PATH)
        elif model == "G2IPA":
            self.g2p_model = G2PModel(model, G2IPA_MODEL_PATH, G2IPA_TAG_PATH, G2IPA_VOCAB_PATH)
        else:
            raise ValueError("Invalid choice. Available models: G2P, G2IPA")

    def convert(self, raw_sentence: str) -> List[Tuple[str, str]]:
        if self.g2p_model.model is None:
            raise ValueError("Model is not loaded. Load the model first.")
        try:
            tokenizer = SyllableTokenizer()
            tokens = tokenizer.tokenize(raw_sentence)

            burmese_pattern = re.compile(rf"^[{tokenizer._my_consonant}].*")
            punc_set = set("၊။.,-!?\"'()[]{}:;")

            indexed_tokens = []
            token_types = []
            for t in tokens:
                if all(char in punc_set for char in t):
                    token_types.append("punc")
                    indexed_tokens.append(None)
                elif burmese_pattern.match(t):
                    token_types.append("burmese")
                    indexed_tokens.append(self.g2p_model.vocab.get(t, self.g2p_model.vocab['<UNK>']))
                else:
                    token_types.append("non_burmese")
                    indexed_tokens.append(None)

            burmese_indices = [i for i, typ in enumerate(token_types) if typ == "burmese"]
            burmese_indexed = [indexed_tokens[i] for i in burmese_indices]
            padded_sequence = pad_sequences([burmese_indexed], padding='post')
            predictions = self.g2p_model.model.predict(padded_sequence, verbose=0)
            predicted_tags = np.argmax(predictions, axis=-1)[0]
            reverse_tag_map = {v: k for k, v in self.g2p_model.tag_map.items()}

            result = []
            burmese_tag_idx = 0
            for i, t in enumerate(tokens):
                if token_types[i] == "burmese":
                    phoneme = reverse_tag_map[predicted_tags[burmese_tag_idx]]
                    result.append((t, phoneme))
                    burmese_tag_idx += 1
                elif token_types[i] == "non_burmese":
                    result.append((t, "x"))
                elif token_types[i] == "punc":
                    result.append((t, "punc"))
            return result
        except tf.errors.InvalidArgumentError as e:
            print("Error occurred:", e)
            return []

if __name__ == "__main__":
    converter = G2PConverter(model="G2IPA")  

    sentence = "အင် တင် တင် လုပ် နေ တယ် ။"

    output = converter.convert(sentence)

    print("Input sentence:", sentence)
    print("G2P Output:")
    for syl, phoneme in output:
        print(f"{syl}|{phoneme}", end=" ")
    print() 

