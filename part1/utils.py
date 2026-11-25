import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import random
import argparse
import hashlib
import re
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)

TYPO_PROBABILITY = 0.30
TRANSFORM_SEED = 0
TRANSFORM_VERSION = "keyboard-typos-v1"
WORD_PATTERN = re.compile(r"\b[A-Za-z]+(?:['’][A-Za-z]+)*\b")
PROTECTED_WORDS = {
    "no", "not", "nor", "never", "neither", "nothing", "nobody", "none",
    "nowhere", "without", "hardly", "barely", "scarcely", "cannot",
}
KEYBOARD_NEIGHBORS = {
    "q": "wa", "w": "qeas", "e": "wrsd", "r": "etdf", "t": "ryfg",
    "y": "tugh", "u": "yihj", "i": "uojk", "o": "ipkl", "p": "ol",
    "a": "qwsz", "s": "awedxz", "d": "serfcx", "f": "drtgvc",
    "g": "ftyhbv", "h": "gyujnb", "j": "huikmn", "k": "jiolm",
    "l": "kop", "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb",
    "b": "vghn", "n": "bhjm", "m": "njk",
}


def transformation_settings():
    return {
        "version": TRANSFORM_VERSION,
        "seed": TRANSFORM_SEED,
        "word_probability": TYPO_PROBABILITY,
        "minimum_word_length": 4,
        "protected_words": sorted(PROTECTED_WORDS),
        "keyboard_neighbors": KEYBOARD_NEIGHBORS,
    }


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    text = example["text"]
    seed_key = f"{TRANSFORM_VERSION}:{TRANSFORM_SEED}:{text}".encode("utf-8")
    rng = random.Random(int.from_bytes(hashlib.sha256(seed_key).digest(), "big"))

    def replace_word(match):
        word = match.group()
        if len(word) < 4 or not word.isascii() or not word.isalpha():
            return word
        if word.lower() in PROTECTED_WORDS or rng.random() >= TYPO_PROBABILITY:
            return word
        position = rng.randrange(1, len(word) - 1)
        replacement = rng.choice(KEYBOARD_NEIGHBORS[word[position].lower()])
        if word[position].isupper():
            replacement = replacement.upper()
        return word[:position] + replacement + word[position + 1:]

    example = dict(example)
    example["text"] = WORD_PATTERN.sub(replace_word, text)

    ##### YOUR CODE ENDS HERE ######

    return example
