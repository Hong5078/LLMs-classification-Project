"""Regex word-level tokenizer and vocabulary.

Case, punctuation, newlines and markdown symbols are kept because they may be part of a model's
fingerprint. Every punctuation mark is its own token, so "don't" and "don’t" share the word
pieces and differ only in the apostrophe token.
"""
import re
from collections import Counter

PAD, UNK, INP, OUT, MASK = "<pad>", "<unk>", "[INP]", "[OUT]", "<name>"
SPECIALS = [PAD, UNK, INP, OUT, MASK]

# newline | <name> placeholder | letters (any script) | digits | any other single non-space character
TOKEN_RE = re.compile(r"\n|<name>|[^\W\d_]+|\d+|[^\w\s]|_")


def tokenize(text):
    return TOKEN_RE.findall(text)


class Vocab:
    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD]
        self.unk_id = self.stoi[UNK]

    @classmethod
    def build(cls, token_lists, max_size, min_freq):
        counts = Counter(t for toks in token_lists for t in toks)
        for s in SPECIALS:
            counts.pop(s, None)
        kept = [t for t, c in counts.most_common(max_size) if c >= min_freq]
        vocab = cls(SPECIALS + kept)
        vocab.n_types_seen = len(counts)
        return vocab

    def encode(self, tokens):
        unk = self.unk_id
        return [self.stoi.get(t, unk) for t in tokens]

    def __len__(self):
        return len(self.itos)
