"""Load the prepared subset and turn rows into model inputs for a given condition."""
import numpy as np
import pandas as pd

from .config import ROOT
from .tokenizer import INP, OUT, TOKEN_RE, tokenize
from .transforms import TRANSFORMS

CONDITIONS = ("input", "output", "input+output")


def load_subset(cfg):
    path = ROOT / cfg["data"]["subset_path"]
    df = pd.read_csv(path, encoding="utf-8", keep_default_na=False)
    df["label"] = df["model"].map({m: i for i, m in enumerate(cfg["models"])}).astype(int)
    return df


def truncate_to_tokens(text, n):
    """Cut a string right after its n-th token."""
    end = None
    for i, m in enumerate(TOKEN_RE.finditer(text)):
        if i == n:
            return text[:end]
        end = m.end()
    return text


def build_inputs(df, cfg, condition="output", transform="none", out_len=None, truncate=None):
    """Return (token_lists, strings) for the rows of df.

    transform is an ablation applied to the response; truncate keeps only the first N response tokens.
    The strings are what the TF-IDF baseline sees and the token lists are what the neural models see.
    """
    assert condition in CONDITIONS
    fn = TRANSFORMS[transform]
    out_len = out_len or cfg["lengths"]["output"]
    if truncate:
        out_len = min(out_len, truncate)

    responses = [truncate_to_tokens(fn(r), out_len) for r in df["response"]]
    if condition == "output":
        return [tokenize(r) for r in responses], responses

    prompts = list(df["prompt"])
    if condition == "input":
        n = cfg["lengths"]["input"]
        return [tokenize(p)[:n] for p in prompts], prompts

    n = cfg["lengths"]["prompt_in_concat"]
    tokens = [[INP] + tokenize(p)[:n] + [OUT] + tokenize(r) for p, r in zip(prompts, responses)]
    strings = [f"{INP} {p} {OUT} {r}" for p, r in zip(prompts, responses)]
    return tokens, strings


def make_data(df, cfg, condition="output", transform="none", truncate=None, mask=None):
    from .config import output_max_len

    tokens, strings = build_inputs(df, cfg, condition, transform, output_max_len(cfg), truncate)
    return by_split(df, tokens, strings, mask)


def by_split(df, tokens, strings, mask=None):
    """Split parallel lists into {'train'|'val'|'test': {...}} using the saved split column."""
    out = {}
    split = df["split"].to_numpy()
    keep = np.ones(len(df), dtype=bool) if mask is None else np.asarray(mask)
    for name in ("train", "val", "test"):
        idx = np.flatnonzero((split == name) & keep)
        out[name] = {
            "tokens": [tokens[i] for i in idx],
            "strings": [strings[i] for i in idx],
            "labels": df["label"].to_numpy()[idx],
            "row_id": df["row_id"].to_numpy()[idx],
        }
    return out
