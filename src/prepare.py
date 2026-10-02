"""Build the six-model subset of TXD-22, clean it minimally, label question types and make the split.

Usage: python -m src.prepare
"""
import hashlib
import re

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import RESULTS, ROOT, load_config, write_json

# Spreadsheet error strings that were exported in place of a response.
SPREADSHEET_ERRORS = {"#NAME?", "#VALUE!", "#REF!", "#N/A", "#DIV/0!", "#NULL!", "#NUM!"}

# Question-type rules for RQ3 (applied to the lowercased prompt).
DEFINITIONAL_STARTS = {"what", "define", "who", "which", "name", "state", "when", "where", "list", "identify"}
EXPLANATORY_STARTS = {"explain", "how", "describe", "discuss", "analyze", "analyse", "why", "compare",
                      "differentiate", "distinguish", "evaluate", "examine", "illustrate", "elaborate"}
EXPLANATORY_ANYWHERE = re.compile(
    r"\b(explain|how|describe|discuss|analy[sz]e|why|compare|differentiate|distinguish|evaluate|examine|"
    r"illustrate|elaborate)\b")


def find_raw_csv(cfg):
    for pattern in cfg["data"]["raw_globs"]:
        hits = sorted(ROOT.glob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError("TXD-22 CSV not found; see README.md for where to put it.")


def load_raw(path):
    # The file is Windows-1252 (curly quotes and dashes are single bytes 0x91-0x97), not UTF-8.
    # The five byte values that cp1252 leaves undefined become U+FFFD.
    df = pd.read_csv(path, encoding="cp1252", encoding_errors="replace")
    df = df.rename(columns={"Source": "model", "Question": "prompt", "Answer": "response"})
    df.insert(0, "row_id", range(len(df)))
    return df


def prompt_key(prompt):
    """Normalised prompt used only for grouping, so variants of one question share a group.

    The same question is stored with different apostrophe encodings for different models
    (a real curly apostrophe for some rows, a garbled multi-byte sequence for others), so all
    non-ASCII characters and apostrophes are deleted before comparing.
    """
    s = str(prompt).encode("ascii", "ignore").decode().replace("'", "").lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def question_type(prompt):
    s = prompt_key(prompt)
    first = s.split(" ", 1)[0] if s else ""
    if first in EXPLANATORY_STARTS:
        return "explanatory"
    if first in DEFINITIONAL_STARTS:
        # "What is X? Discuss its scope." asks for both, so it is kept out of the two clean groups.
        return "mixed" if EXPLANATORY_ANYWHERE.search(s) else "definitional"
    return "other"


def clean(df):
    """Drop missing/empty responses, spreadsheet-error placeholders and exact duplicates. Text is not altered."""
    log = {"rows_before": df.groupby("model").size().to_dict()}
    resp = df["response"]
    missing = resp.isna()
    empty = ~missing & (resp.astype(str).str.strip() == "")
    sheet_err = ~missing & resp.astype(str).str.strip().isin(SPREADSHEET_ERRORS)
    log["missing"] = df[missing].groupby("model").size().to_dict()
    log["empty"] = df[empty].groupby("model").size().to_dict()
    log["spreadsheet_error"] = df[sheet_err].groupby("model").size().to_dict()
    df = df[~(missing | empty | sheet_err)].copy()

    dup_rows = df.duplicated(["model", "prompt", "response"], keep="first")
    log["duplicate_rows"] = df[dup_rows].groupby("model").size().to_dict()
    df = df[~dup_rows]
    dup_resp = df.duplicated(["model", "response"], keep="first")
    log["duplicate_responses_same_model"] = df[dup_resp].groupby("model").size().to_dict()
    df = df[~dup_resp]
    log["duplicate_responses_across_models"] = int(df.duplicated(["response"], keep=False).sum())
    log["rows_after"] = df.groupby("model").size().to_dict()
    log["total_after"] = int(len(df))
    return df.reset_index(drop=True), log


def make_split(df, cfg):
    sp = cfg["split"]
    sgkf = StratifiedGroupKFold(n_splits=sp["n_folds"], shuffle=True, random_state=cfg["seed"])
    fold = pd.Series(-1, index=df.index)
    for k, (_, idx) in enumerate(sgkf.split(df, df["model"], df["group"])):
        fold.iloc[idx] = k
    n_test, n_val = sp["test_folds"], sp["val_folds"]
    first_test = sp["n_folds"] - n_test
    first_val = first_test - n_val
    split = fold.map(lambda k: "test" if k >= first_test else ("val" if k >= first_val else "train"))
    return fold, split


def main():
    cfg = load_config()
    raw_path = find_raw_csv(cfg)
    raw = load_raw(raw_path)
    all_sources = raw.groupby("model").size().to_dict()

    df = raw[raw["model"].isin(cfg["models"])].copy()
    df, log = clean(df)
    log["raw_file"] = raw_path.name
    log["raw_rows"] = int(len(raw))
    log["raw_sources"] = all_sources

    df["group"] = df["prompt"].map(lambda p: hashlib.md5(prompt_key(p).encode()).hexdigest()[:12])
    df["qtype"] = df["prompt"].map(question_type)
    df["fold"], df["split"] = make_split(df, cfg)

    # No prompt group may appear in more than one split.
    assert df.groupby("group")["split"].nunique().max() == 1

    out = ROOT / cfg["data"]["subset_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")

    splits = ROOT / cfg["data"]["splits_path"]
    splits.parent.mkdir(parents=True, exist_ok=True)
    df[["row_id", "model", "group", "qtype", "fold", "split"]].to_csv(splits, index=False)

    log["split_rows"] = df.groupby(["split", "model"]).size().unstack(0).to_dict()
    log["split_groups"] = df.groupby("split")["group"].nunique().to_dict()
    log["n_groups"] = int(df["group"].nunique())
    write_json(log, RESULTS / "audit" / "cleaning_log.json")
    print(f"subset: {len(df)} rows -> {out}")
    print(df.groupby(["split", "model"]).size().unstack(0))


if __name__ == "__main__":
    main()
