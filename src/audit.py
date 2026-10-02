"""Data audit of the six-model subset. Writes tables to results/audit/.

Usage: python -m src.audit
"""
import re

import numpy as np
import pandas as pd

from .config import RESULTS, load_config, write_json
from .data import load_subset
from .tokenizer import tokenize

OUT = RESULTS / "audit"

# Product/family names and developer names, checked separately because developer names
# such as "Google" also occur as ordinary topic words.
FAMILY_NAMES = {
    "ChatGPT": r"\b(chatgpt|gpt-?\d\w*)\b",
    "Gemini": r"\b(gemini|bard)\b",
    "Claude": r"\bclaude\b",
    "Llama": r"\bllama\b",
    "DeepSeek": r"\bdeepseek\b",
    "Qwen": r"\b(qwen|tongyi)\b",
}
DEVELOPER_NAMES = {
    "ChatGPT": r"\bopenai\b",
    "Gemini": r"\bgoogle\b",
    "Claude": r"\banthropic\b",
    "Llama": r"\bmeta ai\b|\bMeta(?= Platforms|'s)",
    "DeepSeek": r"\bdeepseek\b",
    "Qwen": r"\balibaba\b",
}
AI_SELF_REFERENCE = (r"\bas an ai\b|\bi am an? (ai|artificial|language|large)\b|\bi'?m an? (ai|language|large)\b|"
                     r"\bas a (large )?language model\b|\bmy training data\b|\bmy knowledge cutoff\b")

TYPOGRAPHY = {
    "curly_apostrophe_or_single_quote": r"[‘’]",
    "curly_double_quote": r"[“”]",
    "straight_apostrophe": r"'",
    "straight_double_quote": r'"',
    "en_dash": r"–",
    "em_dash": r"—",
    "ellipsis_character": r"…",
    "replacement_character": r"�",
    "any_non_ascii": r"[^\x00-\x7f]",
    "newline": r"\n",
    "markdown_bold": r"\*\*",
    "markdown_header": r"(?m)^\s{0,3}#{1,6}\s",
    "backtick": r"`",
    "latex_dollar": r"\$",
    "ends_without_terminal_punctuation": r"[^.!?\"'”’)\]]\s*\Z",
}


def rate_table(df, patterns, flags=re.IGNORECASE):
    rows = {}
    for name, pat in patterns.items():
        rx = re.compile(pat, flags)
        rows[name] = df.groupby("model")["response"].apply(lambda s: s.map(lambda t: bool(rx.search(t))).mean())
    return pd.DataFrame(rows).T


def main():
    cfg = load_config()
    models = cfg["models"]
    df = load_subset(cfg)
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- lengths -------------------------------------------------------------------------
    df["n_chars"] = df["response"].str.len()
    df["n_words"] = df["response"].str.split().str.len()
    df["n_tokens"] = df["response"].map(lambda t: len(tokenize(t)))
    df["prompt_tokens"] = df["prompt"].map(lambda t: len(tokenize(t)))
    df[["row_id", "model", "split", "n_chars", "n_words", "n_tokens"]].to_csv(OUT / "lengths_per_row.csv", index=False)

    q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    length_stats = (df.groupby("model")[["n_chars", "n_words", "n_tokens"]]
                    .describe(percentiles=q).round(1).loc[models])
    length_stats.columns = ["_".join(c) for c in length_stats.columns]
    length_stats.to_csv(OUT / "length_stats.csv")

    tok = df["n_tokens"]
    train_tok = df.loc[df["split"] == "train", "n_tokens"]
    length_summary = {
        "token_percentiles_all": {str(p): float(tok.quantile(p)) for p in q + [0.995, 1.0]},
        "token_percentiles_train": {str(p): float(train_tok.quantile(p)) for p in q + [0.995, 1.0]},
        "coverage_train": {str(L): float((train_tok <= L).mean()) for L in (256, 384, 512, 768)},
        "share_shorter_than": {str(L): float((tok < L).mean()) for L in (64, 128)},
        "share_shorter_than_by_model": {str(L): (df["n_tokens"] < L).groupby(df["model"]).mean().to_dict()
                                        for L in (64, 128)},
        "prompt_tokens": {"mean": float(df["prompt_tokens"].mean()), "max": int(df["prompt_tokens"].max()),
                          "p99": float(df["prompt_tokens"].quantile(0.99))},
        "mean_tokens_by_model": df.groupby("model")["n_tokens"].mean().to_dict(),
        "mean_words_by_model": df.groupby("model")["n_words"].mean().to_dict(),
    }
    write_json(length_summary, OUT / "length_summary.json")

    # ---- prompt overlap ------------------------------------------------------------------
    per_group = df.groupby("group")["model"].nunique()
    overlap = {
        "unique_prompts_raw": int(df["prompt"].nunique()),
        "unique_prompt_groups": int(df["group"].nunique()),
        "groups_by_number_of_models": {str(k): int(v) for k, v in per_group.value_counts().sort_index().items()},
        "share_of_groups_with_all_six": float((per_group == len(models)).mean()),
        "share_of_rows_in_groups_with_all_six": float(df["group"].map(per_group).eq(len(models)).mean()),
        "repeated_model_group_pairs": int(df.duplicated(["model", "group"]).sum()),
        "groups_per_split": df.groupby("split")["group"].nunique().to_dict(),
        "groups_shared_between_splits": int((df.groupby("group")["split"].nunique() > 1).sum()),
    }
    # The same question is not always stored as the same string: apostrophes are encoded differently
    # for different models. Count, per model, the prompts whose exact string the other models do not share.
    n_exact = df.groupby("prompt")["model"].transform("nunique")
    prompt_side = pd.DataFrame({
        "rows": df.groupby("model").size(),
        "exact_string_not_shared_by_all_six": (n_exact < len(models)).groupby(df["model"]).sum(),
        "exact_string_unique_to_this_model": (n_exact == 1).groupby(df["model"]).sum(),
        "contains_non_ascii": df["prompt"].str.contains(r"[^\x00-\x7f]").groupby(df["model"]).sum(),
        "contains_curly_apostrophe": df["prompt"].str.contains("’").groupby(df["model"]).sum(),
        "contains_garbled_apostrophe": df["prompt"].str.contains("â€").groupby(df["model"]).sum(),
    }).loc[models]
    prompt_side.to_csv(OUT / "prompt_encoding_by_model.csv")
    overlap["exact_strings_shared_by_all_six"] = int((df.groupby("prompt")["model"].nunique() == len(models)).sum())
    write_json(overlap, OUT / "prompt_overlap.json")

    # ---- self-identification -------------------------------------------------------------
    fam = rate_table(df, FAMILY_NAMES)          # rows: mentioned family, cols: source model
    dev = rate_table(df, DEVELOPER_NAMES)
    fam.loc[:, models].to_csv(OUT / "family_name_mentions.csv")
    dev.loc[:, models].to_csv(OUT / "developer_name_mentions.csv")
    ai_ref = rate_table(df, {"ai_self_reference": AI_SELF_REFERENCE}).loc["ai_self_reference"]
    any_family = re.compile("|".join(FAMILY_NAMES.values()), re.IGNORECASE)
    self_id = pd.DataFrame({
        "names_own_family": {m: fam.loc[m, m] for m in models},
        "names_own_developer": {m: dev.loc[m, m] for m in models},
        "names_any_family": df.groupby("model")["response"].apply(
            lambda s: s.map(lambda t: bool(any_family.search(t))).mean()),
        "ai_self_reference_phrase": ai_ref,
    }).loc[models]
    self_id.to_csv(OUT / "self_identification.csv")

    # ---- typography / formatting ---------------------------------------------------------
    typo = rate_table(df, TYPOGRAPHY, flags=0).loc[:, models]
    typo.to_csv(OUT / "typography_rates.csv")

    # ---- question types (RQ3 domains) ----------------------------------------------------
    qtype = df.groupby(["qtype", "model"]).size().unstack("model").loc[:, models]
    qtype.to_csv(OUT / "qtype_counts.csv")
    df.groupby(["qtype", "split"])["group"].nunique().unstack("split").to_csv(OUT / "qtype_prompts_per_split.csv")
    prompts = df.drop_duplicates("group")
    spot = pd.concat([g.sample(min(10, len(g)), random_state=cfg["seed"]) for _, g in prompts.groupby("qtype")])
    spot[["qtype", "prompt"]].to_csv(OUT / "qtype_spotcheck.csv", index=False)

    print(length_stats[["n_tokens_mean", "n_tokens_50%", "n_tokens_99%", "n_tokens_max"]])
    print(pd.Series(length_summary["coverage_train"]))
    print(overlap)
    print(prompt_side)
    print(self_id.round(4))
    print(typo.round(3))
    print(qtype)


if __name__ == "__main__":
    main()
