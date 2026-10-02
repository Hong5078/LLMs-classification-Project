"""Stylometric features of a response (RQ4)."""
import re

WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])[\"'”’)\]]*\s+")
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿⬀-⯿]")

# Feature groups, used to test which kind of signal carries the classification.
GROUPS = {
    "length": ["n_chars", "n_words", "n_sentences", "words_per_sentence"],
    "lexical": ["avg_word_len", "ttr", "mattr50", "pron_1p_rate", "pron_2p_rate", "upper_rate", "digit_rate"],
    "punctuation": ["comma_rate", "period_rate", "semicolon_rate", "colon_rate", "question_rate",
                    "exclaim_rate", "paren_rate", "hyphen_rate", "slash_rate"],
    "typography": ["curly_apos_rate", "straight_apos_rate", "curly_dquote_rate", "straight_dquote_rate",
                   "en_dash_rate", "em_dash_rate", "non_ascii_rate"],
    "formatting": ["md_headers", "md_bullets", "md_numbered", "inline_numbered", "md_bold", "md_code",
                   "md_table_rows", "latex_rate", "emoji_count", "newline_rate", "ends_terminal"],
}
FEATURES = [f for g in GROUPS.values() for f in g]


def mattr(words, window=50):
    """Moving-average type-token ratio: mean TTR over all windows of fixed size, so it does not fall with length."""
    if not words:
        return 0.0
    if len(words) <= window:
        return len(set(words)) / len(words)
    counts, total = {}, 0.0
    for w in words[:window]:
        counts[w] = counts.get(w, 0) + 1
    total += len(counts) / window
    for i in range(window, len(words)):
        old, new = words[i - window], words[i]
        counts[old] -= 1
        if counts[old] == 0:
            del counts[old]
        counts[new] = counts.get(new, 0) + 1
        total += len(counts) / window
    return total / (len(words) - window + 1)


def sentences(text):
    return [s for s in SENT_SPLIT_RE.split(text.strip()) if s]


def extract_features(text):
    words = WORD_RE.findall(text)
    lower = [w.lower() for w in words]
    n_words = max(len(words), 1)
    n_chars = max(len(text), 1)
    sents = sentences(text)
    letters = [c for c in text if c.isalpha()]
    per100 = lambda n: 100.0 * n / n_words  # noqa: E731

    return {
        "n_chars": len(text),
        "n_words": len(words),
        "n_sentences": len(sents),
        "words_per_sentence": len(words) / max(len(sents), 1),
        "avg_word_len": sum(len(w) for w in words) / n_words,
        "ttr": len(set(lower)) / n_words,
        "mattr50": mattr(lower, 50),
        "pron_1p_rate": per100(sum(w in {"i", "we", "our", "us", "my"} for w in lower)),
        "pron_2p_rate": per100(sum(w in {"you", "your", "yours"} for w in lower)),
        "upper_rate": sum(c.isupper() for c in letters) / max(len(letters), 1),
        "digit_rate": sum(c.isdigit() for c in text) / n_chars,
        "comma_rate": per100(text.count(",")),
        "period_rate": per100(text.count(".")),
        "semicolon_rate": per100(text.count(";")),
        "colon_rate": per100(text.count(":")),
        "question_rate": per100(text.count("?")),
        "exclaim_rate": per100(text.count("!")),
        "paren_rate": per100(text.count("(")),
        "hyphen_rate": per100(text.count("-")),
        "slash_rate": per100(text.count("/")),
        "curly_apos_rate": per100(text.count("’") + text.count("‘")),
        "straight_apos_rate": per100(text.count("'")),
        "curly_dquote_rate": per100(text.count("“") + text.count("”")),
        "straight_dquote_rate": per100(text.count('"')),
        "en_dash_rate": per100(text.count("–")),
        "em_dash_rate": per100(text.count("—")),
        "non_ascii_rate": sum(ord(c) > 127 for c in text) / n_chars,
        "md_headers": len(re.findall(r"(?m)^\s{0,3}#{1,6}\s", text)),
        "md_bullets": len(re.findall(r"(?m)^\s*[-*+•]\s+", text)),
        "md_numbered": len(re.findall(r"(?m)^\s*\d+[.)]\s+", text)),
        "inline_numbered": len(re.findall(r"(?<!^)(?<=\s)\(?\d{1,2}[.)]\s", text)),
        "md_bold": len(re.findall(r"\*\*[^*\n]+\*\*", text)),
        "md_code": text.count("`"),
        "md_table_rows": len(re.findall(r"(?m)^\s*\|.*\|\s*$", text)),
        "latex_rate": per100(text.count("$") + text.count("\\(") + text.count("\\[")),
        "emoji_count": len(EMOJI_RE.findall(text)),
        "newline_rate": per100(text.count("\n")),
        "ends_terminal": float(bool(re.search(r"[.!?][\"'”’)\]]*\s*\Z", text))),
    }


def opening_phrase(text, n=3):
    return " ".join(w.lower() for w in WORD_RE.findall(text)[:n])


def closing_sentence_start(text, n=2):
    """First n words of the final sentence (e.g. 'in conclusion', 'overall the')."""
    sents = sentences(text)
    return " ".join(w.lower() for w in WORD_RE.findall(sents[-1])[:n]) if sents else ""
