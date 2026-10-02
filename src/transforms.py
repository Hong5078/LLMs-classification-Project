"""Text transforms used by the RQ4 ablations. Each takes and returns a response string."""
import re
import unicodedata

# Names of the six families and their developers (and common product names).
SELF_ID_RE = re.compile(
    r"\b(chatgpt|gpt-?\d(?:\.\d)?\w*|openai|gemini|bard|google|claude|anthropic|llama|meta ai|"
    r"deepseek|qwen|tongyi|alibaba)\b", re.IGNORECASE)

TYPOGRAPHY_MAP = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "−": "-",
    "…": "...", " ": " ", "•": "-", "�": "",
})

_PUNCT_RE = re.compile(r"[^\w\s]|_")


def identity(text):
    return text


def strip_markdown(text):
    """Remove markdown/formatting syntax but keep the words inside it."""
    text = re.sub(r"```.*?```", lambda m: m.group(0).strip("`"), text, flags=re.DOTALL)
    text = text.replace("`", "")
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)            # headers
    text = re.sub(r"(?m)^\s*[-*+•]\s+", "", text)           # bullets
    text = re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)               # numbered lists at line start
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)              # bold
    text = re.sub(r"(?<!\w)[*_](.+?)[*_](?!\w)", r"\1", text)    # italics
    text = re.sub(r"(?m)^\s*\|?[\s:|-]{3,}\|?\s*$", "", text)    # table rules
    text = text.replace("|", " ").replace("*", "").replace("#", "")
    return text


def normalize_whitespace(text):
    """Replace every run of whitespace (including newlines and tabs) with one space."""
    return re.sub(r"\s+", " ", text).strip()


def lower_nopunct(text):
    """Lowercase and delete all punctuation and symbols; only words, digits and spaces remain."""
    return re.sub(r"\s+", " ", _PUNCT_RE.sub(" ", text.lower())).strip()


def mask_self_id(text):
    return SELF_ID_RE.sub("<name>", text)


def normalize_typography(text):
    """Map curly quotes, long dashes, ellipsis characters and non-breaking spaces to plain ASCII forms."""
    return unicodedata.normalize("NFKC", text.translate(TYPOGRAPHY_MAP))


TRANSFORMS = {
    "none": identity,
    "strip_markdown": strip_markdown,
    "normalize_whitespace": normalize_whitespace,
    "lower_nopunct": lower_nopunct,
    "mask_self_id": mask_self_id,
    "normalize_typography": normalize_typography,
}
