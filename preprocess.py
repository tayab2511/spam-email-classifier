"""Shared text preprocessing used by both training and the web app."""
import re

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\S+@\S+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
SPACE_RE = re.compile(r"\s+")


def clean_text(text) -> str:
    """Lowercase, replace URLs/emails with tokens, strip symbols and digits."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()
