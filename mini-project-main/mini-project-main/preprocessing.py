"""Shared NLP preprocessing helpers for resumes and job descriptions."""

import re
import string
from typing import Iterable, List

try:
    import spacy
except ImportError:  # Streamlit can still show a helpful dependency message.
    spacy = None


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "or", "that", "the", "to",
    "was", "were", "will", "with", "you", "your", "this", "their", "they",
    "we", "our", "i", "me", "my",
}


def load_spacy_model():
    """Load spaCy if available; fall back to a blank English pipeline."""
    if spacy is None:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return spacy.blank("en")


NLP = load_spacy_model()


def normalize_text(text: str) -> str:
    """Normalize whitespace and remove invisible control characters."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation, remove stopwords, and lemmatize tokens."""
    text = normalize_text(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = tokenize(text)

    cleaned_tokens = []
    if NLP is not None:
        doc = NLP(" ".join(tokens))
        for token in doc:
            lemma = token.lemma_.lower().strip() if token.lemma_ else token.text
            if lemma and lemma not in STOPWORDS and len(lemma) > 1:
                cleaned_tokens.append(lemma)
    else:
        cleaned_tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 1]

    return " ".join(cleaned_tokens)


def tokenize(text: str) -> List[str]:
    """Extract word-like tokens while preserving technical tokens such as c++."""
    return re.findall(r"[a-zA-Z][a-zA-Z0-9+#.\-]*", text)


def keyword_overlap(source_keywords: Iterable[str], target_keywords: Iterable[str]) -> List[str]:
    """Return keywords from target that do not appear in source."""
    source = {item.lower() for item in source_keywords}
    return sorted({item for item in target_keywords if item.lower() not in source})

