"""Text processing utilities for Russian medical text."""

import re
from functools import lru_cache

import pymorphy3

_morph = None


def get_morph():
    global _morph
    if _morph is None:
        _morph = pymorphy3.MorphAnalyzer()
    return _morph


@lru_cache(maxsize=100000)
def _lemmatize_word(word: str) -> str:
    """Lemmatize a single word with caching."""
    morph = get_morph()
    parsed = morph.parse(word)
    return parsed[0].normal_form if parsed else word


def lemmatize_text(text: str) -> list[str]:
    """Lemmatize Russian text using pymorphy3 with word-level caching."""
    words = re.findall(r"[а-яёА-ЯЁa-zA-Z]+", text.lower())
    return [_lemmatize_word(w) for w in words]


# Cache symptom lemmatization (same symptoms repeat across queries)
_symptom_cache: dict[str, set[str]] = {}


def compute_symptom_overlap(query: str, symptoms: list[str]) -> float:
    """Compute fraction of protocol symptoms found in query text."""
    if not symptoms or not query:
        return 0.0

    query_lemmas = set(lemmatize_text(query))
    matches = 0
    for symptom in symptoms:
        if symptom not in _symptom_cache:
            _symptom_cache[symptom] = set(lemmatize_text(symptom))
        symptom_lemmas = _symptom_cache[symptom]
        if not symptom_lemmas:
            continue
        overlap = len(symptom_lemmas & query_lemmas) / len(symptom_lemmas)
        if overlap >= 0.5:
            matches += 1

    return matches / len(symptoms)
