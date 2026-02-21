"""Text processing utilities for Russian medical text."""

import re

import pymorphy3

_morph = None


def get_morph():
    global _morph
    if _morph is None:
        _morph = pymorphy3.MorphAnalyzer()
    return _morph


def lemmatize_text(text: str) -> list[str]:
    """Lemmatize Russian text using pymorphy3."""
    morph = get_morph()
    # Simple tokenization: split on non-word chars
    words = re.findall(r"[а-яёА-ЯЁa-zA-Z]+", text.lower())
    lemmas = []
    for word in words:
        parsed = morph.parse(word)
        if parsed:
            lemmas.append(parsed[0].normal_form)
        else:
            lemmas.append(word)
    return lemmas


def compute_symptom_overlap(query: str, symptoms: list[str]) -> float:
    """Compute fraction of protocol symptoms found in query text.

    Uses lemmatization for better matching.
    """
    if not symptoms:
        return 0.0

    query_lemmas = set(lemmatize_text(query))
    matches = 0
    for symptom in symptoms:
        symptom_lemmas = set(lemmatize_text(symptom))
        # A symptom matches if at least half its lemmas are in the query
        if not symptom_lemmas:
            continue
        overlap = len(symptom_lemmas & query_lemmas) / len(symptom_lemmas)
        if overlap >= 0.5:
            matches += 1

    return matches / len(symptoms)
