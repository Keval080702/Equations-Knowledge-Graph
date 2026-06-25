"""
Phases 3-6 — Shared Constants and Helper Utilities
====================================================

This module contains constants and small, side-effect-free helper functions
that are used across the enrichment pipeline phases:

* Phase 3 — ``symbol_extraction.py``
* Phase 4 — ``symbol_definition_extraction.py``
* Phase 5 — ``meaning_extraction.py``
* Phase 6 — ``relation_detection.py``

Constants
---------
* ``GREEK_COMMANDS``    — set of LaTeX Greek letter command names (no backslash)
* ``NAMESPACE``         — globally stable symbol definitions (hbar, nabla, partial)
* ``UNICODE_SYMBOLS``   — mapping from Unicode math chars to LaTeX command names
* ``MEANING_RULES``     — keyword-driven fallback meaning phrases
* ``GENERIC_MEANINGS``  — set of boilerplate meanings to reject during deduplication
"""

import re

GREEK_COMMANDS = {
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta",
    "eta", "theta", "vartheta", "iota", "kappa", "lambda", "mu", "nu",
    "xi", "pi", "rho", "varrho", "sigma", "varsigma", "tau", "upsilon",
    "phi", "varphi", "chi", "psi", "omega", "Gamma", "Delta", "Theta",
    "Lambda", "Xi", "Pi", "Sigma", "Phi", "Psi", "Omega",
}

COMMAND_TO_SYMBOL = {
    "hbar": "hbar",
    "nabla": "nabla",
    "partial": "partial",
}

UNICODE_SYMBOLS = {
    "ψ": "psi", "φ": "phi", "α": "alpha", "β": "beta", "γ": "gamma",
    "δ": "delta", "ε": "epsilon", "λ": "lambda", "μ": "mu", "ν": "nu",
    "ω": "omega", "ρ": "rho", "σ": "sigma", "θ": "theta", "π": "pi",
    "Ω": "Omega", "ℏ": "hbar", "∂": "partial", "∇": "nabla",
}

# Only keep symbols whose meaning is effectively global. Domain symbols such
# as H, k, rho, sigma, x/y/z, and j/k/l/m are resolved from local paper context
# in symbol_extraction.py rather than from this dictionary.
NAMESPACE = {
    "hbar": "reduced Planck constant",
    "nabla": "nabla/del differential operator",
    "partial": "partial derivative operator",
}

MEANING_RULES = [
    (("heisenberg", "hamiltonian"), "Heisenberg Hamiltonian"),
    (("extended bose", "hubbard"), "extended Bose-Hubbard Hamiltonian"),
    (("commutation", "relation"), "commutation relation"),
    (("finite-difference",), "finite-difference equation"),
    (("loschmidt", "rate"), "Loschmidt rate function"),
    (("jump operator",), "jump operator"),
    (("liouvillian",), "Liouvillian superoperator expression"),
    (("density matrix",), "density matrix expression"),
    (("differential equation",), "coupled differential equation"),
    (("hamiltonian",), "Hamiltonian expression"),
    (("lagrangian",), "Lagrangian expression"),
]

GENERIC_MEANINGS = {
    "mathematical relation",
    "Hamiltonian expression",
    "enumerated mathematical expression",
    "operator commutation relation",
}

def _sentence_distance_score(sentence_index):
    """Proximity weight: sentences closer to the equation score higher."""
    if sentence_index <= 1:
        return 1.2
    if sentence_index <= 3:
        return 0.8
    return 0.4


def _context(equation):
    section = equation.get("section", "")
    text_before = equation.get("text_before", "")
    text_after = equation.get("text_after", "")
    context = " ".join(p for p in (section, text_before, text_after) if p)
    context = re.sub(r"\s+", " ", context)
    if len(context) > 4000:
        # Definitions almost always follow the equation (text_after).
        # Keep the tail of text_before for backward-looking patterns, then
        # the full text_after up to the limit so definition paragraphs are preserved.
        head = re.sub(r"\s+", " ", f"{section} {text_before}".strip())[-700:]
        tail = re.sub(r"\s+", " ", text_after.strip())[:3200]
        context = re.sub(r"\s+", " ", f"{head} {tail}".strip())
    return context


def _sentences(text):
    return [sentence.strip() for sentence in re.split(r"(?<=[.;])\s+", text) if sentence.strip()]


def _normalize_symbol_name(symbol):
    cleaned = symbol.strip().strip("$,.;:()[]{}")
    greek_map = {
        "λ": "lambda",
        "ρ": "rho",
        "σ": "sigma",
        "θ": "theta",
        "β": "beta",
        "γ": "gamma",
        "ω": "omega",
        "ν": "nu",
    }
    return greek_map.get(cleaned, cleaned)


def _clean_phrase(text):
    phrase = re.sub(r"\s+", " ", text).strip(" :;,.")
    phrase = re.sub(r"\([^)]*\)", "", phrase).strip()
    if len(phrase.split()) > 10:
        phrase = " ".join(phrase.split()[:10])
    return phrase


def _clean_definition_phrase(text):
    phrase = _clean_phrase(text)
    phrase = re.sub(
        r"^(?:the|a|an)\s+",
        "",
        phrase,
        flags=re.IGNORECASE,
    )
    phrase = re.sub(
        r"\s+(?:where|with|and|while|which)\b.*$",
        "",
        phrase,
        flags=re.IGNORECASE,
    ).strip(" :;,.")
    if not phrase or len(phrase) < 2:
        return ""
    if re.search(r"^(?:is|are|denotes|represents)$", phrase, re.IGNORECASE):
        return ""
    return phrase


def _shorten(text, limit):
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _normalize_math_text(text):
    return (
        text.replace("ˆ", r"\hat")
        .replace("†", r"\dagger")
        .replace("−", "-")
        .replace("𝑡", "t")
    )


def _reverse_description(description):
    if description.startswith("shares") or description.startswith("same"):
        return description
    if description == "explicit equation reference in derivation context":
        return "referenced by related derivation context"
    return description
