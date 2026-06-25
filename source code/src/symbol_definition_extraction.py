r"""Phase 4 — Symbol Definition Extraction.

Fills the values of the symbols dictionary built by ``symbol_extraction.py``.
Works entirely from the text extracted alongside each equation (the ``context``
string built from ``text_before`` and ``text_after``).
**Never uses any language model or external API.**

Extraction pipeline (5 stages in order)
-----------------------------------------
Stage 1 — *where-clause parser*
Stage 2 — *definitor-verb sentence scan*
Stage 3 — *appositive noun-before scan*
Stage 4 — *structural equation analysis + context disambiguation*
Stage 5 — *honest empty* (remaining symbols keep an empty string)

Public API
----------
* ``extract_symbol_definitions(symbol_keys, equation_text, context)``
* ``extract_definitions_from_text(symbol_keys, context)``
* ``extract_nounphrase_appositives(symbol_keys, context)``

References
----------
* Stathopoulos & Stevenson (2018) — "Variable Typing."
* Pagael & Schubotz (2014) — "Mathematical Language Processing."
* SemEval-2020 Task 6 — definition extraction from scientific text.
"""

import re

from src.symbol_definition_text import (
    _AND_NEW_DEF_RE,
    _DEFVERB_RE,
    _clean_definition,
    _find_best_key,
    _find_matching_keys,
    _parse_where_clause,
    _scan_appositive,
    _scan_definitor_sentences,
    _scrub_ar5iv_noise,
)
from src.symbol_structural_analysis import (
    _bound_only_letters,
    _dirac_gamma_context,
    _hatted_pauli_letters,
    _index_letters,
    _lindblad_context,
    _lorentz_index_context,
    _operator_marked_symbols,
    _partial_derivative_vars,
    _probability_call_symbols,
)


# ── Public entry point ────────────────────────────────────────────────────────

def extract_symbol_definitions(symbol_keys, equation_text, context):
    """Fill symbol definitions using a 5-stage deterministic pipeline.

    Stages are applied in order; a key is removed from the remaining pool
    as soon as any stage assigns it a non-empty definition.

    Parameters
    ----------
    symbol_keys : list of str
        Normalised keys produced by ``extract_symbols``.
    equation_text : str
        Raw LaTeX of the numbered equation (used for structural stages).
    context : str
        Combined ``section + text_before + text_after``.

    Returns
    -------
    definitions : dict
        ``{key: definition_str}`` – empty string when no evidence found.
    audit : dict
        ``{key: short_audit_string}`` describing how each definition
        was extracted.
    """
    definitions = {k: "" for k in symbol_keys}
    audit = {}
    remaining = list(symbol_keys)

    def _apply(stage_results, stage_label):
        for key, (defn, snippet) in stage_results.items():
            if key in remaining:
                definitions[key] = defn
                full_snippet = re.sub(r"\s+", " ", str(snippet)).strip()
                short_defn = str(defn)[:40]
                audit[key] = f"{stage_label}: {full_snippet!r} -> {short_defn!r}"
                remaining.remove(key)

    bound_letters = _bound_only_letters(equation_text)

    def _ctx_keys():
        return [k for k in remaining if not (len(k) == 1 and k in bound_letters)]

    # Stage 1 – where clause
    _apply(_parse_where_clause(context, _ctx_keys()), "where-clause")

    _SHORT_CTX_CHARS = 2000
    short_context = context[-_SHORT_CTX_CHARS:] if len(context) > _SHORT_CTX_CHARS else context

    # Stage 2 – definitor verb scan
    if _ctx_keys():
        _apply(_scan_definitor_sentences(short_context, _ctx_keys()), "defn-verb")

    # Stage 3 – appositive noun-before
    if _ctx_keys():
        _apply(_scan_appositive(short_context, _ctx_keys()), "appositive")

    # Stage 4a – summation / integration / subscript indices
    index_letters = _index_letters(equation_text)
    for key in list(remaining):
        if key in index_letters:
            definitions[key] = "summation index"
            audit[key] = f"index: {key!r} used as summation/subscript index"
            remaining.remove(key)

    # Stage 4b – hatted Pauli operators
    pauli_letters = _hatted_pauli_letters(equation_text)
    for key in list(remaining):
        base = re.sub(r"[_^].*$", "", key)
        if base in pauli_letters:
            definitions[key] = f"Pauli {base} operator"
            audit[key] = f"hatted-pauli: {key!r} -> Pauli {base}"
            remaining.remove(key)

    # Stage 4c – Dirac gamma matrices
    dirac_letters = _dirac_gamma_context(equation_text)
    for key in list(remaining):
        base = re.sub(r"[_^].*$", "", key)
        if base in dirac_letters:
            definitions[key] = "Dirac gamma matrix"
            audit[key] = f"dirac-gamma: {key!r} -> Dirac gamma matrix"
            remaining.remove(key)

    # Stage 4d – Lorentz indices
    lorentz_letters = _lorentz_index_context(equation_text)
    for key in list(remaining):
        base = re.sub(r"[_^].*$", "", key)
        if base in lorentz_letters:
            definitions[key] = "Lorentz index"
            audit[key] = f"lorentz-index: {key!r} -> Lorentz index"
            remaining.remove(key)

    # Stage 4e – p / q in function-call syntax → probability distribution
    prob_syms = _probability_call_symbols(equation_text)
    for key in list(remaining):
        base = re.sub(r"[_^].*$", "", key)
        if base in prob_syms:
            definitions[key] = "probability distribution"
            audit[key] = f"prob-call: {key!r} used as p(...) function call"
            remaining.remove(key)

    # Stage 4f – a / b / c without operator markers are left undefined
    op_marked = _operator_marked_symbols(equation_text)
    _BOSONIC_DICT_KEYS = {"a", "b", "c", "f"}
    for key in list(remaining):
        base = re.sub(r"[_^].*$", "", key)
        if base in _BOSONIC_DICT_KEYS and base not in op_marked:
            remaining.remove(key)

    # Stage 4g – L is Lindblad only when structural/prose evidence confirms it
    for key in list(remaining):
        if re.sub(r"[_^].*$", "", key) == "L":
            if _lindblad_context(equation_text, context):
                definitions[key] = "Lindblad operator"
                audit[key] = f"lindblad-ctx: {key!r} confirmed by equation/prose"
            remaining.remove(key)

    # Stage 4h – partial derivative denominator variables
    deriv_vars = _partial_derivative_vars(equation_text)
    for key in list(remaining):
        if key in deriv_vars:
            definitions[key] = "differential variable"
            audit[key] = f"partial-deriv: {key!r} used as differentiation variable"
            remaining.remove(key)

    # Stage 5 – honest empty; remaining keys keep definitions[key] == ""

    return definitions, audit


def extract_definitions_from_text(symbol_keys, context):
    """Run ONLY the document-text definition stages over *context*.

    Applies Stages 1–3 (prose-reading stages) with no structural inference.
    Used by the paper-wide recovery pass.

    Returns ``(definitions, evidence)`` where ``definitions`` maps key →
    definition string and ``evidence`` maps key → source-sentence snippet.
    """
    if not symbol_keys or not context:
        return {}, {}
    found = {}
    evidence = {}

    def _absorb(stage_results):
        for key, (defn, snip) in stage_results.items():
            if defn and key not in found:
                found[key] = defn
                evidence[key] = snip

    _absorb(_parse_where_clause(context, list(symbol_keys)))
    pending = [k for k in symbol_keys if k not in found]
    if pending:
        _absorb(_scan_definitor_sentences(context, pending))
    pending = [k for k in symbol_keys if k not in found]
    if pending:
        _absorb(_scan_appositive(context, pending))
    return found, evidence


# ── Stage 6: noun-phrase appositive via spaCy ─────────────────────────────────

_GENERIC_HEADS = frozenset({
    "system", "case", "form", "following", "result", "results", "expression",
    "equation", "way", "term", "terms", "fact", "order", "set", "number",
    "value", "values", "sense", "example", "section", "figure", "table",
    "paper", "problem", "method", "part", "parts", "context", "limit", "basis",
    "presence", "absence", "addition", "choice", "notation", "quantity",
    "quantities", "thing", "things", "object", "objects", "kind", "type",
    "above", "below", "left", "right", "rest", "remainder", "latter", "former",
})

_GREEK_GLYPHS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "rho": "ρ", "sigma": "σ",
    "tau": "τ", "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Omega": "Ω", "Gamma": "Γ", "Delta": "Δ", "Phi": "Φ", "Psi": "Ψ",
    "Sigma": "Σ", "Lambda": "Λ", "Theta": "Θ", "Pi": "Π",
}


def _clean_appositive_np(chunk_text):
    """Strip a leading determiner and validate a noun-phrase candidate."""
    np = re.sub(r"^\s*(?:the|a|an|this|these|those|that)\s+", "", chunk_text.strip(), flags=re.IGNORECASE)
    if re.match(r"^(?:following|above|below|aforementioned|same|latter|former|"
                r"preceding|previous|next|other|whole|entire|overall)\b", np, re.IGNORECASE):
        return ""
    if len(np.split()) == 1 and re.search(r"(?:ed|ing)$", np, re.IGNORECASE):
        return ""
    if re.search(r"\bi\.e\b|\be\.g\b|:", np):
        return ""
    np = _clean_definition(np)
    if not np:
        return ""
    words = np.split()
    if len(words) == 1 and words[-1].lower().rstrip("s") in {w.rstrip("s") for w in _GENERIC_HEADS}:
        return ""
    if words[-1].lower() in _GENERIC_HEADS:
        return ""
    return np


def extract_nounphrase_appositives(symbol_keys, context):
    """Stage 6 – anchored noun-phrase appositive extraction (spaCy).

    Finds definitions written as appositives adjacent to the symbol:
        "the coupling constant g"  /  "g, the coupling constant"

    Uses spaCy noun-phrase chunking for accurate phrase boundaries.
    The phrase must be anchored by a determiner and sit immediately
    adjacent (≤4 chars) to the symbol mention.
    """
    if not symbol_keys or not context:
        return {}
    from src.meaning_extraction import _spacy  # reuse the shared loader
    nlp = _spacy()
    if nlp is None or "parser" not in nlp.pipe_names:
        return {}

    text = _scrub_ar5iv_noise(context)[:30000]
    try:
        doc = nlp(text)
    except Exception:
        return {}
    chunks = [
        (c.start_char, c.end_char, c.text)
        for c in doc.noun_chunks
        if c.root.pos_ in ("NOUN", "PROPN") and c.root.tag_ not in ("VBN", "VBG")
    ]
    if not chunks:
        return {}

    results = {}
    evidence = {}
    for key in symbol_keys:
        base = re.split(r"[_^]", key)[0]
        tokens = {base}
        if base in _GREEK_GLYPHS:
            tokens.add(_GREEK_GLYPHS[base])
        best = None
        best_ev = ""
        for tok in tokens:
            for m in re.finditer(r"(?<![A-Za-z])" + re.escape(tok) + r"(?![A-Za-z])", text):
                s, e = m.start(), m.end()
                for cs, ce, ctext in chunks:
                    if 0 <= s - ce <= 4 and re.match(r"\s*(?:the|a|an)\b", ctext, re.IGNORECASE):
                        cand = _clean_appositive_np(ctext)
                    elif 0 <= cs - e <= 4 and re.match(r"\s*(?:the|a|an)\b", ctext, re.IGNORECASE):
                        cand = _clean_appositive_np(ctext)
                    else:
                        continue
                    if cand:
                        best = cand
                        lo, hi = min(s, cs), max(e, ce)
                        left = max(
                            (text.rfind(p, 0, lo) for p in (". ", "? ", "! ", "\n")),
                            default=-1,
                        )
                        left = left + 2 if left >= 0 else max(0, lo - 200)
                        rights = [r for r in (text.find(p, hi) for p in (". ", "? ", "! ", "\n")) if r >= 0]
                        right = min(rights) if rights else min(len(text), hi + 200)
                        best_ev = re.sub(r"\s+", " ", text[left:right]).strip()
                        break
                if best:
                    break
            if best:
                break
        if best:
            results[key] = best
            evidence[key] = best_ev or best
    return results, evidence


__all__ = ["extract_symbol_definitions", "extract_definitions_from_text",
           "extract_nounphrase_appositives"]
