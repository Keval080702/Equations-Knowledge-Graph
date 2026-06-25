"""Equation-structure analysis for symbol classification.

Detects single-letter symbols whose role is determined by the equation's
LaTeX structure rather than surrounding prose (summation indices, Pauli
operators, Dirac matrices, Lorentz indices, etc.).

Public API
----------
All functions accept the raw LaTeX equation string and return a set of
symbol keys or a boolean.

* ``_index_letters(equation_text)``
* ``_partial_derivative_vars(equation_text)``
* ``_bound_only_letters(equation_text)``
* ``_probability_call_symbols(equation_text)``
* ``_operator_marked_symbols(equation_text)``
* ``_lindblad_context(equation_text, context_text)``
* ``_hatted_pauli_letters(equation_text)``
* ``_dirac_gamma_context(equation_text)``
* ``_lorentz_index_context(equation_text)``
"""

import re


def _index_letters(equation_text):
    """Return single letters used as summation/product/integration indices."""
    text = str(equation_text or "")
    letters = set()

    _SUM_PROD = r"\\(?:sum|prod)(?:\\(?:limits|nolimits))?\s*"
    for match in re.finditer(_SUM_PROD + r"_\{?\s*([A-Za-z])\s*=", text):
        letters.add(match.group(1))
    for match in re.finditer(_SUM_PROD + r"_\{?\s*([A-Za-z])\s*[},]", text):
        letters.add(match.group(1))
    for match in re.finditer(_SUM_PROD + r"_\{([^}]{1,30})\}", text):
        for letter in re.findall(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])", match.group(1)):
            letters.add(letter)
    for match in re.finditer(_SUM_PROD + r"_[^\\]*?\^\{?\s*([A-Za-z])\s*\}?", text):
        letters.add(match.group(1))
    for match in re.finditer(r"(?<![A-Za-z])d\s*(?:\\[A-Za-z]+)?_\{?\s*([A-Za-z])\s*\}?", text):
        letters.add(match.group(1))

    _COMPLEX_UNIT_RE = re.compile(
        r"(?<![A-Za-z_^{])i\\hbar"
        r"|e\s*\^\s*\{\s*i"
        r"|e\s*\^\s*i(?![A-Za-z])"
        r"|\\(?:mathrm|text)\{\s*i\s*\}"
        r"|\\imath\b"
    )
    _complex_present = bool(_COMPLEX_UNIT_RE.search(text))
    for letter in ("i", "j", "k", "l", "m"):
        if letter in letters:
            continue
        in_subscript = bool(
            re.search(rf"_\{{\s*{letter}(?![A-Za-z0-9])", text) or
            re.search(rf"_\s*{letter}(?![A-Za-z0-9])", text) or
            re.search(rf"\^\{{\s*\(\s*{letter}\s*\)\s*\}}", text)
        )
        standalone = bool(re.search(
            rf"(?<![A-Za-z0-9_{{^\\]){letter}(?![A-Za-z0-9_{{\\])", text
        ))
        complex_ctx = _complex_present and letter == "i"
        if in_subscript and not standalone and not complex_ctx:
            letters.add(letter)
    return letters


def _partial_derivative_vars(equation_text):
    """Return single letters used exclusively as partial differentiation variables."""
    text = str(equation_text or "")
    candidates = set()
    for m in re.finditer(r"\\partial\s*\{?\s*([A-Za-z])\s*\}?(?![A-Za-z_^{])", text):
        candidates.add(m.group(1))
    result = set()
    for letter in candidates:
        scrubbed = re.sub(
            r"\\partial\s*\{?\s*" + re.escape(letter) + r"\s*\}?(?![A-Za-z_^{])",
            "", text,
        )
        standalone = bool(
            re.search(rf"(?<![A-Za-z0-9_{{^\\]){re.escape(letter)}(?![A-Za-z0-9_{{\\])", scrubbed)
        )
        if not standalone:
            result.add(letter)
    return result


def _bound_only_letters(equation_text):
    """Return single letters that appear ONLY in bound positions in the equation."""
    text = str(equation_text or "")
    present = set(re.findall(r"(?<![A-Za-z\\])([A-Za-z])(?![A-Za-z])", text))
    if not present:
        return set()
    free = text
    free = re.sub(r"\\partial\s*\{?\s*[A-Za-z]\s*\}?(?![A-Za-z_^{])", " ", free)
    free = re.sub(r"[_^]\{[^{}]*\}", " ", free)
    free = re.sub(r"[_^]\\?[A-Za-z0-9]+", " ", free)
    free = re.sub(r"\bd\s*([A-Za-z])\b", " ", free)
    free = re.sub(r"\\(?:sum|prod|int)(?:\\limits)?", " ", free)
    free = re.sub(r"([A-Za-z])\s*\(\s*[A-Za-z](?:\s*[-+]\s*\d+)?\s*\)", r"\1 ", free)
    free_letters = set(re.findall(r"(?<![A-Za-z\\])([A-Za-z])(?![A-Za-z])", free))
    return {c for c in present if c not in free_letters}


def _probability_call_symbols(equation_text):
    """Return symbol keys that appear as probability/density function calls."""
    text = str(equation_text or "")
    letters = set()
    for letter in ("p", "q"):
        if re.search(
            rf"(?<![A-Za-z])({re.escape(letter)})\s*(?:_\{{[^}}]{{1,10}}\}})?\s*(?:\\left)?\(",
            text,
        ):
            letters.add(letter)
    return letters


def _operator_marked_symbols(equation_text):
    """Return lowercase letters that carry explicit quantum-operator markers (hat/dagger)."""
    text = str(equation_text or "")
    letters = set()
    for letter in ("a", "b", "c", "f"):
        has_hat = bool(re.search(rf"\\hat\{{\s*{letter}\s*\}}", text))
        has_dagger = bool(re.search(
            rf"(?<![A-Za-z]){letter}[_^]?" + r"\s*\{?\\?(?:dagger|†)\}?", text
        ))
        if has_hat or has_dagger:
            letters.add(letter)
    return letters


def _lindblad_context(equation_text, context_text):
    """Return True when ``L`` is a Lindblad (jump) operator based on equation structure or prose."""
    eq = str(equation_text or "")
    ctx = str(context_text or "").lower()
    if re.search(r"L\s*(?:\\rho|ρ|\\rho_[A-Za-z0-9])\s*L\s*(?:\\dagger|†)", eq):
        return True
    if re.search(r"L_[a-z0-9]\s*(?:\\rho|ρ)\s*L_[a-z0-9]", eq):
        return True
    if re.search(r"\blindblad\b|\bmaster\s+equation\b|\bjump\s+operator\b", ctx):
        return True
    return False


def _hatted_pauli_letters(equation_text):
    """Return X/Y/Z letters that appear as hatted Pauli operators."""
    text = str(equation_text or "")
    letters = set()
    for letter in ("X", "Y", "Z"):
        if re.search(r"\\hat\{?\s*" + letter + r"\b", text):
            letters.add(letter)
    return letters


def _dirac_gamma_context(equation_text):
    """Return {"gamma"} when γ appears with a Lorentz index (Dirac matrix context)."""
    text = str(equation_text or "")
    if re.search(r"\\gamma\s*[\^_]\s*\{?\\(?:mu|nu)\}?", text):
        return {"gamma"}
    if re.search(r"\\gamma\s*\^\s*\{?\s*[0-3]\s*\}?", text):
        return {"gamma"}
    return set()


def _lorentz_index_context(equation_text):
    """Return symbol keys to label as Lorentz indices when μ/ν appear as contracted indices."""
    text = str(equation_text or "")
    letters = set()
    if re.search(r"[A-Za-z]\s*[\^_]\s*\{\\mu\\nu\}", text):
        letters.update({"mu", "nu"})
    if re.search(r"\\(?:partial|mathcal\{D\}|nabla)\s*_\s*\{?\\mu\}?", text):
        letters.add("mu")
    has_mu = bool(re.search(r"\\gamma\s*[\^_]\{?\\mu\}?", text))
    has_nu = bool(re.search(r"\\gamma\s*[\^_]\{?\\nu\}?", text))
    if has_mu:
        letters.add("mu")
    if has_nu:
        letters.add("nu")
    return letters
