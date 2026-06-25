r"""
Phase 3 — Symbol Identifier Extraction
========================================

Extracts symbol *tokens* from a LaTeX equation string.  Does **not** assign
meanings or definitions — that is done in Phase 4
(``symbol_definition_extraction.py``).

The output keys preserve the LaTeX-ish form found in the equation text, e.g.:
``R_{Mem4}``, ``v_2``, ``\hat{S}^{z}_{j}``, ``\rho``, ``Xi``.

Approach
--------
1. A recursive LaTeX atom parser (``_latex_atoms``) tokenises the equation
   into individual identifiers, Greek-letter commands, and hatted/subscripted
   symbols.
2. Heuristic filters (``_atom_allowed``) reject operators, decorators,
   formatting commands, and artifact tokens produced by PDF text extraction.
3. Index letters (summation variables, integration differentials) are
   identified by ``_index_letters`` in ``symbol_definition_extraction.py``
   and excluded from the symbol dictionary.

Public API
----------
* ``extract_symbols(equation_text, context, identifier_candidates)``
  Returns ``(symbols_dict, audit_dict)``.
"""

import re

from src.enrichment_common import GREEK_COMMANDS
from src.enrichment_common import _normalize_math_text


FORMAT_COMMANDS = {
    "hat",
    "widehat",
    "bar",
    "overline",
    "tilde",
    "vec",
    "mathbf",
    "boldsymbol",
    "mathcal",
    "mathbb",
    "mathrm",
    "rm",
}

SYMBOL_COMMANDS = set(GREEK_COMMANDS) | {"hbar"}

NOISE_COMMANDS = {
    "left",
    "right",
    "frac",
    "dfrac",
    "tfrac",
    "sqrt",
    "sum",
    "prod",
    "int",
    "lim",
    "begin",
    "end",
    "text",
    "hbox",
    "mbox",
    "operatorname",
    "quad",
    "qquad",
    "cdot",
    "times",
    "partial",
    "nabla",
    "langle",
    "rangle",
    "leftangle",
    "rightangle",
}

WORD_OPERATORS = {
    "sin",
    "cos",
    "tan",
    "exp",
    "log",
    "ln",
    "min",
    "max",
    "tr",
    "trace",
    "det",
    "rank",
    "diag",
    "argmax",
    "argmin",
}

TEXT_LABELS = {
    "in",
    "out",
    "sys",
    "th",
    "eff",
    "opt",
    "max",
    "min",
    "phys",
    "vac",
    "diel",
    "res",
    "fwhm",
    "energy",
    "loss",
}

# Capitalised English prose words that leak out of garbled aligned/PDF blocks
# and get mis-tokenised as symbols (e.g. "Schrödinger Term" → "Term").  These
# are sentence connectives / common nouns, never physics identifiers.  Matched
# case-insensitively on the atom base; kept to ≥3-letter words so 2-letter
# physics tokens (Tr, Re, Im, Id) are never affected.
PROSE_WORDS = {
    "here", "there", "finally", "where", "then", "thus", "also", "note",
    "since", "this", "that", "these", "those", "and", "but", "for", "with",
    "from", "into", "onto", "the", "new", "term", "let", "our", "its", "their",
    "such", "when", "while", "moreover", "hence", "therefore", "thereby",
    "given", "using", "consider", "define", "defined", "denotes", "where",
    "respectively", "namely", "above", "below", "each", "any", "all", "both",
}

LATEX_ENVIRONMENTS = {
    "align",
    "aligned",
    "array",
    "bmatrix",
    "cases",
    "equation",
    "gather",
    "matrix",
    "multline",
    "pmatrix",
    "split",
    "vmatrix",
}

UNICODE_TO_LATEX = {
    "ψ": r"\psi",
    "φ": r"\phi",
    "ϕ": r"\phi",
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "δ": r"\delta",
    "ε": r"\epsilon",
    "λ": r"\lambda",
    "μ": r"\mu",
    "ν": r"\nu",
    "ω": r"\omega",
    "ρ": r"\rho",
    "σ": r"\sigma",
    "θ": r"\theta",
    "π": r"\pi",
    "Ω": r"\Omega",
    "ℏ": r"\hbar",
    "ħ": r"\hbar",   # U+0127 small letter h with stroke (PDF rendering of ℏ)
}

SIMPLE_SUFFIX_COMMANDS = SYMBOL_COMMANDS | FORMAT_COMMANDS | {
    "dagger",
    "pm",
    "mp",
    "prime",
    "star",
    "ast",
}


def extract_symbols(equation_text, context="", identifier_candidates=None):
    """Extract symbol identifiers and fill their definitions from surrounding text.

    Parameters
    ----------
    equation_text : str
        Raw LaTeX of the equation.
    context : str
        Combined section + text_before + text_after from the HTML/PDF adapter.
    identifier_candidates : list or None
        MathML <mi> candidates from the HTML adapter (may be None).

    Returns
    -------
    symbols : dict
        ``{normalized_key: definition_str}`` – definition is ``''`` when
        no evidence was found in the text.
    audit : str
        Short summary for the audit trail.
    """
    identifiers = extract_symbol_identifiers(identifier_candidates, equation_text)
    symbols = {}
    for symbol in identifiers[:40]:
        key = _normalize_symbol_key(symbol)
        if key and key not in symbols:
            symbols[key] = ""
    if not symbols:
        return {}, "Exact-token symbol extraction found no reliable identifiers"

    # Lazy import to avoid circular dependency (symbol_definition_extraction
    # imports _normalize_symbol_key / _symbols_from_equation from this module).
    from src.symbol_definition_extraction import extract_symbol_definitions

    definitions, def_audit = extract_symbol_definitions(
        list(symbols.keys()), equation_text, context
    )
    for key, defn in definitions.items():
        symbols[key] = defn

    # Remove single-letter atoms that have NO definition and only appear inside
    # subscript/superscript braces in the equation (they are component indices,
    # not independent variables). E.g. 'z' from \hat{\sigma}_{z,\alpha}.
    symbols = _drop_subscript_only_index_letters(symbols, equation_text)

    found = sum(1 for value in symbols.values() if value)
    return symbols, {
        "extract_symbols": f"stored {len(symbols)} symbol(s): {', '.join(symbols)}",
        "extract_symbol_definitions": _definition_audit(symbols, def_audit, found),
        # Raw per-symbol evidence (key → "stage: 'snippet' -> 'def'") kept for the
        # provenance audit-trail; only present for symbols resolved per-equation.
        "_symbol_evidence": {k: v for k, v in def_audit.items() if k in symbols},
    }


def _definition_audit(symbols, def_audit, found):
    """Return a short audit value for extract_symbol_definitions()."""
    resolved = [f"{key}={value}" for key, value in symbols.items() if value]
    unresolved = [key for key, value in symbols.items() if not value]
    parts = [f"{found}/{len(symbols)} definition(s) found"]
    if resolved:
        parts.append(_short_list(resolved, 4))
    if unresolved:
        parts.append("unresolved: " + _short_list(unresolved, 4))
    evidence = [
        f"{key}: {desc}"
        for key, desc in def_audit.items()
        if key in symbols and symbols.get(key)
    ]
    if evidence:
        parts.append("evidence: " + _short_list(evidence, 2))
    return "; ".join(parts)


def _short_list(items, limit):
    """Format a bounded list for one-line audit output."""
    shown = [str(item) for item in items[:limit]]
    if len(items) > limit:
        shown.append(f"+{len(items) - limit} more")
    return ", ".join(shown)


def extract_symbol_identifiers(identifier_candidates, equation_text):
    """Extract unique identifier tokens without assigning meanings."""
    identifiers = []
    for symbol in _symbols_from_equation(equation_text):
        # Split known composite tokens that PDF/OCR renders as a single token.
        # "ih" = i·ℏ (iħ∂/∂t form) → split into imaginary unit + hbar.
        if symbol in ("ih", "ihbar"):
            _append(identifiers, "i")
            _append(identifiers, "hbar")
            continue
        _append(identifiers, symbol)

    # HTML MathML <mi> candidates are used only if the same token appears in the
    # selected equation text. The equation text remains the canonical source so
    # we do not replace ``\rho`` with ``rho`` or split multi-letter identifiers.
    for candidate in identifier_candidates or []:
        candidate = _clean_candidate(candidate)
        if candidate and _candidate_present(candidate, equation_text):
            _append(identifiers, candidate)
    return identifiers


def _symbols_from_equation(equation_text):
    """Extract exact LaTeX-ish identifier atoms from an equation string."""
    text = _prepare_equation_text(equation_text)
    atoms = _latex_atoms(text)
    compacted = []
    for atom in atoms:
        _append(compacted, atom)
    return compacted


def _prepare_equation_text(equation_text):
    text = _normalize_math_text(equation_text or "")
    text = _fix_missing_subscript_underscore(text)
    text = _fix_bare_hat_superscript(text)
    text = _strip_raw_prose_head(text)
    text = _strip_raw_prose_tail(text)
    text = _strip_aligned_prose_tail(text)
    text = _remove_text_blocks(text)
    text = _remove_latex_environments(text)
    text = _strip_aligned_prose_head(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _latex_atoms(text):
    atoms = []
    index = 0
    while index < len(text):
        char = text[index]

        if char == "\\":
            command_match = re.match(r"\\([A-Za-z]+)", text[index:])
            if not command_match:
                index += 1
                continue
            command = command_match.group(1)
            if command in FORMAT_COMMANDS:
                parsed = _consume_wrapped_command_atom(text, index)
                if parsed:
                    atom, index = parsed
                    if _atom_allowed(atom):
                        atoms.append(atom)
                    continue
            if command in SYMBOL_COMMANDS:
                end = index + len(command) + 1
                end = _consume_simple_suffixes(text, end)
                atom = text[index:end]
                if _atom_allowed(atom):
                    atoms.append(atom)
                index = end
                continue
            index += len(command) + 1
            continue

        if char in UNICODE_TO_LATEX:
            atom = UNICODE_TO_LATEX[char]
            end = _consume_simple_suffixes(text, index + 1)
            if end > index + 1:
                atom = atom + text[index + 1:end]
            if _atom_allowed(atom):
                atoms.append(atom)
            index = end
            continue

        if char.isalpha():
            match = re.match(r"[A-Za-z][A-Za-z0-9]*", text[index:])
            if not match:
                index += 1
                continue
            token = match.group(0)
            end = index + len(token)
            end = _consume_simple_suffixes(text, end)
            atom = text[index:end]
            if _atom_allowed(atom):
                atoms.append(atom)
            index = end
            continue

        index += 1
    return atoms


def _consume_wrapped_command_atom(text, start):
    command_match = re.match(r"\\([A-Za-z]+)\s*", text[start:])
    if not command_match:
        return None
    command_end = start + command_match.end()
    if command_end >= len(text) or text[command_end] != "{":
        return None
    brace_end = _balanced_brace_end(text, command_end)
    if brace_end is None:
        return None
    end = _consume_simple_suffixes(text, brace_end + 1)
    return text[start:end], end


def _consume_simple_suffixes(text, index):
    while index < len(text) and text[index] in "_^":
        suffix_end = _consume_one_simple_suffix(text, index)
        if suffix_end == index:
            break
        index = suffix_end
    return index


def _consume_one_simple_suffix(text, index):
    if index >= len(text) or text[index] not in "_^":
        return index
    cursor = index + 1
    if cursor >= len(text):
        return index

    if text[cursor] == "{":
        brace_end = _balanced_brace_end(text, cursor)
        if brace_end is None:
            return index
        content = text[cursor + 1:brace_end]
        return brace_end + 1 if _simple_suffix_content(content) else index

    if text[cursor] == "\\":
        command_match = re.match(r"\\([A-Za-z]+)", text[cursor:])
        if not command_match:
            return index
        command = command_match.group(1)
        if command in SIMPLE_SUFFIX_COMMANDS:
            return cursor + len(command) + 1
        return index

    match = re.match(r"[A-Za-z0-9]+", text[cursor:])
    if not match:
        return index
    content = match.group(0)
    return cursor + len(content) if _simple_suffix_content(content) else index


def _simple_suffix_content(content):
    cleaned = content.strip()
    if not cleaned:
        return False
    if cleaned in {"+", "-", r"\pm", r"\mp"}:
        return True
    if any(token in cleaned for token in ("=", "/", "|", ",", ";", ":", r"\sim", r"\to")):
        return False
    normalized = re.sub(r"\\(?:%s)\s*\{([^{}]*)\}" % "|".join(sorted(FORMAT_COMMANDS)), r"\1", cleaned)
    normalized = re.sub(r"\\[A-Za-z]+", "X", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    return bool(re.fullmatch(r"[A-Za-z0-9*+\-]+", normalized))


def _balanced_brace_end(text, open_index):
    depth = 0
    for index in range(open_index, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _atom_allowed(atom):
    atom = atom.strip()
    if not atom:
        return False

    base = _atom_base(atom)
    base_lower = base.lower()
    if base_lower in LATEX_ENVIRONMENTS:
        return False
    if base_lower in NOISE_COMMANDS or base_lower in WORD_OPERATORS or base_lower in TEXT_LABELS:
        return False
    # Reject capitalised/lowercase English prose words (≥3 letters) that leak
    # from garbled text — never physics identifiers.
    if len(base) >= 3 and base.isalpha() and base_lower in PROSE_WORDS:
        return False
    # Reject numeric-base atoms ("0", "1", "\hat{1}", "\mathbf{0}") — a pure
    # integer is a constant, not a symbol identifier (e.g. the "1" from the
    # identity \hat{1} or the "0" from the zero ket |0>).
    if base.isdigit():
        return False
    if base_lower.startswith("\\") and base_lower[1:] in NOISE_COMMANDS:
        return False
    if re.fullmatch(r"d(?:\^\{?\d+\}?|[txydz])?", atom):
        return False
    # Reject bare multi-letter all-lowercase alphabetic tokens that are not
    # Greek-letter names, known math abbreviations, or LaTeX commands.
    # Single letters (len=1) are always valid; LaTeX commands start with '\'.
    # Multi-letter lowercase tokens like "for", "anti", "particles", "their"
    # are prose leakage from PDF piecewise condition labels, not identifiers.
    if (len(base) >= 2
            and base == base_lower          # all lowercase
            and base.isalpha()              # no digits, no backslash
            and base not in SYMBOL_COMMANDS # Greek names: alpha, gamma, rho…
            and base not in WORD_OPERATORS
            and base not in TEXT_LABELS
            and base not in LATEX_ENVIRONMENTS):
        return False
    # Reject CamelCase+digit tokens (≥4 chars) that are subscript-text
    # concatenation artifacts from garbled HTML/PDF rendering.
    # Legitimate subscripted symbols are written as M_{2}, not "Mem2".
    # Pattern: uppercase letter + 2+ lowercase letters + 1+ digits (e.g. Mem2).
    if re.match(r"^[A-Z][a-z]{2,}\d+$", base):
        return False
    return True


def _atom_base(atom):
    command_match = re.match(r"\\([A-Za-z]+)", atom)
    if command_match:
        command = command_match.group(1)
        if command in FORMAT_COMMANDS:
            inner = _first_brace_content(atom)
            return _atom_base(inner or atom)
        return "\\" + command
    plain_match = re.match(r"[A-Za-z][A-Za-z0-9]*", atom)
    return plain_match.group(0) if plain_match else atom


def _first_brace_content(text):
    start = text.find("{")
    if start < 0:
        return ""
    end = _balanced_brace_end(text, start)
    if end is None:
        return ""
    return text[start + 1:end]


def _remove_text_blocks(text):
    text = re.sub(r"\\(?:text|hbox|mbox|operatorname)\s*\{[^{}]*\}", " ", text)
    return text


def _remove_latex_environments(text):
    text = re.sub(r"\\(?:begin|end)\{[^{}]*\}", " ", text)
    for environment in LATEX_ENVIRONMENTS:
        text = re.sub(rf"\b{re.escape(environment)}\b", " ", text)
    return text


def _strip_raw_prose_tail(text):
    """Drop explanatory prose that OCR may append to a formula string."""
    patterns = [
        r"\s*,?\s+where\b",
        r"\s*\.\s+(?:Since|Consequently|Therefore|Thus|Hence|This|The)\b",
        r"\s+(?:represents|characterized|assumed|expressed|indicated)\b",
    ]
    cut = len(text or "")
    for pattern in patterns:
        match = re.search(pattern, text or "", flags=re.IGNORECASE)
        if match:
            cut = min(cut, match.start())
    return (text or "")[:cut]


_OCR_MISSING_SUBSCRIPT_RE = re.compile(
    r"(\\(?:hat|widehat|tilde|bar|vec|overline|mathbb|mathcal)\{[^{}]+\})"
    r"([A-Za-z0-9]{1,3})"
    r"(?=[^A-Za-z0-9]|$)"
)


def _fix_missing_subscript_underscore(text):
    """Repair PDF OCR dropout of '_': \\hat{H}0 → \\hat{H}_{0}.

    Detects decorated commands (\\hat, \\tilde, \\mathbb, etc.) whose closing
    brace is immediately followed by 1-3 alphanumeric chars without a '_' or '^'
    separator — a structural pattern of PDF OCR subscript dropout.
    """
    return _OCR_MISSING_SUBSCRIPT_RE.sub(r"\1_{\2}", text)


def _strip_raw_prose_head(text):
    """Strip leading prose before the first LaTeX command.

    When PDF extraction prepends figure captions or surrounding sentences to the
    equation text, the result starts with plain English words before any '\\cmd'.
    With 5+ prose words, the entire prefix is stripped (high confidence it is a
    caption). With 3-4 prose words, strip to the last sentence boundary ('. '
    ': ' or '; ') found in the prefix; if none, strip the whole prefix.
    Single-letter tokens are treated as math variables and excluded from the
    prose-word count so that equations like 'G(t) = \\lim ...' are not affected.
    """
    if not text:
        return text
    m = re.search(r"\\[A-Za-z]", text)
    if not m or m.start() == 0:
        return text
    prefix = text[: m.start()]
    # Only count 2+-letter words; single letters are math, not prose
    prose_words = re.findall(r"\b[A-Za-z]{2,}\b", prefix)
    if len(prose_words) < 3:
        return text
    if len(prose_words) >= 5:
        # Clearly a caption/sentence prefix — drop everything up to first LaTeX
        return text[m.start() :]
    # 3-4 prose words: try to honour the last sentence boundary
    last_boundary = None
    for bm in re.finditer(r"[.;:]\s+", prefix):
        last_boundary = bm.end()
    if last_boundary is not None:
        return text[last_boundary:]
    return text[m.start() :]


def _fix_bare_hat_superscript(text):
    r"""Convert \\hat without braces to ^ when it appears mid-expression.

    ``_normalize_math_text`` converts the Unicode modifier ˆ (U+02C6) to
    ``\\hat``.  When ``\\hat`` appears without braces between two tokens (e.g.
    ``H \\hat 0``), it is a PDF-OCR artefact for a superscript/subscript
    separator.  Convert it to ``^`` so ``_consume_simple_suffixes`` can attach
    it: ``H \\hat 0`` → ``H^0``.

    Only fires when ``\\hat`` is immediately preceded by an alphanumeric or
    ``}`` and followed (after optional spaces) by an alphanumeric character,
    which excludes operators like ``= \\hat S`` and properly-braced atoms like
    ``\\hat{H}``.
    """
    return re.sub(
        r"(?<=[A-Za-z0-9}])\s+\\hat\s+(?=[A-Za-z0-9])",
        "^",
        text,
    )


# Matches: sentence terminator + LaTeX aligned line-break (\\) + whitespace.
# Used to detect prose sentences appended after a math equation inside a
# \begin{aligned} environment.
_ALIGNED_PROSE_TAIL_RE = re.compile(r"[.\]]\s*(?:\\\\)+\s+")


def _strip_aligned_prose_tail(text):
    r"""Strip prose sentences that follow a ``\\`` line-break inside aligned envs.

    Pattern: ``] . \\ A key ingredient …`` — the equation ends, then a LaTeX
    line-break (``\\``) introduces a prose annotation.  Detects this by
    requiring 4+ plain alphabetic words between the line-break and the next
    LaTeX command (or end of string).
    """
    for m in _ALIGNED_PROSE_TAIL_RE.finditer(text):
        rest = text[m.end() :]
        first_latex = re.search(r"\\[A-Za-z]", rest)
        prose_region = rest[: first_latex.start()] if first_latex else rest
        prose_words = re.findall(r"\b[A-Za-z]{2,}\b", prose_region)
        if len(prose_words) >= 4:
            return text[: m.start()]
    return text


def _strip_aligned_prose_head(text):
    r"""Strip prose head revealed after ``\\begin{aligned}`` markers are removed.

    Uses a *stricter* prose-word criterion than :func:`_strip_raw_prose_head`:
    only counts 3+-letter ALL-LOWERCASE words (``are``, ``defined``, ``the``).
    Technical abbreviations like ``Mem2``, ``dRMem`` contain uppercase letters
    or digits and are not counted, so real math content is never stripped.
    A sentence-initial capital word (e.g. ``They``) is also counted when it
    is followed by lowercase words.
    """
    if not text:
        return text
    m = re.search(r"\\[A-Za-z]", text)
    if not m or m.start() == 0:
        return text
    prefix = text[: m.start()]
    # 3+-char all-lowercase words = prose; mixed-case / short = math
    prose_words = re.findall(r"\b[a-z]{3,}\b", prefix)
    # A sentence-initial Capital+lower word (e.g. "They") also counts
    if re.match(r"\s*[A-Z][a-z]{2,}", prefix):
        prose_words = prose_words + ["<sentence-start>"]
    if len(prose_words) < 4:
        return text
    if len(prose_words) >= 5:
        return text[m.start() :]
    last_boundary = None
    for bm in re.finditer(r"[.;:]\s+", prefix):
        last_boundary = bm.end()
    if last_boundary is not None:
        return text[last_boundary:]
    return text[m.start() :]


def _clean_candidate(candidate):
    candidate = str(candidate or "").strip()
    if not candidate:
        return ""
    if len(candidate) == 1 and candidate in UNICODE_TO_LATEX:
        return UNICODE_TO_LATEX[candidate]
    if candidate in SYMBOL_COMMANDS:
        return "\\" + candidate
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", candidate):
        return candidate
    return ""


def _candidate_present(candidate, equation_text):
    return candidate in _symbols_from_equation(equation_text)


def _append(items, atom):
    atom = re.sub(r"\s+", " ", str(atom or "")).strip()
    if atom and atom not in items:
        items.append(atom)


# Decorator commands stripped from a key while keeping their braced argument.
_KEY_DECORATORS = FORMAT_COMMANDS | {
    "mathbb", "mathscr", "mathfrak", "mathsf", "mathit", "operatorname", "text",
}
_KEY_DECO_RE = "|".join(sorted(_KEY_DECORATORS, key=len, reverse=True))


def _normalize_symbol_key(atom):
    r"""Normalize a LaTeX-ish symbol atom into a clean, readable JSON key.

    Removes formatting decorators (``\hat``, ``\bar``, ``\mathbb``, ...) keeping
    their argument, drops the backslash from remaining commands (``\rho`` ->
    ``rho``, ``\dagger`` -> ``dagger``), and removes braces while preserving the
    sub/superscript structure so a compound symbol stays whole. Examples::

        \hat{S}^{z}            -> S^z
        \hat{S}^{\pm}          -> S^pm        (never split into S and pm)
        \hat{A}_{p}^{\dagger}  -> A_p^dagger
        R_{3}                  -> R_3
        \rho                   -> rho
    """
    key = re.sub(r"\s+", "", str(atom or ""))
    if not key:
        return ""
    previous = None
    while previous != key:  # peel nested/repeated decorators
        previous = key
        key = re.sub(r"\\(?:%s)\s*\{([^{}]*)\}" % _KEY_DECO_RE, r"\1", key)
        key = re.sub(r"\\(?:%s)(?![A-Za-z])" % _KEY_DECO_RE, "", key)
    key = re.sub(r"\\([A-Za-z]+)", r"\1", key)  # \rho -> rho, \dagger -> dagger
    key = key.replace("{", "").replace("}", "")
    return re.sub(r"\s+", "", key)


def _drop_subscript_only_index_letters(symbols, equation_text):
    """Remove single-letter atoms with no definition that never appear standalone.

    When a single letter (e.g. 'z' from \\hat{\\sigma}_{z,\\alpha}) has no
    definition and only occurs inside subscript/superscript braces in the equation
    body, it is a component index — not an independent physics variable.  Keeping
    it clutters the symbol list with empty entries and can cause false 'time'
    assignments when the letter is 't' appearing only as a subscript label.

    Letters that DO appear standalone (like 't' in 'i\\delta t') or that already
    have a definition (like 'alpha' from the domain dictionary) are always kept.
    """
    if not equation_text or not symbols:
        return symbols
    result = {}
    for key, defn in symbols.items():
        # Only apply to single-letter backslash-free keys with NO definition
        if defn or len(key) != 1 or not key.isalpha():
            result[key] = defn
            continue
        if _appears_standalone_in_equation(key, equation_text):
            result[key] = defn
        # else: subscript-only index letter → drop silently
    return result


def _appears_standalone_in_equation(letter, equation_text):
    """Return True if 'letter' appears outside any subscript/superscript braces."""
    text = equation_text
    # Strip content inside _{...} and ^{...}
    text = re.sub(r"[_^]\{[^{}]*\}", " ", text)
    # Strip bare single-char subscripts/superscripts like _x or ^n
    text = re.sub(r"[_^][A-Za-z0-9]", " ", text)
    # Strip LaTeX commands so letters inside \alpha don't count as standalone 'a'
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    # The letter must appear as a non-letter-bounded token
    return bool(re.search(rf"(?<![A-Za-z]){re.escape(letter)}(?![A-Za-z])", text))


__all__ = ["extract_symbols", "extract_symbol_identifiers", "_symbols_from_equation"]
