"""
Phase 2 — PDF Math & LaTeX Normalization Utilities
===================================================

Pure text-transformation functions that convert Unicode math glyphs and
PDF-extracted notation into stable LaTeX-like strings.

These functions carry **no side effects** and make no network or filesystem
calls.  They are imported by the PDF extraction engines and by the public
PDF adapter.

Pipeline role
-------------
Called as the last step in every PDF extraction engine before an equation
string is stored.  The transformations are deliberately conservative: if
structure is missing from the PDF text we record low-confidence output
rather than inventing formula parts.

Public helpers
--------------
* ``_normalize_pdf_math``         — replace mathematical Unicode glyph variants
* ``_postprocess_pdf_equation``   — fix common flattened-PDF math patterns
* ``_pdf_text_to_latex_like``     — full single-equation conversion pipeline
* ``_strip_prose_prefix``         — drop lead-in prose from mixed text+math blocks
"""

import re


__all__ = [
    "_normalize_pdf_math",
    "_postprocess_pdf_equation",
    "_pdf_text_to_latex_like",
    "_pdf_text_to_latex_like_single",
    "_strip_prose_prefix",
    "_replace_unicode_math_symbols",
    "_latexify_known_operators",
    "_latexify_hats_and_daggers",
    "_latexify_derivatives",
    "_latexify_indices",
    "_latexify_simple_fractions",
    "_latexify_flattened_functions",
    "_latexify_brackets",
    "_strip_prose_suffix",
]


# ── Unicode glyph normalisation ───────────────────────────────────────────────

def _normalize_pdf_math(text):
    """Normalise common PDF-extracted math glyphs to stable plain-text chars.

    PDF renderers often embed mathematical italic and Greek characters as
    private-use Unicode codepoints (U+1D400–U+1D7FF).  This function maps
    the most common ones back to their ASCII/Latin equivalents so that
    downstream regex patterns remain simple and portable.
    """
    replacements = {
        "𝐴": "A", "𝐵": "B", "𝐺": "G", "𝐻": "H", "𝑡": "t", "𝑣": "v",
        "𝑑": "d", "𝐾": "K", "𝐶": "C", "𝑅": "R", "𝑀": "M", "𝑒": "e",
        "𝑓": "f", "𝑔": "g", "𝑖": "i", "𝑚": "m", "𝑟": "r", "𝑠": "s",
        "𝑐": "c", "𝑜": "o", "𝑢": "u", "𝑞": "q", "𝑎": "a", "𝑏": "b",
        "𝑝": "p", "𝑙": "l", "𝑛": "n", "𝑥": "x", "𝑘": "k", "𝑦": "y",
        "𝑧": "z", "𝑁": "N", "𝑉": "V", "𝑊": "W", "𝑋": "X", "𝑇": "T",
        "𝜌": "ρ", "𝜎": "σ", "𝜃": "θ", "𝛽": "β", "𝜔": "ω", "𝜆": "λ",
        "𝜇": "µ", "𝜈": "ν", "𝜏": "τ", "𝜓": "ψ", "ℎ": "h",
    }
    normalized = text
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return normalized


def _replace_unicode_math_symbols(text):
    """Replace Greek letters and operator Unicode symbols with LaTeX commands."""
    replacements = {
        "ρ": r"\rho",    "σ": r"\sigma",  "θ": r"\theta",  "β": r"\beta",
        "ω": r"\omega",  "λ": r"\lambda", "µ": r"\mu",     "μ": r"\mu",
        "τ": r"\tau",    "ψ": r"\psi",    "Ψ": r"\Psi",    "φ": r"\phi",
        "ν": r"\nu",     "π": r"\pi",     "γ": r"\gamma",  "∆": r"\Delta",
        "Δ": r"\Delta",  "∂": r"\partial","ℏ": r"\hbar",   "∑": r"\sum",
        "Σ": r"\sum",    "Π": r"\prod",   "⊗": r"\otimes", "∞": r"\infty",
        "±": r"\pm",     "−": "-",        "∗": r"^{*}",    "·": r"\cdot",
        "→": r"\to",     "←": r"\leftarrow", "†": r"^{\dagger}", "√": r"\sqrt",
        "⟨": r"\langle ","⟩": r" \rangle","⃗": r"\vec",    "̂": r"\hat",
        "˙": r"\dot",                                          # U+02D9 DOT ABOVE (time derivative)
        "\x02": r"\left[", "\x03": r"\right]",
        "\x14": r"\left[", "\x15": r"\right]",
    }
    converted = text
    for source, target in replacements.items():
        converted = converted.replace(source, target)
    return converted


# ── Operator / function latexification ───────────────────────────────────────

def _latexify_known_operators(text):
    """Convert bare operator tokens (X, Y, tr, Tr, exp, ln …) to LaTeX."""
    text = re.sub(r"(?<![A-Za-z\\])X(?![A-Za-z])", r"\\sum", text)
    text = re.sub(r"(?<![A-Za-z\\])Y(?![A-Za-z])", r"\\prod", text)
    text = re.sub(
        r"(?<![A-Za-z\\])X\s+([A-Za-z\\][A-Za-z0-9_{}\\^*]*)", r"\\sum_{\1}", text
    )
    text = re.sub(r"\blim_\{?([A-Za-z][^}\s]*)\}?", r"\\lim_{\1}", text)
    text = re.sub(r"\blim\s*([A-Za-z]\\to[^\s]+)", r"\\lim_{\1}", text)
    text = re.sub(r"\btr\s*", r"\\operatorname{tr}", text)
    text = re.sub(r"\bTr\s*", r"\\operatorname{Tr}", text)
    text = re.sub(r"\bexp\s*", r"\\exp", text)
    text = re.sub(r"\bln\b", r"\\ln", text)
    text = re.sub(r"(?<![A-Za-z])2ln2\b", r"2\\ln 2", text)
    text = re.sub(r"\bargmax\b", r"\\arg\\max", text)
    return text


def _latexify_hats_and_daggers(text):
    """Convert hat/dagger notation to LaTeX commands."""
    text = re.sub(r"ˆ\s*([A-Za-z])", r"\\hat{\1}", text)
    text = re.sub(r"\\hat\s*([A-Za-z])", r"\\hat{\1}", text)
    text = re.sub(
        r"\\hat\{([A-Za-z])\}\^\{\\dagger\}", r"\\hat{\1}^{\\dagger}", text
    )
    text = re.sub(r"\\hat\{([A-Za-z])\}\\pm\b", r"\\hat{\1}^{\\pm}", text)
    text = re.sub(r"\\hat\{([A-Za-z])\}([+\-])", r"\\hat{\1}^{\2}", text)
    text = re.sub(r"\\hat\{([A-Za-z])\}z\b", r"\\hat{\1}^{z}", text)
    text = re.sub(r"([A-Za-z])\^\{\\dagger\}", r"\1^{\\dagger}", text)
    # Fix overdot \dot without braces before \hat: \dot\hat{X} → \dot{\hat{X}}
    text = re.sub(r"\\dot\s*(\\hat\{[A-Za-z]\})", r"\\dot{\1}", text)
    # Fix PDF layout artifact where subscript follows dagger: \hat{X} ^{\dagger} Y → \hat{X}_Y^{\dagger}
    text = re.sub(
        r"(\\hat\{[A-Za-z]\})\s*\^\{\\dagger\}\s+([A-Z])(?=\s|\\|\{|$)",
        r"\1_{\2}^{\\dagger}",
        text,
    )
    return text


def _latexify_derivatives(text):
    """Convert common derivative patterns to \\frac{d...}{dt} form."""
    text = re.sub(r"\bd\\rho\s+dt\b", r"\\frac{d\\rho}{dt}", text)
    text = re.sub(r"\\partialt\b", r"\\partial t", text)
    text = re.sub(
        r"\\partial\|?\\?psi\s*\\rangle\s+\\partial\s*t",
        lambda _m: r"\frac{\partial|\psi\rangle}{\partial t}",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bd2([A-Za-z])_?(\d*)\(([^)]*)\)/dt2\b",
        lambda m: (
            rf"\frac{{d^2 {m.group(1)}{_subscript(m.group(2))}"
            rf"({m.group(3)})}}{{dt^2}}"
        ),
        text,
    )
    text = re.sub(
        r"\bd([A-Za-z][A-Za-z0-9_]*)\(([^)]*)\)/dt\b",
        lambda m: (
            rf"\frac{{d{_latexify_symbol_token(m.group(1))}({m.group(2)})}}{{dt}}"
        ),
        text,
    )
    text = re.sub(
        r"\\partial\|?\\?psi\\?\\rangle\s*/?\s*\\partial\s*t",
        lambda _m: r"\frac{\partial|\psi\rangle}{\partial t}",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _latexify_indices(text):
    """Rewrite common subscript and index concatenation patterns."""
    text = re.sub(
        r"\bRMem(\d*)", lambda m: rf"R_{{\mathrm{{Mem}}{m.group(1)}}}", text
    )
    text = re.sub(
        r"\bK(\d*)coupled\b",
        lambda m: rf"K_{{{m.group(1)}\mathrm{{coupled}}}}",
        text,
    )
    text = re.sub(
        r"\\hat\{A\}\^\{\\dagger\}\s*([pm])\b", r"\\hat{A}_{\1}^{\\dagger}", text
    )
    text = re.sub(r"\\hat\{A\}\s*([pm])\b", r"\\hat{A}_{\1}", text)
    text = re.sub(r"\bf([pm])\s*\(", r"f_\1(", text)
    text = re.sub(r"\^\{\\dagger\}f_([pm])", r"^{\\dagger} f_\1", text)
    text = re.sub(r"\\rho([A-Za-z]+)", r"\\rho_{\1}", text)
    text = re.sub(r"\\rho(\d+)", r"\\rho_{\1}", text)
    text = re.sub(r"\\sigma([A-Za-z]{1,3})", r"\\sigma_{\1}", text)
    text = re.sub(r"\\psi([A-Za-z]+)", r"\\psi_{\1}", text)
    text = re.sub(r"\\Psi(\d+)", r"\\Psi_{\1}", text)
    text = re.sub(r"([A-Za-z])([0-9]+):([A-Za-z0-9]+)", r"\1_{\2:\3}", text)
    text = re.sub(
        r"(\\(?:theta|lambda|rho|sigma|tau|nu|gamma|beta|omega|psi|Psi|Delta))"
        r"([A-Za-z0-9]+)",
        r"\1_{\2}",
        text,
    )
    text = re.sub(r"\bImax\b", r"I_{\\max}", text)
    text = re.sub(r"\bImin\b", r"I_{\\min}", text)
    text = re.sub(r"\bhoptimal\b", r"h_{\\mathrm{optimal}}", text)
    text = re.sub(r"(\\tau)_?cFWHM\b", r"\1_c^{\\mathrm{FWHM}}", text)
    text = re.sub(r"(\\tau)_\{cFWHM\}", r"\1_c^{\\mathrm{FWHM}}", text)
    text = re.sub(r"(\\lambda)FWHM\b", r"\1_{\\mathrm{FWHM}}", text)
    text = re.sub(r"(\\lambda)0\b", r"\1_0", text)
    text = re.sub(r"\b([A-Za-z])([0-9]+)\b", r"\1_{\2}", text)
    text = re.sub(r"\b([A-Za-z])([a-z])\b", _maybe_subscript_pair, text)
    # Fix PDF layout artifact: _{X},Y → _{X,Y} (subscript brace closed before comma-index)
    # [^{}] guard prevents false matches inside nested subscripts like _{t_{f},t_{r}}
    # Negative lookahead (?![_\\{]) prevents fixing legitimate tuples like (z_{1},z_{2})
    text = re.sub(
        r"_\{([^{}]{1,20})\},([A-Za-z0-9]+)(?![_\\{])",
        r"_{\1,\2}",
        text,
    )
    text = re.sub(
        r"\\sum\s+([A-Za-z_{}\\][A-Za-z0-9_{}\\]*)", r"\\sum_{\1}", text
    )
    text = re.sub(
        r"\\prod\s*([A-Za-z0-9_{}\\]*)",
        lambda m: r"\prod" + (rf"_{{{m.group(1)}}}" if m.group(1) else ""),
        text,
    )
    return text


def _maybe_subscript_pair(match):
    """Subscript common two-letter tokens like 'vx', 'Hc' but skip 'dt', 'ln'."""
    token = match.group(0)
    if token in {"dt", "dx", "dy", "ln", "tr", "pm"}:
        return token
    if token[0] in {"v", "x", "y", "p", "q", "L", "H", "c", "h", "V", "W"}:
        return f"{token[0]}_{{{token[1]}}}"
    return token


def _latexify_simple_fractions(text):
    """Convert simple division patterns to \\frac{}{} form."""
    text = re.sub(r"\b-?1/([A-Za-z0-9_{}\\]+)", r"-\\frac{1}{\1}", text)
    text = re.sub(r"\b([0-9]+)/([0-9]+)\b", r"\\frac{\1}{\2}", text)
    text = re.sub(r"-1\s+2(?=\{)", r"-\\frac{1}{2}", text)
    text = re.sub(r"\\sqrt\s*([A-Za-z0-9_{}\\]+)", r"\\sqrt{\1}", text)
    return text


def _latexify_flattened_functions(text):
    """Repair generic flattened function/operator tokens from PDF text.

    PDF text often places fractions, summation limits, matrix brackets, and
    transposes on neighbouring rows or drops LaTeX command prefixes.  This
    function restores the most common cases.
    """
    text = re.sub(r"(?<=\s)!(?=\s|$)", "", text)
    text = re.sub(r"(?<!\\)\bsin\s*\(", r"\\sin(", text)
    text = re.sub(r"(?<!\\)\bcos\s*\(", r"\\cos(", text)
    text = re.sub(r"\b([A-Za-z])sin\s*\(", r"\1\\sin(", text)
    text = re.sub(r"\b([A-Za-z])cos\s*\(", r"\1\\cos(", text)
    text = re.sub(r"\\pic\b", r"\\pi c", text)
    text = re.sub(r"\\leftarrown\b", r"\\leftarrow n", text)
    # "Z" used as integral sign when surrounded by differentials
    text = re.sub(
        r"\bZ\s+([A-Za-z0-9\\_{}+-]+)\s+([A-Za-z0-9\\_{}+-]+)"
        r"\s+d(\\tau|\\theta|t|x|y)",
        r"\\int_{\2}^{\1} d\3",
        text,
    )
    return text


def _latexify_brackets(text):
    """Normalise bracket and ket notation."""
    text = text.replace("{", "{").replace("}", "}")
    text = text.replace("|0\\rangle", r"|0\rangle")
    text = text.replace("|1\\rangle", r"|1\rangle")
    text = re.sub(r"\|\s*\+\\rangle", r"|+\rangle", text)
    text = re.sub(r"\\vec([A-Za-z])", r"\\vec{\1}", text)
    return text


# ── Prose prefix/suffix removal ───────────────────────────────────────────────

def _strip_prose_prefix(text):
    """Remove short prose lead-ins when the same line contains the equation.

    PDF text blocks sometimes merge a paragraph ending with a following
    equation into one string.  This function strips the prose portion so only
    the math content is stored.
    """
    cleaned = text.strip()
    known_prefixes = (
        "following identity:",
        "the following identity:",
        "using this identity,",
        "relationship:",
    )
    lowered = cleaned.lower()
    for prefix in known_prefixes:
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            return re.sub(r"^\[[0-9,\s-]+\]\s*", "", cleaned).strip()

    cleaned = re.sub(r"^[A-Za-z]+\[[0-9,\s-]+\]\s+", "", cleaned)
    cleaned = re.sub(r"^\[[0-9,\s-]+\]\s*", "", cleaned)

    # Drop prose fragments that precede the first identifiable math token
    math_anchor = re.search(
        r"(\\(?:hat|rho|sigma|tau|lambda|Delta|vec)"
        r"|[ˆ|]|[A-Z]\s*[=_(]"
        r"|[A-Za-z]+\(.*?\)\s*="
        r"|m\\?lambda)",
        cleaned,
    )
    if math_anchor and math_anchor.start() > 0:
        lead = cleaned[:math_anchor.start()]
        if re.search(r"[A-Za-z]{3,}", lead) and not re.search(
            r"[=+*/∑∫√ρσψθ]", lead
        ):
            return cleaned[math_anchor.start():].strip()
    return cleaned


def _strip_prose_suffix(text):
    """Remove explanatory prose appended after an equation in PDF text."""
    if "=" not in text:
        return text
    prose_cues = (
        " Acting ", " Using ", " Therefore ", " Thus ", " where ", " with ",
    )
    for cue in prose_cues:
        index = text.find(cue)
        if index > 0:
            before = text[:index].strip()
            after = text[index + len(cue):]
            if not re.search(r"[A-Za-z0-9_{}\\]+\s*=", after[:80]):
                return before
    return text


# ── Full conversion pipeline ──────────────────────────────────────────────────

def _postprocess_pdf_equation(text):
    """Clean common flattened-PDF math patterns (paper-independent rules only)."""
    if "\n" in str(text):
        return "\n".join(
            _postprocess_pdf_equation(row)
            for row in str(text).splitlines()
            if row.strip()
        )
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(
        r"\blim\s+([A-Za-z]→[+∞\-0-9]+)−1\s+([A-Za-z])\s+ln",
        r"lim_{\1} -1/\2 ln",
        cleaned,
    )
    cleaned = re.sub(
        r"\blim\s+([A-Za-z]→[+∞\-0-9]+)-1\s+([A-Za-z])\s+ln",
        r"lim_{\1} -1/\2 ln",
        cleaned,
    )
    cleaned = re.sub(r"\bd2([A-Za-z0-9()]+)\s+dt2\b", r"d2\1/dt2", cleaned)
    cleaned = re.sub(r"\bd([A-Za-z0-9()]+)\s+dt\s*\+", r"d\1/dt +", cleaned)
    cleaned = cleaned.replace("X Lµ", "Σ_µ Lµ")
    cleaned = cleaned.replace(" X µ ", " Σ_µ ")
    return cleaned


def _pdf_text_to_latex_like(text):
    """Convert PDF-extracted math text to general LaTeX-like notation.

    Multi-line input is wrapped in an ``aligned`` environment.  Single-line
    input is processed by ``_pdf_text_to_latex_like_single``.

    This is intentionally *not* a reconstruction system.  It only converts
    common Unicode math glyphs and layout tokens into stable TeX-like syntax.
    If the PDF extraction dropped structure, the output remains low-confidence
    rather than inventing missing formula parts.
    """
    raw_rows = [row.strip() for row in str(text).splitlines() if row.strip()]
    if len(raw_rows) >= 2:
        rows = [_pdf_text_to_latex_like_single(row) for row in raw_rows]
        rows = [row for row in rows if row]
        if len(rows) >= 2:
            return r"\begin{aligned} " + r" \\ ".join(rows) + r" \end{aligned}"
    return _pdf_text_to_latex_like_single(text)


def _pdf_text_to_latex_like_single(text):
    """Convert one PDF-extracted math row to stable LaTeX-like notation."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = _strip_prose_prefix(cleaned)
    cleaned = _replace_unicode_math_symbols(cleaned)
    cleaned = _latexify_known_operators(cleaned)
    cleaned = _latexify_hats_and_daggers(cleaned)
    cleaned = _latexify_derivatives(cleaned)
    cleaned = _latexify_indices(cleaned)
    cleaned = _latexify_simple_fractions(cleaned)
    cleaned = _latexify_flattened_functions(cleaned)
    cleaned = _strip_prose_suffix(cleaned)
    cleaned = _latexify_brackets(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ── Small internal helpers ────────────────────────────────────────────────────

def _latexify_symbol_token(token):
    """Add subscript to a simple alphanumeric token, e.g. ``x2`` → ``x_{2}``."""
    return re.sub(r"([A-Za-z])(\d+)", r"\1_{\2}", token)


def _subscript(value):
    """Return a LaTeX subscript group or empty string."""
    return rf"_{{{value}}}" if value else ""
