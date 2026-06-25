r"""Text-based symbol definition extraction utilities.

Provides the shared text utilities and extraction stages (1–3) used by
``symbol_definition_extraction.py``:

* Unicode/KaTeX noise scrubbing and conversion
* Symbol–text key matching
* Definition text cleaning
* Stage 1 — where-clause parser
* Stage 2 — definitor-verb sentence scan
* Stage 3 — appositive "the <NP> <symbol>" scan

All functions work on raw text strings; none use any language model.
"""

import re
import unicodedata

from src.symbol_extraction import _normalize_symbol_key, _latex_atoms


# ── Definitor-verb pattern ────────────────────────────────────────────────────

_DEFVERB_PATTERN = (
    r"(?:is|are)\s+(?:(?:the|a|an)\s+)?(?:defined\s+as\s+)?"
    r"|being\s+(?:(?:the|a|an)\s+)?"
    r"|called\s+(?:(?:the|a|an)\s+)?"
    r"|known\s+as\s+(?:(?:the|a|an)\s+)?"
    r"|denotes?\s+(?:(?:the|a|an)\s+)?"
    r"|represents?\s+(?:(?:the|a|an)\s+)?"
    r"|stands?\s+for\s+(?:(?:the|a|an)\s+)?"
    r"|refers?\s+to\s+(?:(?:the|a|an)\s+)?"
    r"|corresponds?\s+to\s+(?:(?:the|a|an)\s+)?"
    r"|is\s+defined\s+as\s+"
    r"|determine[sd]?\s+(?:(?:the|a|an)\s+)?"
    r"|describe[sd]?\s+(?:(?:the|a|an)\s+)?"
)

_DEFVERB_RE = re.compile(r"\b(?:" + _DEFVERB_PATTERN + r")", re.IGNORECASE)


# ── Unicode / ar5iv noise scrubbing ──────────────────────────────────────────

_MATH_UNICODE_RE = re.compile(r"[\U0001D400-\U0001D7FF\U00002100-\U0000214F]")

_GREEK_NAME_MAP = {
    "α": "alpha", "β": "beta",  "γ": "gamma",   "δ": "delta",
    "ε": "epsilon","ζ": "zeta", "η": "eta",      "θ": "theta",
    "ι": "iota",   "κ": "kappa","λ": "lambda",   "μ": "mu",
    "ν": "nu",     "ξ": "xi",   "π": "pi",       "ρ": "rho",
    "σ": "sigma",  "τ": "tau",  "υ": "upsilon",  "φ": "phi",
    "χ": "chi",    "ψ": "psi",  "ω": "omega",
    "Γ": "Gamma",  "Δ": "Delta","Θ": "Theta",    "Λ": "Lambda",
    "Ξ": "Xi",     "Π": "Pi",   "Σ": "Sigma",    "Υ": "Upsilon",
    "Φ": "Phi",    "Ψ": "Psi",  "Ω": "Omega",
    "∆": "Delta",
}

_SEPARATOR_CHARS = frozenset(' \t\n\r_\\{}()[],;:!?=+*/^')

_KATEX_BLOCK_RE = re.compile(
    r"start_POST(?:SUBSCRIPT|SUPERSCRIPT).*?end_POST(?:SUBSCRIPT|SUPERSCRIPT)",
    re.DOTALL,
)
_KATEX_TOKEN_RE = re.compile(
    r"\b(?:italic|roman|bold|normal)_[A-Za-z0-9]+"
    r"|\bPOST(?:SUBSCRIPT|SUPERSCRIPT)\b"
    r"|\b(?:subscript|superscript)\b",
    re.IGNORECASE,
)


def _convert_unicode_math(text):
    """Convert math-Unicode in context text to ASCII names instead of deleting."""
    if not text:
        return text
    text = _MATH_UNICODE_RE.sub(lambda m: unicodedata.normalize("NFKD", m.group()), text)
    if not any(c in _GREEK_NAME_MAP for c in text):
        return text
    result = []
    prev_greek = False
    for char in text:
        if char in _GREEK_NAME_MAP:
            name = _GREEK_NAME_MAP[char]
            if result and result[-1] not in _SEPARATOR_CHARS:
                if prev_greek or result[-1].isalnum():
                    result.append("_")
            result.append(name)
            prev_greek = True
        elif char.isalnum():
            if prev_greek:
                result.append("_")
            result.append(char)
            prev_greek = False
        else:
            result.append(char)
            prev_greek = False
    return "".join(result)


def _scrub_ar5iv_noise(text):
    r"""Remove KaTeX verbose-annotation artifacts from ar5iv-rendered prose text."""
    text = _KATEX_BLOCK_RE.sub(" ", text)
    text = _KATEX_TOKEN_RE.sub(" ", text)
    text = _convert_unicode_math(text)
    text = text.replace("​", "")
    return re.sub(r"\s+", " ", text).strip()


# ── Symbol ↔ text matching ────────────────────────────────────────────────────

def _match_score(key, norm_atom):
    """Score match between a normalised atom and a symbol key (0–3)."""
    if not norm_atom or not key:
        return 0
    if key == norm_atom:
        return 3
    _strip = re.compile(r"_[a-z0-9]+$")
    key_base = _strip.sub("", key)
    norm_base = _strip.sub("", norm_atom)
    if key_base and key_base == norm_base:
        return 2
    key_root = re.split(r"[_^]", key)[0]
    norm_root = re.split(r"[_^]", norm_atom)[0]
    if key_root and key_root == norm_root:
        return 1
    return 0


def _find_best_key(prefix_text, symbol_keys):
    """Find which symbol key the prefix text refers to."""
    scrubbed = _scrub_ar5iv_noise(prefix_text)
    scrubbed_norm = re.sub(r"\s+", " ", scrubbed).strip()
    atoms = _latex_atoms(scrubbed_norm)

    best_key = None
    best_score = 0
    for atom in reversed(atoms):
        norm = _normalize_symbol_key(atom)
        for key in symbol_keys:
            score = _match_score(key, norm)
            if score > best_score:
                best_score = score
                best_key = key

    if best_score == 0:
        free = re.sub(r"[_^]\{[^{}]*\}", " ", scrubbed)
        free = re.sub(r"[_^]\\?[A-Za-z0-9]+", " ", free)
        for key in symbol_keys:
            haystack = scrubbed if len(key) > 1 else free
            pattern = r"(?<![A-Za-z0-9])" + re.escape(key) + r"(?![A-Za-z0-9])"
            if re.search(pattern, haystack):
                return key

    return best_key if best_score >= 2 else None


def _find_matching_keys(prefix_text, symbol_keys):
    """Return every symbol key explicitly mentioned in prefix_text."""
    scrubbed = _scrub_ar5iv_noise(prefix_text)
    scrubbed = re.sub(r"\s*\([^)]{1,5}\)", "", scrubbed)
    scrubbed_norm = re.sub(r"\s+", " ", scrubbed).strip()
    atoms = _latex_atoms(scrubbed_norm)

    scores = {key: 0 for key in symbol_keys}
    for atom in atoms:
        norm = _normalize_symbol_key(atom)
        for key in symbol_keys:
            scores[key] = max(scores[key], _match_score(key, norm))

    exact = [key for key in symbol_keys if scores.get(key, 0) >= 3]
    if exact:
        return exact

    matched = [key for key in symbol_keys if scores.get(key, 0) >= 2]
    if matched:
        return matched

    matched = []
    free = re.sub(r"[_^]\{[^{}]*\}", " ", scrubbed)
    free = re.sub(r"[_^]\\?[A-Za-z0-9]+", " ", free)
    for key in symbol_keys:
        haystack = scrubbed if len(key) > 1 else free
        pattern = r"(?<![A-Za-z0-9])" + re.escape(key) + r"(?![A-Za-z0-9])"
        if re.search(pattern, haystack):
            matched.append(key)
    return matched


# ── Definition text cleanup ───────────────────────────────────────────────────

_COMMA_NEW_DEF_RE = re.compile(
    r",\s+(?=(?:\S+\s+){1,4}(?:is|are|denotes?|represents?|stands?\s+for"
    r"|refers?\s+to|determine[sd]?|describe[sd]?))"
)
_AND_NEW_DEF_RE = re.compile(
    r"\s+and\s+(?=\S{1,15}"
    r"(?:\s*\([^)]{0,15}\))?"
    r"\s+(?:is|are|denotes?|represents?|stands?\s+for"
    r"|refers?\s+to|determine[sd]?|describe[sd]?))"
)
_LATEX_IN_DEF_RE = re.compile(
    r"\\[A-Za-z]+\{[^}]*\}"
    r"|\\[A-Za-z]+"
    r"|\{[^}]*\}"
    r"|\[[^\]]*\]"
    r"|\$[^$]*\$"
)
_PROPERTY_START_RE = re.compile(
    r"^(?:given\s+by|expressed\s+(?:as|by)|parametri[sz]ed\s+by"
    r"|characterized\s+by|defined\s+by|obtained\b|related\s+by"
    r"|non-|generally\b|typically\b|usually\b|often\b|approximately\b"
    r"|continuously\b|directly\b|purely\b|simply\b|formally\b"
    r"|dominated\b|assumed\b|chosen\b|normalized\b|written\b|based\b"
    r"|invariant\b|complete(?:ly)?\b|required\b|determined\b"
    r"|lost\b|governed\b|illustrated\b|transferred\b|numerically\b"
    r"|markedly\b|overlooked\b|merely\b|essentially\b"
    r"|different\b|equivalent\b|proportional\b|independent\b|dependent\b"
    r"|so\b|thus\b|hence\b|therefore\b|because\b|since\b|always\b|never\b)",
    re.IGNORECASE,
)
_GARBLED_DEF_RE = re.compile(
    r"→|start_ARG|end_ARG|italic_|roman_|bold_|over→|\\\\|…|⟶|⌜|⌝"
    r"|\bover\b\s*$"
    r"|(?:POST)?(?:SUPER|SUB)SCRIPT|caligraphic_|mathchar|relax|displaystyle"
    r"|[⁡-⁤]"
    r"|[Ͱ-Ͽ∀-⋿]"
)
_CITATION_DEF_RE = re.compile(
    r"[\[\]]|\b(?:fig|figs|figure|table|tab|eq|eqs|ref|refs|sec|section"
    r"|appendix|app)\b\.?",
    re.IGNORECASE,
)
_DANGLING_TAIL_RE = re.compile(
    r"\b(?:the|a|an|of|in|on|for|to|by|with|and|or|as|that|which|from"
    r"|is|are|be|been|being|its|their|this|these|those|where|when|enough"
    r"|at|into|onto|than|via|upon|near|between|among|within|without|per"
    r"|represented|given|depending|independent|over)\s*$",
    re.IGNORECASE,
)
_HYPHEN_WRAP_RE = re.compile(r"([A-Za-z])-\s+([a-z])")
_MAX_DEF_WORDS = 9


def _clean_definition(raw):
    """Strip LaTeX and annotation noise from a raw definition string."""
    if not raw:
        return ""
    raw = _HYPHEN_WRAP_RE.sub(r"\1\2", raw)
    if _GARBLED_DEF_RE.search(raw):
        return ""
    raw = re.split(r"\.\s", raw)[0]
    raw = re.split(r";\s", raw)[0]
    raw = _COMMA_NEW_DEF_RE.split(raw, maxsplit=1)[0]
    raw = _AND_NEW_DEF_RE.split(raw, maxsplit=1)[0]
    raw = _LATEX_IN_DEF_RE.sub("", raw)
    raw = raw.replace("{", "").replace("}", "")
    raw = re.sub(r"[†‡∗±×⊗⊕⟩⟨|]", " ", raw)
    raw = re.sub(r"\s*\^\s*", " ", raw)
    raw = re.sub(r"\s*_\s+", " ", raw)
    raw = re.sub(r"(\d+)\s*/\s*(\d+)\s+\1/\2", r"\1/\2", raw)
    raw = re.sub(r"\s+[0-9]\s+", " ", raw)
    raw = re.sub(r"\b([A-Za-z])\s+\1\b", r"\1", raw)
    raw = re.sub(r"\s+", " ", raw).strip().strip(".,;:()")
    if raw.count("(") > raw.count(")"):
        raw = raw[:raw.rindex("(")].strip().strip(".,;:")
    if "\\" in raw or "=" in raw:
        return ""
    if re.search(r"[℀-⅏]", raw):
        return ""
    if len(raw) < 3 or not re.search(r"[A-Za-z]", raw):
        return ""
    if len(raw.split()) == 1 and raw.lower() in {
        'even', 'odd', 'bounded', 'unbounded', 'constant', 'converge', 'diverge',
        'present', 'invalid', 'stable', 'unstable', 'above', 'below', 'derived',
        'met', 'closing', 'large', 'small', 'crucial', 'similar', 'shown',
        'given', 'related', 'valid', 'generic', 'arbitrary', 'recent', 'recently',
        'finite', 'infinite', 'positive', 'negative', 'nonzero',
        'operator', 'operators', 'parameter', 'parameters', 'variable',
        'variables', 'quantity', 'quantities', 'term', 'terms', 'state',
        'states', 'considered', 'presented', 'planted', 'introduced', 'obtained',
        'defined', 'expressed', 'written', 'applied', 'required', 'assumed',
        'chosen', 'used', 'taken', 'fitted', 'updated', 'performed', 'condition',
        'local', 'global', 'fixed', 'free', 'real', 'complex',
    }:
        return ""
    if re.match(r"^(?:following|above|below|aforementioned|preceding|latter|"
                r"former|same|previous|next|presented|shown|listed|depicted)\b", raw, re.IGNORECASE):
        return ""
    if re.search(r"\b(?:when|while|if|once|unless|because|since)\s+"
                 r"(?:they|it|we|one|the|this|these|those)\b", raw, re.IGNORECASE):
        return ""
    if re.search(r"\b(?:said|shown|known|taken|assumed|considered|believed)\s+to\b",
                 raw, re.IGNORECASE):
        return ""
    tokens = raw.split()
    if len(tokens) >= 2 and sum(1 for t in tokens if len(t) <= 2) > len(tokens) / 2:
        return ""
    if re.search(r"\bi\.?e\b|\be\.?g\b|:", raw):
        return ""
    if re.search(r"\b(?:our|we|us)\b", raw, re.IGNORECASE):
        return ""
    raw = re.split(
        r"\s+(?:"
        r"given\s+by|defined\s+as|expressed\s+(?:as|by)|such\s+that"
        r"|which|that|where|whose|whom"
        r"|if|when|once|unless|because|whenever|provided"
        r"|according|followed|combined|together|along"
        r"|describing|representing|denoting|corresponding|associated|acting"
        r"|satisfying|characteri[sz]ing|governing|encoding|indicating|accounting"
        r"|capturing|containing|connecting|relating|obeying|measuring|quantifying"
        r"|obtained|written|evaluated|induced|generated|appearing|defined|used"
        r"|given|taken|chosen|assumed|related"
        r")\b",
        raw, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    if raw.lower() in {'updated', 'performed', 'considered', 'presented', 'fitted',
                       'local', 'global', 'obtained', 'defined', 'given', 'used'}:
        return ""
    if re.match(r"^(?:where|such\s+that|with|while)\b", raw, re.IGNORECASE):
        return ""
    if re.match(r"^(?:of|in|on|for|to|by|with|as|from|and|or|at)\b", raw, re.IGNORECASE):
        return ""
    if _CITATION_DEF_RE.search(raw):
        return ""
    if _PROPERTY_START_RE.match(raw):
        return ""
    if re.search(r"\b(?:is|are|was|were)\s+(?:the|a|an)\b.*\bto\s+\w+\b", raw, re.IGNORECASE):
        return ""
    if re.search(r"\b(?:which|that)\s+(?:\w+\s+){0,2}(?:involves?|acts?|gives?"
                 r"|describes?|represents?|denotes?|refers?|favou?rs?|amounts?"
                 r"|increases?|decreases?|usually|only)\b",
                 raw, re.IGNORECASE):
        return ""
    if re.search(r"\bthen\s+it\s+is\b|\bamounts?\s+to\b", raw, re.IGNORECASE):
        return ""
    if re.search(r"\beach\s+other\b|\bone\s+another\b", raw, re.IGNORECASE):
        return ""
    if len(raw.split()) > _MAX_DEF_WORDS:
        return ""
    if _DANGLING_TAIL_RE.search(raw):
        return ""
    return raw


# ── Stage 1: where-clause parser ─────────────────────────────────────────────

def _split_where_items(clause_text):
    """Split a where-clause into individual symbol-definition items."""
    items = []
    depth = 0
    pdepth = 0
    buf = []
    total_chars = 0

    for ch in clause_text:
        if ch == "{":
            depth += 1
            buf.append(ch)
        elif ch == "}":
            depth = max(depth - 1, 0)
            buf.append(ch)
        elif ch == "(":
            pdepth += 1
            buf.append(ch)
        elif ch == ")":
            pdepth = max(pdepth - 1, 0)
            buf.append(ch)
        elif ch in ",;\n" and depth == 0 and pdepth == 0:
            items.append("".join(buf))
            total_chars += len(buf)
            buf = []
            if total_chars > 1200:
                break
        else:
            buf.append(ch)

    if buf:
        items.append("".join(buf))

    return [s.strip() for s in items if s.strip()]


def _parse_where_clause(context, symbol_keys):
    """Stage 1 – extract definitions from *where* … clauses."""
    results = {}

    for m_where in re.finditer(r"\b(?:where|here)\b", context, re.IGNORECASE):
        clause_text = context[m_where.end(): m_where.end() + 1200]
        items = _split_where_items(clause_text)
        pending_keys = []

        for item in items:
            item_clean = _scrub_ar5iv_noise(item)
            m_verb = _DEFVERB_RE.search(item_clean)

            if m_verb:
                prefix = item_clean[: m_verb.start()]
                suffix = item_clean[m_verb.end():]
                defn = _clean_definition(suffix)

                if defn:
                    nested_where = re.search(r"\bwhere\b", prefix, re.IGNORECASE)
                    if nested_where:
                        prefix = prefix[nested_where.end():]
                    matched_keys = _find_matching_keys(prefix, symbol_keys)
                    all_keys = list(pending_keys)
                    for matched in matched_keys:
                        if matched and matched not in all_keys:
                            all_keys.append(matched)

                    snippet = item_clean.strip()
                    for k in all_keys:
                        if k not in results:
                            results[k] = (defn, snippet)

                    and_parts = _AND_NEW_DEF_RE.split(suffix, maxsplit=1)
                    if len(and_parts) > 1:
                        rest = and_parts[1]
                        m_verb2 = _DEFVERB_RE.search(rest)
                        if m_verb2:
                            prefix2 = rest[: m_verb2.start()]
                            nw2 = re.search(r"\bwhere\b", prefix2, re.IGNORECASE)
                            if nw2:
                                prefix2 = prefix2[nw2.end():]
                            suffix2 = rest[m_verb2.end():]
                            defn2 = _clean_definition(suffix2)
                            if defn2:
                                matched2 = _find_matching_keys(prefix2, symbol_keys)
                                for k in matched2:
                                    if k not in results:
                                        results[k] = (defn2, snippet)

                pending_keys = []

            else:
                keys = _find_matching_keys(item_clean, symbol_keys)
                prose_words = re.findall(r"[A-Za-z]{2,}", re.sub(r"\\[A-Za-z]+", " ", item_clean))
                if keys and len(prose_words) <= 3:
                    for key in keys:
                        if key not in pending_keys:
                            pending_keys.append(key)
                elif item_clean:
                    pending_keys = []

    return results


# ── Stage 2: definitor-verb sentence scan ────────────────────────────────────

def _scan_definitor_sentences(context, symbol_keys):
    """Stage 2 – scan every sentence for "[symbol] <verb> [definition]" patterns."""
    results = {}
    sentences = re.split(r"(?<=[.!?])\s+|\n", context)

    for sentence in sentences:
        if re.search(r"\bwhere\b", sentence, re.IGNORECASE):
            continue
        if re.search(r"\bi\.e\.\s*,\s*where\b", sentence, re.IGNORECASE):
            continue
        clauses = re.split(r";\s*", sentence)
        for clause in clauses:
            clean = _scrub_ar5iv_noise(clause)
            m_verb = _DEFVERB_RE.search(clean)
            if not m_verb:
                continue
            prefix = clean[: m_verb.start()]
            suffix = clean[m_verb.end():]
            keys = _find_matching_keys(prefix, symbol_keys)
            defn = _clean_definition(suffix)
            if keys and defn:
                for key in keys:
                    if key in results:
                        continue
                    results[key] = (defn, clean.strip())

    return results


# ── Stage 3: appositive "the [noun] [symbol]" scan ───────────────────────────

_APPOS_RE = re.compile(
    r"\bthe\s+"
    r"([A-Za-z][A-Za-z0-9\s\-]{2,40}?)\s+"
    r"("
    r"\\[A-Za-z]+(?:\{[^}]*\})?(?:[_^]\{?[A-Za-z0-9]*\}?)*"
    r"|[A-Z][A-Za-z0-9]*(?:[_^][A-Za-z0-9]+)?"
    r")",
)


def _scan_appositive(context, symbol_keys):
    """Stage 3 – find "the <noun phrase> <symbol>" appositive patterns."""
    results = {}
    clean = _scrub_ar5iv_noise(context)

    for m in _APPOS_RE.finditer(clean):
        noun_phrase = m.group(1).strip()
        symbol_text = m.group(2).strip()
        key = _find_best_key(symbol_text, symbol_keys)
        if key and key not in results:
            defn = _clean_definition(noun_phrase)
            if defn:
                s0 = clean.rfind(". ", 0, m.start())
                s0 = s0 + 2 if s0 >= 0 else max(0, m.start() - 80)
                s1 = clean.find(". ", m.end())
                s1 = s1 if s1 >= 0 else min(len(clean), m.end() + 80)
                snippet = clean[s0:s1].strip()
                results[key] = (defn, snippet)

    return results
