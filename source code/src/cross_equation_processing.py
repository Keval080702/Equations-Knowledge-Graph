"""Paper-wide post-enrichment processing.

After per-equation enrichment runs, these passes pool all context from a
paper to fill symbols that were not defined near their first appearance,
and propagate definitions between equations.

----------
* ``recover_definitions_paper_wide(equations)``
* ``apply_cross_eq_symbol_inheritance(equations)``
"""

import re

from src.enrichment_common import _context
from src.symbol_definition_extraction import (
    extract_definitions_from_text,
    extract_nounphrase_appositives,
)
from src.symbol_structural_analysis import _bound_only_letters


def _add_postprocess_audit(equation, method, message):
    audit = equation.setdefault("audit_enrichment", {})
    audit.setdefault(method, []).append(_short(message, 360))


def _short(text, limit):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def recover_definitions_paper_wide(equations):
    """Fill still-empty symbols using definition text pooled from all equations.

    A symbol is often introduced in prose near one equation but only appears
    as a variable in a later equation.  This pass pools the surrounding text
    of all equations and re-runs the document-text stages over that prose.
    Only empty symbols are filled; bound-only index letters are excluded.
    """
    if not equations:
        return
    seen, chunks = set(), []
    for eq in equations:
        ctx = _context(eq)
        if ctx and ctx not in seen:
            seen.add(ctx)
            chunks.append(ctx)
    pooled = re.sub(r"\s+", " ", " ".join(chunks)).strip()
    if not pooled:
        return

    empty_keys = set()
    for eq in equations:
        bound = _bound_only_letters(eq.get("equation", ""))
        for sym, val in eq.get("symbols", {}).items():
            if not val and not (len(sym) == 1 and sym in bound):
                empty_keys.add(sym)
    if not empty_keys:
        return

    doc_defs, doc_evidence = extract_definitions_from_text(sorted(empty_keys), pooled)

    still_empty = sorted(empty_keys - set(doc_defs))
    if still_empty:
        appos, appos_ev = extract_nounphrase_appositives(still_empty, pooled)
        for k, v in appos.items():
            if k not in doc_defs:
                doc_defs[k] = v
                doc_evidence[k] = appos_ev.get(k, "")

    if not doc_defs:
        return
    for eq in equations:
        symbols = eq.get("symbols", {})
        prov = eq.setdefault("symbol_provenance", {})
        for sym in list(symbols.keys()):
            if not symbols[sym] and sym in doc_defs:
                symbols[sym] = doc_defs[sym]
                snip = str(doc_evidence.get(sym, "")).strip()
                prov[sym] = f"paper-wide text '{snip}'" if snip else "paper-wide text"
                _add_postprocess_audit(
                    eq,
                    "recover_definitions_paper_wide",
                    f"{sym}: filled '{doc_defs[sym]}' from paper context",
                )


def apply_cross_eq_symbol_inheritance(equations):
    """Propagate context-derived definitions across a paper's equations.

    (a) Exact-key propagation — the identical symbol is defined in another
        equation (e.g. m='atomic mass' near eq 1 fills empty m in eq 6).
    (b) Conservative base inheritance — a subscripted variant (H_y) inherits
        the base symbol's broad type only when the base itself was defined and
        the definition has a safe type head.  Superscripted variants are
        excluded.
    """
    auto_labels = {
        "summation index", "differential variable", "Lorentz index",
        "Dirac gamma matrix", "probability distribution",
        "Pauli X operator", "Pauli Y operator", "Pauli Z operator",
    }
    _RELATIONAL_RE = re.compile(r"\beach\s+other\b|\bone\s+another\b", re.IGNORECASE)

    exact_defs: dict[str, str] = {}
    exact_src: dict[str, str] = {}
    base_defs: dict[str, tuple[str, str]] = {}
    for eq in equations:
        label = eq.get("eq_number", "?")
        for sym, defn in eq.get("symbols", {}).items():
            if defn and defn not in auto_labels and not _RELATIONAL_RE.search(defn):
                if sym not in exact_defs:
                    exact_defs[sym] = defn
                    exact_src[sym] = label
                base = re.sub(r"[_^].*$", "", sym)
                if sym == base and _base_definition_can_be_inherited(defn):
                    base_defs.setdefault(base, (defn, label))

    for eq in equations:
        symbols = eq.get("symbols", {})
        prov = eq.setdefault("symbol_provenance", {})
        for sym in list(symbols.keys()):
            if symbols[sym]:
                continue
            if sym in exact_defs:
                symbols[sym] = exact_defs[sym]
                prov[sym] = f"same symbol {sym} defined in eq {exact_src[sym]}"
                _add_postprocess_audit(
                    eq,
                    "apply_cross_eq_symbol_inheritance",
                    f"{sym}: filled from same symbol in Eq({exact_src[sym]})",
                )
                continue
            if "^" in sym:
                continue
            base = re.sub(r"[_^].*$", "", sym)
            if base != sym and base in base_defs:
                base_defn, base_src = base_defs[base]
                if not _safe_base_variant(sym, base):
                    continue
                symbols[sym] = base_defn
                prov[sym] = f"inherited from base symbol {base} defined in paper"
                _add_postprocess_audit(
                    eq,
                    "apply_cross_eq_symbol_inheritance",
                    f"{sym}: filled from base symbol {base} in Eq({base_src})",
                )


_SAFE_INHERITANCE_HEADS = {
    "operator",
    "operators",
    "matrix",
    "matrices",
    "state",
    "states",
    "field",
    "fields",
    "function",
    "functions",
    "tensor",
    "tensors",
    "vector",
    "vectors",
}


def _base_definition_can_be_inherited(definition):
    """Return True for broad type definitions that can apply to variants."""
    text = re.sub(r"\s+", " ", str(definition or "").strip().lower())
    if not text:
        return False
    words = re.findall(r"[a-z]+", text)
    if not words:
        return False
    if re.search(r"\b(?:factor|coefficient|constant|index|number|phase|angle|time|"
                 r"frequency|energy|charge|position|momentum|distance|rate|gap)\b", text):
        return False
    return words[-1] in _SAFE_INHERITANCE_HEADS


def _safe_base_variant(symbol, base):
    """Return True when symbol is a simple subscripted variant of base."""
    if not symbol.startswith(base + "_"):
        return False
    suffix = symbol[len(base) + 1:]
    if not suffix or "^" in suffix or "_" in suffix:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9]{1,4}", suffix))
