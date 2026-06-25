"""Deterministic enrichment orchestrator.

This file intentionally stays small. The project stages are split into:

* `symbol_extraction.py`
* `meaning_extraction.py`
* `relation_detection.py`
* `audit_trail.py`

The historical import path is preserved, so existing scripts can keep importing
`enrich_equations` from this module.
"""

from src.audit_trail import add_relation_audit
from src.audit_trail import build_enrichment_audit
from src.meaning_extraction import extract_meaning
from src.relation_detection import build_relations
from src.symbol_extraction import extract_symbols
from src.enrichment_common import _context


def enrich_equations(equations):
    """Fill meaning, symbols, relations, and enrichment audits."""
    for equation in equations:
        context = _context(equation)
        symbols, symbol_audit = extract_symbols(
            equation["equation"],
            context,
            equation.get("identifier_candidates"),
        )
        meaning, confidence, meaning_audit = extract_meaning(equation)
        equation["symbols"] = symbols
        equation["meaning"] = meaning
        equation["meaning_confidence"] = confidence
        # Per-symbol provenance for the audit trail: start with the per-equation
        # evidence; later paper-wide / inheritance passes add their own notes.
        # (symbol_audit is a plain string when the equation has no symbols.)
        equation["symbol_provenance"] = (
            dict(symbol_audit.get("_symbol_evidence", {}))
            if isinstance(symbol_audit, dict) else {}
        )
        equation["audit_enrichment"] = build_enrichment_audit(
            meaning_audit,
            symbol_audit,
            confidence,
            symbols,
            equation.get("audit_identifiers"),
        )

    _clear_repeated_meanings(equations)

    relation_map, relation_audits = build_relations(equations)
    for equation in equations:
        eq_number = equation["eq_number"]
        equation["relations"] = relation_map.get(eq_number, {})
        add_relation_audit(equation, relation_audits)

    return equations


def _clear_repeated_meanings(equations):
    """Clear meanings that are identical across ≥4 equations in the same paper.

    When the extractor assigns the same phrase to many equations it has latched
    onto background context rather than the specific equation name.  Clearing
    the meaning is more honest than repeating a boilerplate phrase.
    """
    non_empty = [eq["meaning"] for eq in equations if eq.get("meaning")]
    if not non_empty:
        return
    from collections import Counter
    counts = Counter(non_empty)
    repeated = {phrase for phrase, cnt in counts.items() if cnt >= 4}
    for eq in equations:
        if eq.get("meaning") in repeated:
            eq["meaning"] = ""
            eq["meaning_confidence"] = "low"


__all__ = [
    "build_relations",
    "enrich_equations",
    "extract_meaning",
    "extract_symbols",
]
