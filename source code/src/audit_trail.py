"""Audit-trail helpers.

The audit-trail keys are the actual function names used in extraction.
Values are short, meaningful descriptions of what each function found.
"""

import os
import re
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
# Allow a full source sentence in provenance values (symbol definitions and the
# meaning sentence now carry the whole sentence they were read from).
MAX_AUDIT_VALUE_CHARS = 400

AUDIT_METHOD_NAMES = {
    "detect_best_format",
    "download_html",
    "download_pdf",
    "parse_html",
    "parse_pdf",
    "extract_pdf_equations_with_latex_ocr",
    "clean_equation_text",
    "extract_symbol_identifiers",
    "extract_symbols",
    "extract_symbol_definitions",
    "recover_definitions_paper_wide",
    "apply_cross_eq_symbol_inheritance",
    "extract_meaning",
    "build_relations",
}


def build_meaning_audit(meaning, confidence, equation, decision):
    """Return an audit string for extract_meaning().

    Shows the meaning and the **actual source sentence** it was read from
    (``decision['source_sentence']``).  Falls back to the cue evidence only when
    no source sentence is available (e.g. the meaning came from the LHS symbol /
    equation structure rather than prose).
    """
    sentence = _short_audit_text(decision.get("source_sentence", ""), 320)
    if meaning and sentence:
        return f"'{meaning}' from text: '{sentence}' (confidence={confidence})"
    if meaning:
        cue = _short_audit_text(decision.get("evidence", ""), 120)
        if cue:
            return f"'{meaning}' from equation structure ({cue}; confidence={confidence})"
        return f"'{meaning}' (confidence={confidence}; no direct text evidence)"
    return f"no meaning found (confidence={confidence})"


def build_enrichment_audit(
    meaning_audit,
    symbol_audit,
    confidence,
    symbols=None,
    identifier_audit=None,
):
    """Create the enrichment audit block keyed by actual function names.

    Keys map directly to the public functions that produced each result:
    - ``extract_symbol_identifiers`` — symbol tokenisation
    - ``extract_symbols``            — symbols dictionary construction
    - ``extract_meaning``            — equation meaning
    """
    symbols = symbols or {}
    identifiers = ", ".join(symbols) if symbols else "none"
    symbol_audits = symbol_audit if isinstance(symbol_audit, dict) else {}
    audit = {
        "extract_symbol_identifiers": identifier_audit
        or f"found {len(symbols)} symbol(s): {identifiers}",
        "extract_symbols": symbol_audits.get(
            "extract_symbols",
            f"stored {len(symbols)} symbol(s): {identifiers}",
        ),
        "extract_meaning": meaning_audit,
    }
    if symbols:
        audit["extract_symbol_definitions"] = symbol_audits.get(
            "extract_symbol_definitions",
            symbol_audit if isinstance(symbol_audit, str) else "no symbol definitions extracted",
        )
    return audit


def add_relation_audit(equation, relation_audits):
    """Attach build_relations audit text for one equation in-place."""
    eq_number = equation["eq_number"]
    equation["audit_enrichment"]["build_relations"] = relation_audits.get(
        eq_number,
        "no other equations in paper",
    )


def build_output_audit_trail(equation, source_audit):
    """Build the final JSON audit-trail for one equation.

    Keys are actual implemented function names; values trace provenance.
    Single-step methods carry a string; per-item methods carry a list with one
    entry per symbol or relation, including ``not found`` / ``none`` so the
    trail is complete.
    """
    enrichment = equation.get("audit_enrichment", {})
    symbols = equation.get("symbols", {})
    provenance = equation.get("symbol_provenance", {})

    audit = _source_audit_entries(equation, source_audit)   # detect_best_format, download_*
    audit.update(_parse_audit_entries(equation))            # parse_html / parse_pdf
    if "stored_equation" in equation:
        audit["clean_equation_text"] = (
            f"stored equation '{preview_audit_text(equation.get('stored_equation', ''), 90)}'"
        )

    # Meaning first — the sentence the meaning was read from.
    audit["extract_meaning"] = compact_audit_text(
        enrichment.get("extract_meaning", "no meaning extracted")
    )
    # Then every symbol token found in the equation.
    audit["extract_symbol_identifiers"] = (
        [f"found {key}" for key in symbols] or ["no symbols found"]
    )
    audit["extract_symbols"] = compact_audit_text(
        enrichment.get(
            "extract_symbols",
            f"stored {len(symbols)} symbol(s): {', '.join(symbols) if symbols else 'none'}",
        ),
        220,
    )
    # Then each symbol's definition with its source sentence (or "not found").
    audit["extract_symbol_definitions"] = _symbol_definition_audit_list(symbols, provenance)
    for method in (
        "recover_definitions_paper_wide",
        "apply_cross_eq_symbol_inheritance",
    ):
        if enrichment.get(method):
            audit[method] = enrichment[method]
    # Finally one entry per other equation, including "none".
    audit["build_relations"] = _relations_audit_list(
        equation.get("relations", {}),
        enrichment.get("build_relations"),
    )

    final_audit = {}
    for key, value in audit.items():
        if key not in AUDIT_METHOD_NAMES:
            continue
        final_audit[key] = value if isinstance(value, list) else compact_audit_text(value, 220)
    validate_audit_trail(final_audit)
    return final_audit


def _extract_snippet(provenance):
    """Pull the quoted source-sentence out of a raw provenance string.

    Per-equation evidence looks like ``"where-clause: 'X is the Y …' -> 'Y'"``;
    paper-wide looks like ``"paper-wide text 'X is the Y …'"``.  Returns the
    quoted sentence when present, else ``None`` (structural / inheritance notes
    carry no source sentence).
    """
    m = re.search(r"'([^']{3,})'", str(provenance or ""))
    return m.group(1).strip() if m else None


def _symbol_definition_audit_list(symbols, provenance):
    """Return one provenance line per symbol.

    Format: ``"<sym> is <definition> --> from text: '<whole source sentence>'"``
    when the definition was read from prose; structural / inheritance notes are
    appended as ``"<sym> is <definition> --> <reason>"``; and undefined symbols
    are recorded as ``"<sym>: not found"``.
    """
    items = []
    for sym, defn in symbols.items():
        if not defn:
            items.append(f"{sym}: not found")
            continue
        prov = str(provenance.get(sym, ""))
        sentence = _extract_snippet(prov)
        if sentence:
            line = f"{sym} is {defn} --> from text: '{sentence}'"
        elif prov:
            line = f"{sym} is {defn} --> {prov}"
        else:
            line = f"{sym} is {defn}"
        items.append(compact_audit_text(line, MAX_AUDIT_VALUE_CHARS))
    return items or ["no symbols found"]


def _relations_audit_list(relations, relation_audit=None):
    """Return one audit entry per other equation."""
    if relation_audit:
        items = relation_audit if isinstance(relation_audit, list) else [relation_audit]
        return [compact_audit_text(item, MAX_AUDIT_VALUE_CHARS) for item in items]

    items = []
    for other, rel in relations.items():
        grade = (rel or {}).get("grade", "none")
        desc = str((rel or {}).get("description", "")).strip()
        if grade == "none" or not desc:
            items.append(f"vs eq{other}: {grade}")
        else:
            items.append(f"vs eq{other}: {grade} --> {desc}")
    return items or ["no other equations in paper"]


def validate_audit_trail(audit):
    """Validate final audit-trail method keys and values.

    Each value is either a non-empty short string (single-step methods like
    ``parse_html`` / ``extract_meaning``) or a non-empty list of short strings
    (per-item provenance for ``extract_symbol_identifiers`` /
    ``extract_symbol_definitions`` / ``build_relations``).
    """
    if not isinstance(audit, dict) or not audit:
        raise ValueError("audit-trail must be a non-empty dictionary")
    for method, value in audit.items():
        if method not in AUDIT_METHOD_NAMES:
            raise ValueError(f"audit-trail key is not an implemented extraction method: {method}")
        if isinstance(value, list):
            if not value or not all(isinstance(v, str) and v.strip() for v in value):
                raise ValueError(f"audit-trail list for {method} must be non-empty strings")
            for v in value:
                if len(v) > MAX_AUDIT_VALUE_CHARS:
                    raise ValueError(f"audit-trail item for {method} too long ({len(v)} chars)")
        elif isinstance(value, str):
            if not value.strip():
                raise ValueError(f"audit-trail value for {method} must be a non-empty string")
            if len(value) > MAX_AUDIT_VALUE_CHARS:
                raise ValueError(f"audit-trail value for {method} is too long ({len(value)} chars)")
        else:
            raise ValueError(f"audit-trail value for {method} must be a string or list of strings")


def compact_audit_text(text, limit=None):
    """Return a one-line audit value, optionally shortened."""
    text = str(text or "")
    text = text.replace(str(PROJECT_DIR) + os.sep, "")
    text = re.sub(r"\s+", " ", text).strip()
    if limit and len(text) > limit:
        return text[: max(0, limit - 3)].rstrip() + "..."
    return text


def preview_audit_text(text, limit):
    """Return a short preview suitable for audit strings."""
    text = compact_audit_text(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _format_label(source):
    """Map an internal source marker to a human label (HTML / PDF)."""
    if source in ("pdf", "latex_ocr"):
        return "PDF"
    if source == "html":
        return "HTML"
    return source or "unknown"


def _source_decision_audit(equation, source_audit):
    """State which source the EQUATION itself was taken from.

    These acquisition methods describe how the *equation* was obtained, so only
    the equation's order/text source is reported — not the context source, which
    may differ (HTML prose can surround a PDF-extracted equation) and is recorded
    separately by extract_meaning / extract_symbol_definitions.
    """
    order_source = equation.get("equation_order_source", "unknown")
    text_source = equation.get("equation_text_source", "unknown")
    parts = [f"equation taken from {_format_label(text_source)}"]
    if order_source and order_source != text_source:
        parts.append(f"numbering from {_format_label(order_source)}")
    context_source = equation.get("context_source", "")
    if context_source and context_source not in {order_source, text_source}:
        parts.append(f"context from {_format_label(context_source)}")
    return "; ".join(parts)


def _source_audit_entries(equation, source_audit):
    """Return source-acquisition audit entries keyed by called functions.

    Only the download method for the source actually used to take the EQUATION
    (its order/text source) is reported — so a paper whose equation was read
    from PDF does not list ``download_html`` (HTML, if used at all, only supplied
    surrounding context, which is recorded in the meaning/definition provenance).
    """
    entries = {
        "detect_best_format": _source_decision_audit(equation, source_audit),
    }
    order_source = equation.get("equation_order_source", "")
    text_source = equation.get("equation_text_source", "")
    context_source = equation.get("context_source", "")
    used = {order_source, text_source, context_source}
    source_text = compact_audit_text(source_audit)

    if "html" in used:
        entries["download_html"] = (
            _download_fragment(source_text, "HTML") or "used cached/downloaded arXiv HTML"
        )
    if "pdf" in used or "latex_ocr" in used:
        entries["download_pdf"] = (
            _download_fragment(source_text, "PDF") or "used cached/downloaded arXiv PDF"
        )
    return entries


def _download_fragment(text, source_name):
    """Extract the short fragment emitted by download_html/download_pdf."""
    if not text:
        return ""
    fragments = [part.strip() for part in text.split(";") if part.strip()]
    pattern = rf"\b{re.escape(source_name)} (?:loaded|downloaded|download failed|download error)\b"
    matches = [
        fragment for fragment in fragments
        if re.search(pattern, fragment, flags=re.IGNORECASE)
    ]
    if not matches:
        return ""
    return compact_audit_text("; ".join(matches), 170)


def _parse_audit_entries(equation):
    """Return parse/OCR audit entries for the methods that produced equation."""
    eq_number = equation.get("eq_number", "?")
    order_source = equation.get("equation_order_source") or equation.get("context_source") or "pdf"
    text_source = equation.get("equation_text_source") or equation.get("context_source") or order_source
    context_source = equation.get("context_source") or ""
    parse_value = compact_audit_text(equation.get("audit_extract", ""))
    if not parse_value:
        eq_preview = preview_audit_text(equation.get("equation", ""), 80)
        parse_value = f"eq({eq_number})='{eq_preview}'"

    entries = {}
    if order_source == text_source:
        entries[_parse_method_key(text_source)] = parse_value
    else:
        entries[_parse_method_key(order_source)] = f"selected Eq({eq_number}) as canonical order/label"
        if text_source == "latex_ocr":
            entries["extract_pdf_equations_with_latex_ocr"] = parse_value
        else:
            entries[_parse_method_key(text_source)] = parse_value

    if context_source and context_source not in {order_source, text_source}:
        context_value = compact_audit_text(equation.get("audit_context", ""))
        entries[_parse_method_key(context_source)] = (
            context_value
            or f"used Eq({eq_number}) surrounding context from {_format_label(context_source)}"
        )
    return entries


def _parse_method_key(source):
    """Map source marker to the parser method name used in the code."""
    if source == "html":
        return "parse_html"
    return "parse_pdf"


def join_audit(*parts):
    """Join non-empty audit fragments into a single semicolon-separated string."""
    return "; ".join(compact_audit_text(p) for p in parts if p)


__all__ = [
    "add_relation_audit",
    "AUDIT_METHOD_NAMES",
    "build_enrichment_audit",
    "build_meaning_audit",
    "build_output_audit_trail",
    "compact_audit_text",
    "join_audit",
    "validate_audit_trail",
    "preview_audit_text",
]


def _short_audit_text(text, limit):
    """One-line bounded audit text."""
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."
