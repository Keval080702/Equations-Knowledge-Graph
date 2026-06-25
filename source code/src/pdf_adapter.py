"""
Phase 2 — PDF Equation Extraction (Public Adapter)
====================================================

This module is the **public entry point** for PDF-based equation extraction.
It orchestrates the three independent extraction engines, merges their
outputs, applies LaTeX OCR as an optional quality-upgrade pass, and enriches
context with Docling.

Sub-modules
-----------
* ``pdf_math.py``          — LaTeX/Unicode normalisation utilities
* ``pdf_text_helpers.py``  — Text validation, context helpers, raw-text extractor
* ``pdf_engines.py``       — PyMuPDF, pdftohtml XML, and text-layout engines

Usage
-----
::

    from src.pdf_adapter import parse_pdf

    equations = parse_pdf(pdf_bytes, arxiv_id, allow_latex_ocr=False)

Pipeline role
-------------
Phase 2 fallback: called by ``equation_extraction.py`` when no ar5iv HTML is
available for a paper.

Public API
----------
* ``parse_pdf(pdf_bytes, arxiv_id, allow_latex_ocr=False)``
"""

import logging
import re

from src.docling_adapter import enhance_pdf_context_with_docling
from src.latex_ocr_adapter import extract_pdf_equations_with_latex_ocr
from src.pdf_engines import (
    _extract_with_pymupdf_layout,
    _extract_with_pdftohtml_xml,
    _extract_with_pdf_text_layout,
)
from src.pdf_text_helpers import (
    _extract_text_best_effort,
    _looks_incomplete_pdf_fragment,
)
from src.sequence_validation import (
    has_first_equation_prefix as _shared_first_sequence,
    label_sort_key,
    normalize_equation_label,
)

logger = logging.getLogger(__name__)

MAX_EQUATIONS = 7


__all__ = ["parse_pdf"]


# ── Public API ────────────────────────────────────────────────────────────────

def parse_pdf(pdf_bytes, arxiv_id, allow_latex_ocr=False):
    """Extract numbered equations from PDF bytes using a multi-engine approach.

    Runs all three engines, then selects the best output:

    1. If ≥ 2 engines agree on a first-numbered sequence → merge their outputs.
    2. Otherwise use the first engine (in priority order) that produces a valid
       first-numbered sequence.
    3. Applies optional LaTeX OCR to replace equation strings for matched labels.
    4. Enriches context with Docling if available.

    Parameters
    ----------
    pdf_bytes : bytes
        Raw PDF file content.
    arxiv_id : str
        The arXiv paper ID (used for logging and caching).
    allow_latex_ocr : bool
        When True, attempt Nougat/LaTeX-OCR to improve equation strings.

    Returns
    -------
    list[dict]
        Each dict has keys: eq_number, equation, text_before, text_after,
        section, audit_extract, audit_context.
    """
    pymupdf_equations = _extract_with_pymupdf_layout(pdf_bytes, arxiv_id)
    full_text = _extract_text_best_effort(pdf_bytes, arxiv_id)
    text_equations = (
        _extract_with_pdf_text_layout(full_text, arxiv_id) if full_text else []
    )
    xml_equations = _extract_with_pdftohtml_xml(pdf_bytes, arxiv_id)

    first_sequence_outputs = [
        output
        for output in (pymupdf_equations, text_equations, xml_equations)
        if output and _has_first_equation_prefix(output)
    ]
    if len(first_sequence_outputs) >= 2:
        merged = _merge_pdf_engine_outputs(first_sequence_outputs, arxiv_id)
        logger.info(
            "Found %d numbered equations using merged PDF engines", len(merged)
        )
        return _finalize_pdf_equations(
            pdf_bytes, arxiv_id, merged[:MAX_EQUATIONS], allow_latex_ocr
        )

    if pymupdf_equations and _has_first_equation_prefix(pymupdf_equations):
        logger.info(
            "Found %d numbered equations using PyMuPDF layout",
            len(pymupdf_equations),
        )
        return _finalize_pdf_equations(
            pdf_bytes, arxiv_id, pymupdf_equations[:MAX_EQUATIONS], allow_latex_ocr
        )
    if pymupdf_equations:
        logger.info(
            "arXiv:%s → PyMuPDF equation numbers are not the first sequence (%s); "
            "trying PDF text pass",
            arxiv_id,
            [item.get("eq_number") for item in pymupdf_equations],
        )

    if text_equations and _has_first_equation_prefix(text_equations):
        logger.info(
            "Found %d numbered equations using PDF text layout",
            len(text_equations),
        )
        return _finalize_pdf_equations(
            pdf_bytes, arxiv_id, text_equations[:MAX_EQUATIONS], allow_latex_ocr
        )

    if xml_equations and _has_first_equation_prefix(xml_equations):
        logger.info(
            "Found %d numbered equations using PDF XML geometry",
            len(xml_equations),
        )
        return _finalize_pdf_equations(
            pdf_bytes, arxiv_id, xml_equations[:MAX_EQUATIONS], allow_latex_ocr
        )

    logger.info("Found 0 reliable first-sequence equations in PDF source")
    return []


# ── Engine merge & quality scoring ───────────────────────────────────────────

def _merge_pdf_engine_outputs(engine_outputs, arxiv_id):
    """Choose the cleaner PDF extraction per equation number.

    For each equation number present in any engine output, score all
    candidates using ``_pdf_equation_quality_score`` and keep the best one.
    The audit trail records the competing scores for transparency.
    """
    labels = []
    for output in engine_outputs:
        for item in output:
            label = normalize_equation_label(item.get("eq_number"))
            if label and label not in labels:
                labels.append(label)
    labels = sorted(labels, key=label_sort_key)
    merged = []
    switched = 0

    for eq_number in labels:
        candidates = []
        for output in engine_outputs:
            for item in output:
                if item.get("eq_number") == eq_number:
                    candidates.append(item)
                    break
        if not candidates:
            continue

        scored = [(_pdf_equation_quality_score(item), item) for item in candidates]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        best_score, best_item = scored[0]
        chosen = dict(best_item)
        if len(scored) > 1 and best_item is not candidates[0]:
            switched += 1
        score_text = ", ".join(
            f"{_pdf_engine_name(item)}={score:.2f}" for score, item in scored
        )
        chosen["audit_extract"] = (
            f"{chosen['audit_extract']} "
            f"(selected by general PDF quality comparison: {score_text})"
        )
        merged.append(chosen)

    if switched:
        logger.info(
            "arXiv:%s → selected non-primary PDF engine for %d equation text(s)",
            arxiv_id, switched,
        )
    return merged


def _pdf_engine_name(equation):
    """Return a short name for the engine that produced an equation dict."""
    audit = equation.get("audit_extract", "")
    if "PDF XML geometry" in audit:
        return "xml_geometry"
    if "PDF text" in audit:
        return "text_layout"
    if "PyMuPDF" in audit:
        return "pymupdf"
    return "pdf_engine"


def _pdf_equation_quality_score(equation):
    """Score PDF-extracted equation text without using paper-specific rules.

    Rewards LaTeX commands, ``=`` signs, balanced delimiters, and appropriate
    length.  Penalises prose content, non-printable characters, and known
    incomplete fragments.
    """
    text = equation.get("equation", "") or ""
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return 0.0

    score = 0.35
    if equation.get("equation_confidence") == "medium":
        score += 0.08
    if equation.get("equation_confidence") == "low":
        score -= 0.08
    if "\\" in compact:
        score += 0.16
    if any(token in compact for token in (
        r"\frac", r"\sum", r"\prod", r"\int", r"\hat", r"\langle",
        r"\begin", r"\rho", r"\sigma",
    )):
        score += 0.16
    if "=" in compact:
        score += 0.08
    if _balanced_latex_delimiters(compact):
        score += 0.08
    else:
        score -= 0.14
    if _looks_incomplete_pdf_fragment(compact):
        score -= 0.24
    if any(ord(char) < 32 and char not in "\n\t" for char in compact):
        score -= 0.18
    if re.search(
        r"\b(where|therefore|using|represented by|according to|shown in)\b",
        compact, re.IGNORECASE,
    ):
        score -= 0.16
    words = re.findall(r"[A-Za-z]{4,}", compact)
    if len(words) > 12:
        score -= 0.16
    if len(compact) < 8:
        score -= 0.20
    return max(0.0, min(1.0, score))


def _balanced_latex_delimiters(text):
    """Return True if ``{}``, ``()``, and ``[]`` are balanced in *text*."""
    pairs = (("{", "}"), ("(", ")"), ("[", "]"))
    for left, right in pairs:
        if text.count(left) != text.count(right):
            return False
    return True


# ── Post-extraction passes ────────────────────────────────────────────────────

def _finalize_pdf_equations(pdf_bytes, arxiv_id, equations, allow_latex_ocr):
    """Apply optional OCR upgrade and Docling context enrichment."""
    equations = _apply_latex_ocr_text_if_enabled(
        pdf_bytes, arxiv_id, equations, allow_latex_ocr
    )
    return _enhance_context_with_docling(pdf_bytes, arxiv_id, equations)


def _enhance_context_with_docling(pdf_bytes, arxiv_id, equations):
    """Enrich equation context text using Docling (if available)."""
    enhanced, _ = enhance_pdf_context_with_docling(pdf_bytes, arxiv_id, equations)
    return enhanced


def _apply_latex_ocr_text_if_enabled(pdf_bytes, arxiv_id, equations, allow_latex_ocr):
    """Repair already-found PDF equation text with OCR LaTeX when enabled.

    OCR is deliberately **not** an equation finder here.  The PDF layout /
    text / XML engines must first provide equation labels and canonical order;
    OCR only replaces the broken equation string for matching labels.
    """
    if not allow_latex_ocr or not equations:
        return equations

    ocr_equations = extract_pdf_equations_with_latex_ocr(
        pdf_bytes, arxiv_id, limit=MAX_EQUATIONS,
    )
    if not ocr_equations:
        return equations

    by_number = {item["eq_number"]: item for item in ocr_equations}
    merged = []
    replaced = 0
    for equation in equations:
        ocr_eq = by_number.get(equation.get("eq_number"))
        if not ocr_eq:
            merged.append(equation)
            continue

        item = dict(equation)
        old_preview = item.get("equation", "")[:90]
        new_preview = ocr_eq.get("equation", "")[:90]
        item["equation"] = ocr_eq["equation"]
        item["equation_format"] = "latex_ocr"
        item["equation_confidence"] = "medium"
        item["equation_text_source"] = "latex_ocr"
        item["equation_order_source"] = "pdf"
        if ocr_eq.get("text_before") and not item.get("text_before"):
            item["text_before"] = ocr_eq["text_before"]
        if ocr_eq.get("text_after") and not item.get("text_after"):
            item["text_after"] = ocr_eq["text_after"]
        item["audit_extract"] = (
            f"{new_preview}"
            f"{'...' if len(ocr_eq.get('equation', '')) > 90 else ''} "
            f"(Equation ({item['eq_number']}) text from local Nougat LaTeX OCR; "
            f"PDF sequence retained; previous PDF text was: {old_preview})"
        )
        merged.append(item)
        replaced += 1

    logger.info(
        "arXiv:%s → replaced %d/%d PDF equation text(s) with LaTeX OCR",
        arxiv_id, replaced, len(equations),
    )
    return merged


# ── Sequence helper ───────────────────────────────────────────────────────────

def _has_first_equation_prefix(equations):
    """Delegate to the shared sequence-validation check."""
    return _shared_first_sequence(equations)
