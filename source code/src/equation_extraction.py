"""
Phase 2 — Equation Extraction Facade
=====================================

Provides a single public boundary for the equation-extraction stage.

The project has two allowed equation sources, tried in order:

1. **arXiv ar5iv HTML** (``/html/{id}``) — structured MathML equations,
   high confidence.  Handled by :mod:`src.html_adapter`.
2. **arXiv PDF** (``/pdf/{id}``) — fallback when HTML is unavailable.
   Handled by :mod:`src.pdf_adapter` which delegates to three extraction
   engines in :mod:`src.pdf_engines`.

Usage
-----
::

    from src.equation_extraction import parse_html, parse_pdf, clean_equation_text

Pipeline role
-------------
Called by ``tests/test_10_papers.py`` after source acquisition (Phase 1).
"""

from src.html_adapter import parse_html
from src.pdf_adapter import parse_pdf
from src.symbol_extraction import (
    _fix_missing_subscript_underscore,
    _strip_raw_prose_head,
)


def clean_equation_text(text):
    """Remove prose contamination and repair common OCR artefacts in equation text.

    Applies two pattern-based cleaners that mirror what symbol extraction already
    does internally, so the stored ``equation`` field is as clean as the symbols:

    * :func:`_strip_raw_prose_head` – strips leading caption/sentence prose that
      PDF extraction sometimes prepends to the actual LaTeX (e.g. ``SNAIL
      non-linear parameters g_{3} , g_{4} . In Fig. 9 we show a \\rho...``).
    * :func:`_fix_missing_subscript_underscore` – repairs the PDF-OCR dropout of
      ``_`` in subscripts (e.g. ``\\hat{H}0`` → ``\\hat{H}_{0}``).
    """
    text = _fix_missing_subscript_underscore(text or "")
    text = _strip_raw_prose_head(text)
    return text


def extract_equations(source_format, content, arxiv_id, allow_latex_ocr=False):
    """Dispatch equation extraction by source format."""
    if source_format == "html":
        return parse_html(content, arxiv_id)
    if source_format == "pdf":
        return parse_pdf(content, arxiv_id, allow_latex_ocr=allow_latex_ocr)
    raise ValueError(f"Unsupported equation source format: {source_format}")


__all__ = [
    "clean_equation_text",
    "extract_equations",
    "parse_html",
    "parse_pdf",
]
