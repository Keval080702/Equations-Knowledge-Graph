"""
Phase 2 — PDF Text Validation & Context Helpers
================================================

Helper functions that analyse raw PDF text lines to:

* Validate whether a string is a plausible equation.
* Detect split / incomplete equation fragments.
* Find section headings and surrounding context text.
* Collect and assemble multi-line equation candidates.
* Extract raw text from a PDF file using pdftotext or pdfminer.

These helpers are shared by all three PDF extraction engines
(``pdf_engines.py``) and by the public orchestrator in ``pdf_adapter.py``.

Pipeline role
-------------
Called during the extraction pass (Phase 2).  All functions operate on
plain-text strings — no PDF object model is exposed here.

Public helpers
--------------
* ``_extract_text_best_effort``       — get raw text from PDF bytes
* ``_is_valid_equation_text``         — reject broken fragments / prose
* ``_is_salvageable_numbered_display``— allow weak but labeled equations
* ``_find_section_heading``           — nearest heading above a line
* ``_valid_equation_label``           — accept "1", "2a", "A.1" labels
* ``_clean_pdf_line``                 — strip PDF artifacts from a line
"""

import io
import logging
import os
import re
import subprocess
import tempfile

from pdfminer.high_level import extract_text

from src.pdf_math import _strip_prose_prefix

logger = logging.getLogger(__name__)


__all__ = [
    "_extract_text_best_effort",
    "_is_valid_equation_text",
    "_is_salvageable_numbered_display",
    "_looks_incomplete_pdf_fragment",
    "_looks_like_partial_equation_row",
    "_find_section_heading",
    "_collect_equation_lines_before",
    "_build_equation_text",
    "_build_marker_window_equation_text",
    "_is_local_formula_row",
    "_has_nearby_layout_math_fragment",
    "_looks_like_formula_fragment",
    "_looks_like_prose_boundary",
    "_is_page_noise",
    "_clean_pdf_line",
    "_valid_equation_label",
]


# ── Raw text extraction ───────────────────────────────────────────────────────

def _extract_text_best_effort(pdf_bytes, arxiv_id):
    """Extract PDF text, preferring poppler layout output over pdfminer.

    Attempts ``pdftotext -layout`` first; falls back to pdfminer if the
    binary is not available or the PDF cannot be processed.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        result = subprocess.run(
            ["pdftotext", "-layout", tmp_path, "-"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except (OSError, subprocess.SubprocessError) as exc:
        logger.info(
            "arXiv:%s → pdftotext unavailable, using pdfminer: %s", arxiv_id, exc
        )
    finally:
        if "tmp_path" in locals():
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    try:
        return extract_text(io.BytesIO(pdf_bytes))
    except Exception as exc:
        logger.error("arXiv:%s → PDF text extraction failed: %s", arxiv_id, exc)
        return ""


# ── Equation label validation ─────────────────────────────────────────────────

def _valid_equation_label(label):
    """Return True for positive integer, dotted, or grouped equation labels."""
    from src.sequence_validation import equation_sequence_label
    label = equation_sequence_label(label)
    parts = label.split(".")
    return bool(parts) and all(
        part.isdigit() and int(part) >= 1 for part in parts
    )


# ── Equation text validation ──────────────────────────────────────────────────

def _is_valid_equation_text(text):
    """Reject broken PDF fragments and prose accidentally tagged as equations.

    Applies structural heuristics: minimum length, math-character density,
    known bad patterns, and prose-word count thresholds.
    """
    candidate = text.strip()
    if len(candidate) < 10:
        return False
    if "cid:" in candidate:
        return False
    if "Eq. (" in candidate or "Eq.(" in candidate:
        return False
    if candidate in {") ))", "1 6", ", 1 2"}:
        return False
    if "|{z}" in candidate or "{z}" in candidate or "{z }" in candidate:
        return False
    if _looks_incomplete_pdf_fragment(candidate):
        return False

    math_chars = set("=+-−×·*/∇∂∫∑∏≈≠≤≥±→←↔[]{}ˆ†√|⟨⟩")
    math_count = sum(1 for char in candidate if char in math_chars)
    alpha_num = sum(1 for char in candidate if char.isalnum())
    if math_count < 1:
        return False
    if alpha_num and math_count / max(alpha_num, 1) < 0.015 and "=" not in candidate:
        return False

    words = re.findall(r"[A-Za-z]{3,}", candidate)
    lowered = candidate.lower()
    prose_cues = (
        "ensures", "phenomenon", "detailed", "nonzero", "first", "preserve",
        "appendix", "deviations", "theorem", "therefore", "where ", "when ",
        "backflow", "perturbative", "state of", "we treat",
    )
    if any(cue in lowered for cue in prose_cues) and len(words) > 4:
        return False
    if len(words) > 14 and candidate[0].isupper():
        return False
    if len(words) > 8 and not candidate.lstrip().startswith(
        ("=", "d", "H", "L", "G", "Z", "⟨", "|", "ρ")
    ):
        return False
    if len(words) > 22:
        return False
    if candidate.lower().startswith(("where ", "which ", "this ", "the ", "we ")):
        return False
    return True


def _is_salvageable_numbered_display(text):
    """Allow weak but numbered display equations to remain in the output.

    The project requires the first numbered equations.  For PDF-only papers,
    dropping weak middle equations makes the output sequence incorrect, so we
    keep equation-looking fragments with a low-confidence audit note.
    """
    candidate = text.strip()
    if len(candidate) < 8:
        return False
    if any(cue in candidate.lower() for cue in (
        "where ", "therefore", "shown here", "competing interests",
        "distribution statement", "preprint",
    )):
        return False
    if "cid:" in candidate:
        return False
    math_chars = set(
        "=+-−×·*/∇∂∫∑∏≈≠≤≥±→←↔[]{}ˆ†√|⟨⟩⊗γρϕΨψπτλµμσθ𝑑𝑡𝑣𝐻𝐶𝑅𝐾"
    )
    math_count = sum(1 for char in candidate if char in math_chars)
    has_math_letter = bool(re.search(r"[𝑎-𝑧𝐴-𝑍]", candidate))
    return math_count >= 1 or has_math_letter


def _looks_incomplete_pdf_fragment(candidate):
    """Detect common partial-equation artifacts produced by PDF text layout."""
    compact = re.sub(r"\s+", " ", candidate).strip()
    lowered = compact.lower()
    if compact.startswith("="):
        return True
    if _looks_like_partial_equation_row(compact):
        return True
    if compact.startswith("("):
        return True
    if re.match(r"^\(\s*\+\s*\)", compact) or "( + )" in compact:
        return True
    if "Z t Z" in compact or "Z τ" in compact or "Z 2π" == compact:
        return True
    if any(0xF800 <= ord(char) <= 0xF8FF for char in compact):
        return True
    if "𝑔𝑠𝑖n" in compact or "𝑐𝑜𝑠" in compact or "sin ((e" in compact:
        return True
    if "= ," in compact or "√ ," in compact:
        return True
    if (
        compact.startswith("(k)")
        or compact.startswith(", H")
        or compact.startswith("H (")
    ):
        return True
    if lowered.startswith(("following identity", "using this identity")):
        return False
    if "=" not in compact and not re.search(r"[⟨|].*[⟩|]", compact):
        return True
    if "=" in compact:
        left = compact.split("=", 1)[0].strip()
        if len(left) < 2 or left in {"(", "( + )", "+"}:
            return True
    return False


def _looks_like_partial_equation_row(text):
    """Detect denominator/middle rows from stacked PDF equations."""
    compact = re.sub(r"\s+", " ", str(text)).strip()
    if not compact:
        return False
    lowered = compact.lower()
    if re.match(r"^(?:\\partial|∂)\s*t\s*=", compact):
        return True
    if re.match(r"^(?:d\s*t|dt|d𝑡|𝑑𝑡)\s*\d*\s*=", compact):
        return True
    if re.match(r"^(?:\\rangle|⟩)\s*=", compact):
        return True
    if re.match(r"^[A-Za-z0-9_{}\\]+\s*/\s*[A-Za-z0-9_{}\\]+\s*=", compact):
        return True
    if lowered in {"dt", "dt 2", "d t", "∂t", "\\partial t"}:
        return True
    left = compact.split("=", 1)[0].strip() if "=" in compact else compact
    if "=" in compact and len(left) <= 12 and any(
        token in left for token in ("dt", "∂t", "\\partial")
    ):
        return True
    return False


# ── Context and section heading helpers ──────────────────────────────────────

def _find_section_heading(lines, current_line):
    """Walk backward from ``current_line`` to find the nearest section heading."""
    for j in range(current_line - 1, max(0, current_line - 50), -1):
        line = lines[j].strip()
        if not line:
            continue
        # Roman-numeral section: "II. Theory" or "III Theory"
        if re.match(r"^[IVX]+\.?\s+[A-Z]", line) and len(line) < 80:
            return line
        # Numbered section: "2.1 Methods"
        if re.match(r"^\d+\.?\d*\s+[A-Z]", line) and len(line) < 80:
            return line
        # All-caps heading: "INTRODUCTION"
        if line.isupper() and 3 < len(line) < 60:
            return line
    return "Unknown Section"


def _collect_equation_lines_before(lines, marker_line):
    """Collect nearby PDF text lines that probably belong to the equation."""
    collected = []
    for j in range(marker_line - 1, max(-1, marker_line - 45), -1):
        line = lines[j].strip()
        if not line:
            continue
        if _is_page_noise(line):
            continue
        if re.search(r"\(\d{1,3}\)\s*$", line):
            break
        if _looks_like_prose_boundary(line, collected):
            break
        if len(line) <= 180:
            collected.insert(0, line)
    return collected


def _build_equation_text(context_before, marker_text):
    """Join context lines and marker text into one PDF equation candidate."""
    marker_text = _strip_prose_prefix(marker_text.strip())
    if _is_valid_equation_text(marker_text) and len(marker_text) > 35:
        parts = [marker_text]
    else:
        parts = list(context_before)
        if len(parts) > 8:
            parts = parts[-8:]
        if marker_text:
            parts.append(marker_text)
    cleaned = [
        _strip_prose_prefix(_clean_pdf_line(line))
        for line in parts
        if _clean_pdf_line(line)
    ]
    return _strip_prose_prefix(" ".join(cleaned).strip())


def _build_marker_window_equation_text(lines, marker_line, marker_text):
    """Build an equation from the local formula rows around a numbered marker."""
    parts = []
    previous_parts = []
    blank_count = 0

    for offset in range(-1, -9, -1):
        idx = marker_line + offset
        if idx < 0:
            continue
        line = lines[idx].strip()
        if not line:
            blank_count += 1
            if previous_parts and blank_count <= 2:
                continue
            if blank_count > 2:
                break
            continue
        if _is_local_formula_row(line):
            blank_count = 0
            previous_parts.insert(0, line)
        else:
            break
    parts.extend(previous_parts)

    if marker_text.strip():
        parts.append(marker_text.strip())

    blank_count = 0
    for offset in range(1, 9):
        idx = marker_line + offset
        if idx >= len(lines):
            continue
        line = lines[idx].strip()
        if not line:
            blank_count += 1
            if parts and blank_count <= 2:
                continue
            if blank_count > 2:
                break
            continue
        if _is_local_formula_row(line):
            blank_count = 0
            parts.append(line)
        elif parts:
            break

    cleaned = [_clean_pdf_line(line) for line in parts if _clean_pdf_line(line)]
    if len(cleaned) >= 2:
        return _strip_prose_prefix("\n".join(cleaned).strip())
    return _strip_prose_prefix(" ".join(cleaned).strip())


# ── Formula-row classification ────────────────────────────────────────────────

def _is_local_formula_row(line):
    """Return True for short nearby rows that are part of a displayed formula."""
    compact = re.sub(r"\s+", " ", line).strip()
    if not compact or _is_page_noise(compact):
        return False
    if re.search(r"\(\d{1,3}\)\s*$", compact):
        return False
    if compact.lower().startswith(
        ("where ", "with ", "so,", "for ", "in ", "all ", "using ")
    ):
        return False
    if re.match(r"^\d+(?:\.\d+)*\s+[A-Z]", compact):
        return False

    words = re.findall(r"[A-Za-z]{3,}", compact)
    if len(words) > 4:
        return False

    math_chars = set(
        "=+-−×·*/∇∂∫∑∏≈≠≤≥±→←↔[]{}()ˆ†√|⟨⟩⊗γρϕΨψπτλµμσθ"
        "𝑑𝑡𝑣𝐻𝐶𝑅𝐾𝜌𝜎𝜔𝛽"
    )
    math_count = sum(1 for char in compact if char in math_chars)
    has_math_letter = bool(re.search(r"[𝑎-𝑧𝐴-𝑍𝜌𝜎𝜔𝛽𝛼]", compact))
    has_matrix_numbers = (
        bool(re.search(r"[01]\s+[01]", compact))
        or "−𝑖" in compact
        or "𝑖" in compact
    )
    return math_count >= 1 or has_math_letter or has_matrix_numbers


def _has_nearby_layout_math_fragment(lines, marker_line):
    """Detect formulas split across PDF layout rows.

    PDF text often places fractions, summation limits, matrix brackets, and
    transposes on neighbouring rows.  If that happens, the marker row alone is
    a partial equation and should not be accepted as a clean extraction.
    """
    marker = lines[marker_line].strip()
    for offset in (-4, -3, -2, -1, 1, 2):
        idx = marker_line + offset
        if idx < 0 or idx >= len(lines):
            continue
        line = lines[idx].strip()
        if not line or _is_page_noise(line):
            continue
        if re.search(r"\(\d{1,3}\)\s*$", line):
            continue
        if _looks_like_formula_fragment(line, marker):
            return True
    return False


def _looks_like_formula_fragment(line, marker):
    """Return True for neighbouring rows that look like lost equation pieces."""
    compact = re.sub(r"\s+", " ", line).strip()
    if not compact:
        return False

    words = re.findall(r"[A-Za-z]{3,}", compact)
    math_chars = set("=+-−×·*/∇∂∫∑∏≈≠≤≥±→←↔[]{}ˆ†√|⟨⟩⊗γρϕΨπτλµ")
    math_count = sum(1 for char in compact if char in math_chars)

    if "|{z}" in compact or "{z}" in compact:
        return True
    if compact in {"!", "†", "T", "∗", "0", "1"}:
        return True
    if re.fullmatch(r"[0-9π∫∑∏Zτϕπ /+-]+", compact) and any(
        token in compact for token in ("∫", "Z", "π", "/", "+", "-")
    ):
        return True
    if any(char in compact for char in ""):
        return True
    if math_count >= 2 and len(words) <= 3 and len(compact) < 80:
        return True

    marker_has_math = any(
        char in marker for char in "=+-−×·*/∇∂∫∑∏⊗"
    )
    if marker_has_math and math_count >= 1 and len(words) <= 4 and len(compact) < 120:
        return True
    return False


def _looks_like_prose_boundary(line, collected):
    """Return True if a backward scan has reached surrounding prose."""
    if not collected:
        return False
    words = re.findall(r"[A-Za-z]{3,}", line)
    has_math = any(char in line for char in "=+-−×·*/∇∂∫∑[]{}ˆ†√")
    if len(words) > 10 and not has_math:
        return True
    if line.endswith(".") and len(words) > 6 and not has_math:
        return True
    return False


# ── Page noise & line cleaning ────────────────────────────────────────────────

def _is_page_noise(line):
    """Filter headers, footers, page numbers, and preprint boilerplate."""
    if re.fullmatch(r"\d+", line):
        return True
    lowered = line.lower()
    return lowered in {"preprint"} or "distribution statement" in lowered


def _clean_pdf_line(line):
    """Remove obvious PDF artifacts while preserving math tokens."""
    cleaned = re.sub(r"\s+", " ", line).strip()
    cleaned = cleaned.replace("\x0c", " ")
    return cleaned
