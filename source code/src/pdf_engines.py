"""
Phase 2 — PDF Extraction Engines
=================================

Three independent engines that locate and extract numbered equations from a
PDF file.  Each engine is a self-contained extraction strategy; the
orchestrator in ``pdf_adapter.py`` decides which output(s) to keep.

Engines
-------
* **PyMuPDF layout engine** (``_extract_with_pymupdf_layout``)
  Uses block-level coordinates from PyMuPDF (``fitz``) to locate equation
  markers on the right margin and collect surrounding math blocks.

* **pdftohtml XML engine** (``_extract_with_pdftohtml_xml``)
  Runs Poppler's ``pdftohtml -xml`` to get word-level geometry, then uses
  column position to identify equation numbers and align token rows.

* **Text layout engine** (``_extract_with_pdf_text_layout``)
  Operates on plain text from ``pdftotext -layout`` or pdfminer, scanning
  for ``(N)`` markers at end-of-line and collecting context lines.

All engines call the shared utilities in ``pdf_math.py`` and
``pdf_text_helpers.py`` for normalization and validation.

Pipeline role
-------------
Called exclusively by ``parse_pdf`` in ``pdf_adapter.py``.
"""

import logging
import os
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from src.pdf_math import (
    _normalize_pdf_math,
    _postprocess_pdf_equation,
    _pdf_text_to_latex_like,
    _strip_prose_prefix,
)
from src.pdf_text_helpers import (
    _clean_pdf_line,
    _collect_equation_lines_before,
    _build_equation_text,
    _build_marker_window_equation_text,
    _find_section_heading,
    _is_local_formula_row,
    _is_valid_equation_text,
    _is_salvageable_numbered_display,
    _looks_like_partial_equation_row,
    _has_nearby_layout_math_fragment,
    _valid_equation_label,
)
from src.sequence_validation import normalize_equation_label, equation_sequence_label

logger = logging.getLogger(__name__)

MAX_EQUATIONS = 7


__all__ = [
    "_extract_with_pymupdf_layout",
    "_extract_with_pdftohtml_xml",
    "_extract_with_pdf_text_layout",
    "_looks_like_math_text",
]


# ── Engine 1: PyMuPDF layout-based extraction ─────────────────────────────────

def _extract_with_pymupdf_layout(pdf_bytes, arxiv_id):
    """Extract equations using PyMuPDF block coordinates.

    Uses positioned text blocks to find right-margin equation markers (``(N)``)
    and gathers surrounding math content from the same page column.
    Falls back to an empty list if PyMuPDF is not installed or the PDF
    cannot be opened.
    """
    try:
        import fitz
    except ImportError:
        logger.info(
            "arXiv:%s → PyMuPDF unavailable, using text fallback", arxiv_id
        )
        return []

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        logger.info(
            "arXiv:%s → PyMuPDF open failed, using text fallback: %s",
            arxiv_id, exc,
        )
        return []

    equations = []
    seen = set()
    for page_index, page in enumerate(doc):
        blocks = _pymupdf_text_blocks(page)
        markers = _pymupdf_equation_markers(blocks, page.rect.width)
        page_lines = page.get_text("text").splitlines()

        for marker in markers:
            eq_num = marker["eq_num"]
            if eq_num in seen:
                continue

            raw_text = _build_pymupdf_equation(blocks, marker, page.rect)
            equation_text = _pdf_text_to_latex_like(
                _postprocess_pdf_equation(_normalize_pdf_math(raw_text))
            )
            if not _is_valid_equation_text(equation_text):
                if not _is_salvageable_numbered_display(equation_text):
                    logger.info(
                        "arXiv:%s → rejected PyMuPDF Eq(%s): %r",
                        arxiv_id, eq_num, equation_text[:80],
                    )
                    continue
                logger.info(
                    "arXiv:%s → kept low-confidence PyMuPDF Eq(%s): %r",
                    arxiv_id, eq_num, equation_text[:80],
                )
                low_confidence = True
            else:
                low_confidence = False

            section = _find_section_heading(page_lines, len(page_lines))
            text_before, text_after = _pymupdf_context(page, marker)
            preview = (
                equation_text[:80] + "..."
                if len(equation_text) > 80
                else equation_text
            )
            quality_note = "low-confidence/incomplete " if low_confidence else ""
            equations.append({
                "eq_number": eq_num,
                "equation": equation_text,
                "equation_format": "latex_like_from_pdf",
                "equation_confidence": "low" if low_confidence else "medium",
                "text_before": text_before,
                "text_after": text_after,
                "section": section,
                "audit_extract": (
                    f"{preview} (Equation ({eq_num}) from "
                    f"{quality_note}PyMuPDF layout, page {page_index + 1})"
                ),
                "audit_context": (
                    f"{len(text_before)} chars before, "
                    f"{len(text_after)} chars after "
                    f"(Section: '{section}')"
                ),
            })
            seen.add(eq_num)

            if len(equations) >= MAX_EQUATIONS:
                return equations[:MAX_EQUATIONS]

    return equations


def _pymupdf_text_blocks(page):
    """Return normalised text blocks with bounding-box and centre coordinates."""
    blocks = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        pieces = []
        for line in block.get("lines", []):
            line_text = "".join(
                span.get("text", "") for span in line.get("spans", [])
            )
            if line_text.strip():
                pieces.append(line_text.strip())
        text = " ".join(pieces).strip()
        if not text:
            continue
        x0, y0, x1, y1 = block["bbox"]
        blocks.append({
            "text": _clean_pdf_line(text),
            "bbox": (x0, y0, x1, y1),
            "cx": (x0 + x1) / 2,
            "cy": (y0 + y1) / 2,
            "lines": block.get("lines", []),
        })
    return blocks


def _pymupdf_equation_markers(blocks, page_width):
    """Find right-margin equation markers from positioned blocks."""
    markers = []
    for block in blocks:
        matches = list(re.finditer(
            r"\((\d{1,3}(?:\.\d{1,3})*(?:[A-Za-z](?:\s*[-–—−]\s*[A-Za-z])?)?)\)",
            block["text"],
        ))
        if not matches:
            continue

        x0, _, x1, _ = block["bbox"]
        marker_is_right = (
            x1 > page_width * 0.83
            or page_width * 0.45 <= x1 <= page_width * 0.55
        )
        if not marker_is_right:
            continue

        previous_end = 0
        for match in matches:
            eq_num = normalize_equation_label(match.group(1))
            if not _valid_equation_label(eq_num):
                previous_end = match.end()
                continue

            text_without_marker = block["text"][previous_end:match.start()].strip()
            next_start = (
                matches[matches.index(match) + 1].start()
                if matches.index(match) + 1 < len(matches)
                else len(block["text"])
            )
            text_after_marker = block["text"][match.end():next_start].strip()
            marker_at_segment_end = not text_after_marker
            if not marker_at_segment_end and not _is_local_formula_row(text_after_marker):
                previous_end = match.end()
                continue
            if not _is_plausible_marker_prefix(text_without_marker):
                previous_end = match.end()
                continue

            marker_is_standalone = len(text_without_marker) <= 8
            marker_in_math_block = _looks_like_math_text(text_without_marker)
            if not (marker_is_standalone or marker_in_math_block):
                previous_end = match.end()
                continue

            marker = dict(block)
            marker["eq_num"] = eq_num
            marker["prefix"] = text_without_marker
            markers.append(marker)
            previous_end = match.end()
    return markers


def _is_plausible_marker_prefix(text):
    """Reject prose parentheticals like SU(2) while keeping equation prefixes."""
    prefix = text.strip()
    if not prefix:
        return True
    words = re.findall(r"[A-Za-z]{3,}", prefix)
    has_equation_operator = any(
        token in prefix for token in ("=", "±", "−", "+", "†", "ˆ", "[", "]")
    )
    if len(words) > 5 and "=" not in prefix:
        return False
    return (
        has_equation_operator
        or len(prefix) <= 12
        or _looks_like_math_text(prefix)
    )


def _build_pymupdf_equation(blocks, marker, page_rect):
    """Collect and order nearby coordinate blocks for one displayed equation."""
    marker_text = marker["text"]
    marker_match = re.search(r"\(\d{1,3}\)\s*$", marker_text)
    if (
        _looks_like_math_text(marker.get("prefix", ""))
        and not _looks_like_partial_equation_row(marker.get("prefix", ""))
    ):
        return _postprocess_pdf_equation(_strip_prose_prefix(marker["prefix"]))

    marker_cy = marker["cy"]
    column_left, column_right = _pymupdf_column_bounds(blocks, marker, page_rect)
    _, marker_y0, _, marker_y1 = marker["bbox"]
    marker_height = max(marker_y1 - marker_y0, 12)
    window = (
        62
        if page_rect.width * 0.70 < marker["bbox"][2] < page_rect.width * 0.88
        else 38
    )

    candidates = []
    for block in blocks:
        x0, y0, x1, y1 = block["bbox"]
        if x1 < column_left or x0 > column_right:
            continue
        if abs(block["cy"] - marker_cy) > window:
            continue
        if x0 > page_rect.width * 0.93:
            continue
        cleaned = block["text"]
        if block is marker and marker_match:
            cleaned = cleaned[:marker_match.start()].strip()
        else:
            cleaned = re.sub(r"\(\d{1,3}\)\s*$", "", cleaned).strip()
        if not cleaned:
            continue
        if _looks_like_math_text(cleaned):
            candidates.append({
                "text": cleaned,
                "bbox": (x0, y0, x1, y1),
                "cy": block["cy"],
                "raw": block,
            })

    if not candidates and marker.get("prefix"):
        return marker["prefix"]

    candidates = _filter_pymupdf_candidates(candidates, marker_cy)
    rows = _cluster_rows(candidates)
    row_texts = []
    for row in rows:
        ordered = sorted(row, key=lambda item: (item["bbox"][0], item["bbox"][1]))
        row_text = " ".join(
            _linearize_structured_math_block(item) for item in ordered
        )
        row_texts.append(row_text)
    equation = " ".join(row_texts)
    return _postprocess_pdf_equation(_strip_prose_prefix(equation.strip()))


def _pymupdf_column_bounds(blocks, marker, page_rect):
    """Infer the column containing an equation marker."""
    marker_x0, _, marker_x1, _ = marker["bbox"]
    center = page_rect.width / 2
    if marker_x1 <= center + 25:
        return 0, center
    same_band_math = [
        block for block in blocks
        if abs(block["cy"] - marker["cy"]) < 28
        and _looks_like_math_text(block["text"])
        and len(re.findall(r"[A-Za-z]{3,}", block["text"])) <= 6
    ]
    if same_band_math:
        band_left = min(block["bbox"][0] for block in same_band_math)
        band_right = max(block["bbox"][2] for block in same_band_math)
        if band_left < center * 0.6 and band_right > page_rect.width * 0.82:
            return 0, page_rect.width
    if marker_x0 >= center - 25:
        return center, page_rect.width
    return 0, page_rect.width


def _linearize_structured_math_block(item):
    """Preserve small matrix-like blocks using their line structure."""
    raw = item.get("raw") or {}
    lines = raw.get("lines", [])
    line_texts = [
        "".join(span.get("text", "") for span in line.get("spans", [])).strip()
        for line in lines
    ]
    line_texts = [text for text in line_texts if text]
    if len(line_texts) >= 2 and _looks_like_matrix_lines(line_texts):
        return _matrix_lines_to_text(line_texts)
    return item["text"]


def _looks_like_matrix_lines(line_texts):
    """Detect compact matrix/vector text split into top/bottom rows."""
    joined = " ".join(line_texts)
    if "ρ" not in joined and "\\rho" not in joined:
        return False
    if len(line_texts) != 4:
        return False
    return all(re.search(r"(?:ρ|rho)\d", text) for text in line_texts)


def _matrix_lines_to_text(line_texts):
    """Represent a 2×2 matrix-like PDF block as a readable matrix string."""
    top = " ".join(line_texts[:2]).strip()
    bottom = " ".join(line_texts[2:]).strip()
    if top.startswith("(") and bottom.endswith(") ="):
        top = top[1:].strip()
        bottom = bottom[:-2].strip()
        return f"(({top}); ({bottom})) ="
    if top.startswith("(") and bottom.endswith(") +"):
        top = top[1:].strip()
        bottom = bottom[:-2].strip()
        return f"(({top}); ({bottom})) +"
    if bottom.endswith(") +"):
        bottom = bottom[:-2].strip()
        return f"(({top}); ({bottom})) +"
    if bottom.endswith(")"):
        bottom = bottom[:-1].strip()
        return f"(({top}); ({bottom}))"
    return f"(({top}); ({bottom}))"


def _filter_pymupdf_candidates(candidates, marker_cy):
    """Drop prose and page annotations from a PyMuPDF equation window."""
    filtered = []
    for item in candidates:
        text = item["text"]
        words = re.findall(r"[A-Za-z]{3,}", text)
        if len(words) > 7 and not any(
            char in text for char in "=∑∫√†⟨⟩⊗ρψΨσγθ𝑑𝑣𝐻"
        ):
            continue
        if text.lower().startswith(("where ", "the ", "this ", "we ")):
            continue
        if "|{z}" in text or "{z}" in text:
            continue
        filtered.append(item)
    return filtered


def _cluster_rows(items):
    """Cluster positioned blocks into visual rows by vertical centre proximity."""
    rows = []
    for item in sorted(items, key=lambda value: value["cy"]):
        for row in rows:
            center = sum(member["cy"] for member in row) / len(row)
            if abs(item["cy"] - center) <= 7:
                row.append(item)
                break
        else:
            rows.append([item])
    return rows


def _looks_like_math_text(text):
    """Return True when a text block has equation-like content."""
    candidate = text.strip()
    if not candidate:
        return False
    math_chars = set(
        "=+-−×·*/∇∂∫∑∏≈≠≤≥±→←↔[]{}ˆ†√|⟨⟩⊗γρϕΨψπτλµμσθ𝑑𝑡𝑣𝐻𝐶𝑅𝐾"
    )
    if sum(1 for char in candidate if char in math_chars) >= 1:
        return True
    if re.search(r"[A-Za-z]\s*[_^]?\s*\d", candidate):
        return True
    if re.search(r"[𝑎-𝑧𝐴-𝑍]", candidate):
        return True
    return False


def _pymupdf_context(page, marker):
    """Build rough before/after text using vertical page position."""
    before = []
    after = []
    for block in _pymupdf_text_blocks(page):
        text = re.sub(r"\s+", " ", block["text"]).strip()
        if not text:
            continue
        if block["cy"] < marker["cy"] - 35 and len(before) < 8:
            before.append(text)
        elif block["cy"] > marker["cy"] + 35 and len(after) < 8:
            after.append(text)
    return " ".join(before[-8:]), " ".join(after[:8])


# ── Engine 2: pdftohtml XML geometry extraction ───────────────────────────────

def _extract_with_pdftohtml_xml(pdf_bytes, arxiv_id):
    """Extract numbered equations from Poppler XML token geometry.

    Runs ``pdftohtml -xml`` on the PDF and uses word-level coordinates to
    identify right-margin equation numbers and collect aligned formula rows.
    Handles stacked fractions that would otherwise be split across PDF text
    lines.
    """
    xml_text = _pdftohtml_xml_text(pdf_bytes, arxiv_id)
    if not xml_text:
        return []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.info("arXiv:%s → pdftohtml XML parse failed: %s", arxiv_id, exc)
        return []

    equations = []
    seen = set()
    for page in root.findall("page"):
        try:
            page_number = int(page.attrib.get("number", "0"))
            page_width = float(page.attrib.get("width", "0") or 0)
        except ValueError:
            continue

        tokens = _xml_page_tokens(page)
        markers = _xml_equation_markers(tokens, page_width)
        page_lines = _xml_page_lines(tokens)
        for marker in markers:
            eq_num = marker["eq_num"]
            if eq_num in seen:
                continue
            raw_text = _build_xml_geometry_equation(tokens, marker, page_width)
            if not raw_text:
                continue
            equation_text = _pdf_text_to_latex_like(
                _postprocess_pdf_equation(_normalize_pdf_math(raw_text))
            )
            if not _is_valid_equation_text(equation_text):
                if not _is_salvageable_numbered_display(equation_text):
                    continue
                low_confidence = True
            else:
                low_confidence = "\\begin{aligned}" in equation_text

            text_before, text_after = _xml_context(tokens, marker)
            section = _find_section_heading(page_lines, len(page_lines))
            preview = (
                equation_text[:80] + "..."
                if len(equation_text) > 80
                else equation_text
            )
            equations.append({
                "eq_number": eq_num,
                "equation": equation_text,
                "equation_format": "latex_like_from_pdf",
                "equation_confidence": "low" if low_confidence else "medium",
                "text_before": text_before,
                "text_after": text_after,
                "section": section,
                "audit_extract": (
                    f"{preview} (Equation ({eq_num}) from "
                    f"{'low-confidence/stacked ' if low_confidence else ''}"
                    f"PDF XML geometry, page {page_number})"
                ),
                "audit_context": (
                    f"{len(text_before)} chars before, "
                    f"{len(text_after)} chars after "
                    f"(Section: '{section}')"
                ),
            })
            seen.add(eq_num)
            if len(equations) >= MAX_EQUATIONS:
                return equations
    return equations


def _pdftohtml_xml_text(pdf_bytes, arxiv_id):
    """Run pdftohtml and return the raw XML output as a string."""
    tmp_dir = ""
    try:
        tmp_dir = tempfile.mkdtemp(prefix="pdfxml.")
        pdf_path = os.path.join(tmp_dir, "paper.pdf")
        out_prefix = os.path.join(tmp_dir, "out")
        with open(pdf_path, "wb") as handle:
            handle.write(pdf_bytes)
        result = subprocess.run(
            ["pdftohtml", "-xml", "-i", "-noframes", pdf_path, out_prefix],
            capture_output=True,
            text=True,
            timeout=60,
        )
        xml_path = out_prefix + ".xml"
        if result.returncode == 0 and os.path.exists(xml_path):
            with open(xml_path, "r", encoding="utf-8", errors="ignore") as handle:
                return handle.read()
    except (OSError, subprocess.SubprocessError) as exc:
        logger.info(
            "arXiv:%s → pdftohtml XML unavailable: %s", arxiv_id, exc
        )
    finally:
        if tmp_dir:
            for root_dir, _dirs, files in os.walk(tmp_dir, topdown=False):
                for filename in files:
                    try:
                        os.unlink(os.path.join(root_dir, filename))
                    except OSError:
                        pass
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass
    return ""


def _xml_page_tokens(page):
    """Parse XML ``<text>`` elements into dicts with position and cleaned text."""
    tokens = []
    for element in page.findall("text"):
        text = "".join(element.itertext()).strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            continue
        try:
            left = float(element.attrib.get("left", "0"))
            top = float(element.attrib.get("top", "0"))
            width = float(element.attrib.get("width", "0"))
            height = float(element.attrib.get("height", "0"))
        except ValueError:
            continue
        tokens.append({
            "text": _clean_pdf_line(text),
            "left": left,
            "top": top,
            "right": left + width,
            "bottom": top + height,
            "cx": left + width / 2,
            "cy": top + height / 2,
        })
    return tokens


def _xml_equation_markers(tokens, page_width):
    """Identify right-margin equation-number tokens from pdftohtml XML."""
    markers = []
    for token in tokens:
        match = re.fullmatch(
            r"\((\d{1,3}(?:\.\d{1,3})*"
            r"(?:[A-Za-z](?:\s*[-–—−]\s*[A-Za-z])?)?)\)",
            token["text"],
        )
        if not match:
            continue
        if token["left"] < page_width * 0.45:
            continue
        eq_num = normalize_equation_label(match.group(1))
        if _valid_equation_label(eq_num):
            marker = dict(token)
            marker["eq_num"] = eq_num
            markers.append(marker)
    return markers


def _build_xml_geometry_equation(tokens, marker, page_width):
    """Collect and join token rows that constitute one XML-geometry equation."""
    marker_cy = marker["cy"]
    column_left, column_right = _xml_column_bounds(marker, page_width)
    row_window = 34
    nearby = []
    for token in tokens:
        if token is marker or token["text"] == marker["text"]:
            continue
        if (
            token["right"] < column_left
            or token["left"] > min(column_right, marker["left"] - 4)
        ):
            continue
        if abs(token["cy"] - marker_cy) > row_window:
            continue
        if not _looks_like_math_text(token["text"]) and not _is_short_xml_math_token(
            token["text"]
        ):
            continue
        nearby.append(token)

    if not nearby:
        return ""
    rows = _cluster_xml_rows(nearby)
    row_texts = []
    for row in rows:
        ordered = sorted(row, key=lambda item: item["left"])
        text = " ".join(item["text"] for item in ordered)
        text = re.sub(r"\s+", " ", text).strip()
        if text and _xml_row_is_equation_like(text):
            row_texts.append(text)
    return "\n".join(row_texts)


def _is_short_xml_math_token(text):
    """Allow short identifiers/operators once geometry has selected equation rows."""
    compact = re.sub(r"\s+", "", str(text))
    if not compact:
        return False
    if compact in {",", ".", "!", ":", ";"}:
        return True
    if re.fullmatch(r"[A-Za-z0-9µμ]+", compact) and len(compact) <= 4:
        return True
    if re.fullmatch(r"[()[\]{}|+\-=*/<>]+", compact):
        return True
    return False


def _xml_row_is_equation_like(text):
    """Drop prose rows accidentally inside a coordinate equation window."""
    compact = re.sub(r"\s+", " ", str(text)).strip()
    if not compact:
        return False
    lowered = compact.lower()
    if lowered.startswith(
        ("where ", "the ", "this ", "that ", "equation ")
    ):
        return False
    if re.search(r"[=+\-−*/∑∫√†⟨⟩{}()[\]|]", compact):
        return True
    if any(char in compact for char in "ρσψΨθλµμγτ"):
        return True
    if re.findall(r"[A-Za-z]{5,}", compact):
        return False
    return bool(re.search(r"[A-Za-z0-9]", compact))


def _xml_column_bounds(marker, page_width):
    """Return left/right column boundaries for a page-half column."""
    center = page_width / 2
    if marker["left"] < center:
        return 0, center
    return center, page_width


def _cluster_xml_rows(tokens):
    """Cluster XML tokens into rows by vertical centre proximity."""
    rows = []
    for token in sorted(tokens, key=lambda item: item["cy"]):
        for row in rows:
            center = sum(item["cy"] for item in row) / len(row)
            if abs(token["cy"] - center) <= 5:
                row.append(token)
                break
        else:
            rows.append([token])
    return rows


def _xml_page_lines(tokens):
    """Convert token rows into plain-text lines for section-heading detection."""
    rows = _cluster_xml_rows(tokens)
    lines = []
    for row in rows:
        ordered = sorted(row, key=lambda item: item["left"])
        lines.append(" ".join(item["text"] for item in ordered))
    return lines


def _xml_context(tokens, marker):
    """Build before/after context strings from token rows around a marker."""
    before = []
    after = []
    for row in _cluster_xml_rows(tokens):
        row_center = sum(item["cy"] for item in row) / len(row)
        text = " ".join(
            item["text"]
            for item in sorted(row, key=lambda item: item["left"])
        )
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        if row_center < marker["cy"] - 35 and len(before) < 8:
            before.append(text)
        elif row_center > marker["cy"] + 35 and len(after) < 8:
            after.append(text)
    return " ".join(before[-8:]), " ".join(after[:8])


# ── Engine 3: plain-text layout extraction ───────────────────────────────────

def _extract_with_pdf_text_layout(full_text, arxiv_id):
    """Extract numbered equations from pdftotext/pdfminer layout text.

    Performs a two-pass scan:

    * **Pass 1** — find every ``(N)`` marker at end-of-line.
    * **Pass 2** — validate and select equations in document order, up to the
      ``MAX_EQUATIONS`` limit.
    """
    lines = full_text.split("\n")

    # ── Pass 1: Find ALL candidate (N) markers ────────────────────────────
    candidates = []
    for i, line in enumerate(lines):
        match = re.search(
            r"\((\d{1,3}(?:\.\d{1,3})*"
            r"(?:[A-Za-z](?:\s*[-–—−]\s*[A-Za-z])?)?)\)\s*$",
            line.rstrip(),
        )
        if not match:
            continue

        eq_num = normalize_equation_label(match.group(1))
        if not _valid_equation_label(eq_num):
            continue
        eq_text = line[:match.start()].strip()
        context_before = _collect_equation_lines_before(lines, i)
        candidates.append({
            "eq_num": eq_num,
            "eq_text": eq_text,
            "context_before": context_before,
            "line_index": i,
        })

    # ── Pass 2: Select valid equations in document order ─────────────────
    equations = []
    seen_numbers = set()

    for cand in candidates:
        if cand["eq_num"] in seen_numbers:
            continue

        eq_text = _build_marker_window_equation_text(
            lines, cand["line_index"], cand["eq_text"]
        )
        if not _is_valid_equation_text(eq_text) and not _is_salvageable_numbered_display(
            eq_text
        ):
            eq_text = _build_equation_text(cand["context_before"], cand["eq_text"])

        if not _is_valid_equation_text(eq_text):
            if not _is_salvageable_numbered_display(eq_text):
                logger.info(
                    "arXiv:%s → rejected weak PDF Eq(%s): %r",
                    arxiv_id, cand["eq_num"], eq_text[:80],
                )
                continue
            low_confidence = True
        else:
            low_confidence = False

        if _has_nearby_layout_math_fragment(lines, cand["line_index"]):
            logger.info(
                "arXiv:%s → kept low-confidence split PDF Eq(%s): %r",
                arxiv_id, cand["eq_num"], eq_text[:80],
            )
            low_confidence = True

        processed_text = _pdf_text_to_latex_like(
            _postprocess_pdf_equation(_normalize_pdf_math(eq_text))
        )

        i = cand["line_index"]
        start = max(0, i - 10)
        end = min(len(lines), i + 11)
        text_before = "\n".join(lines[start:i])
        text_after = "\n".join(lines[i + 1:end])
        section = _find_section_heading(lines, i)

        eq_preview = (
            processed_text[:80] + "..."
            if len(processed_text) > 80
            else processed_text
        )
        quality_note = "low-confidence/split " if low_confidence else ""
        equations.append({
            "eq_number": cand["eq_num"],
            "equation": processed_text,
            "equation_format": "latex_like_from_pdf",
            "equation_confidence": "low" if low_confidence else "medium",
            "text_before": text_before,
            "text_after": text_after,
            "section": section,
            "audit_extract": (
                f"{eq_preview} (Equation ({cand['eq_num']}) from "
                f"{quality_note}PDF text, line {i})"
            ),
            "audit_context": (
                f"{len(text_before)} chars before, "
                f"{len(text_after)} chars after "
                f"(Section: '{section}')"
            ),
        })

        seen_numbers.add(cand["eq_num"])
        if len(equations) >= MAX_EQUATIONS:
            break

    logger.info("Found %d numbered equations in PDF source", len(equations))
    return equations
