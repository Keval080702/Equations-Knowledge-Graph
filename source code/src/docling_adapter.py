"""
Phase 2 — Optional Docling Context Enrichment
================================================

Uses IBM's Docling library (when installed) to extract cleaner surrounding
prose for PDF-extracted equations.  Docling is **not required** — when it is
not available the adapter returns the original equations unchanged.

Scope
-----
* Equation text and numbering always come from the existing PDF engines.
* Docling only provides higher-quality ``text_before`` / ``text_after``
  context that improves Phase 4 (symbol definitions) and Phase 5 (meanings).

----------
* ``enhance_pdf_context_with_docling(pdf_bytes, arxiv_id, equations)``
"""

import logging
import os
import re
import sys
import tempfile


logger = logging.getLogger(__name__)


def enhance_pdf_context_with_docling(pdf_bytes, arxiv_id, equations):
    """Return equations with cleaner Docling context when available.

    The function is deliberately optional and conservative:
    - no import-time dependency on Docling;
    - no change to equation text or equation numbers;
    - context is replaced only if the Docling window is more useful.
    """
    if not equations or os.environ.get("DISABLE_DOCLING", "").lower() in {"1", "true", "yes"}:
        return equations, 0

    markdown = _convert_pdf_to_markdown(pdf_bytes, arxiv_id)
    if not markdown:
        return equations, 0

    enhanced = []
    changed = 0
    for equation in equations:
        window = _docling_window_for_equation(markdown, equation.get("eq_number", ""))
        if window and _context_quality(window) > _context_quality(_current_context(equation)) + 0.25:
            item = dict(equation)
            before, after = _split_window(window)
            item["text_before"] = before
            item["text_after"] = after
            item["section"] = _section_from_window(before) or item.get("section", "Unknown Section")
            item["audit_context"] = (
                f"{len(before)} chars before, {len(after)} chars after "
                f"(Section: '{item['section']}'; Docling PDF context enhancement)"
            )
            enhanced.append(item)
            changed += 1
        else:
            enhanced.append(equation)

    if changed:
        logger.info("arXiv:%s → Docling enhanced context for %d equation(s)", arxiv_id, changed)
    return enhanced, changed


def _convert_pdf_to_markdown(pdf_bytes, arxiv_id):
    converter = _build_docling_converter(arxiv_id)
    if converter is None:
        return ""

    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
            handle.write(pdf_bytes)
            tmp_path = handle.name

        result = converter.convert(tmp_path)
        document = getattr(result, "document", result)
        for method_name in ("export_to_markdown", "export_to_text"):
            method = getattr(document, method_name, None)
            if callable(method):
                text = method()
                if text:
                    return _normalize_docling_text(text)
    except Exception as exc:
        logger.info("arXiv:%s → Docling conversion failed: %s", arxiv_id, exc)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return ""


def _build_docling_converter(arxiv_id):
    _prefer_active_venv_packages()
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except Exception:
        logger.info("arXiv:%s → Docling unavailable; keeping existing PDF context", arxiv_id)
        return None

    try:
        pipeline_options = PdfPipelineOptions()
        if not hasattr(pipeline_options, "do_ocr"):
            logger.info(
                "arXiv:%s → Docling OCR-disable option unavailable; skipping Docling",
                arxiv_id,
            )
            return None

        pipeline_options.do_ocr = False
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
    except Exception as exc:
        logger.info("arXiv:%s → Docling no-OCR setup failed: %s", arxiv_id, exc)
        return None


def _prefer_active_venv_packages():
    """Prefer project-venv packages over user-site packages for Docling."""
    executable_dir = os.path.dirname(sys.executable)
    if os.path.basename(executable_dir) != "bin":
        return
    venv_root = os.path.dirname(executable_dir)
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_path = os.path.join(venv_root, "lib", version, "site-packages")
    if site_path in sys.path:
        sys.path.remove(site_path)
    if os.path.isdir(site_path):
        sys.path.insert(0, site_path)


def _docling_window_for_equation(text, eq_number):
    if not eq_number:
        return ""
    patterns = [
        rf"\({re.escape(eq_number)}\)",
        rf"\\tag\{{{re.escape(eq_number)}\}}",
        rf"\bEquation\s*\(?{re.escape(eq_number)}\)?",
    ]
    matches = []
    for pattern in patterns:
        matches.extend(re.finditer(pattern, text, re.IGNORECASE))
    if not matches:
        return ""

    # Prefer markers with nearby mathematical material.
    best = max(matches, key=lambda match: _math_density(text[max(0, match.start() - 500): match.end() + 500]))
    start = max(0, best.start() - 1400)
    end = min(len(text), best.end() + 1400)
    return text[start:end]


def _split_window(window):
    middle = len(window) // 2
    return window[:middle].strip(), window[middle:].strip()


def _current_context(equation):
    return " ".join(
        part for part in (
            equation.get("section", ""),
            equation.get("text_before", ""),
            equation.get("text_after", ""),
        )
        if part
    )


def _context_quality(text):
    if not text:
        return 0.0
    normalized = text.lower()
    score = min(len(text) / 1200.0, 1.0) * 0.3
    score += min(len(re.findall(r"[A-Za-z]{3,}", text)) / 120.0, 1.0) * 0.3
    score += min(sum(normalized.count(cue) for cue in (
        "where", "denotes", "represents", "is ", "are ",
        "capacitance", "resistance", "hamiltonian", "density matrix",
        "operator", "equation", "function",
    )) / 8.0, 1.0) * 0.4
    if any(ord(char) < 32 and char not in "\n\t" for char in text):
        score -= 0.25
    return max(0.0, min(1.0, score))


def _math_density(text):
    return len(re.findall(r"[=+\-∑∫√ρσψθ]|\\(?:frac|sum|rho|sigma|theta|hat)", text))


def _section_from_window(text):
    lines = [line.strip("# ").strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines[-8:]):
        if 3 <= len(line) <= 90 and not re.search(r"[=+{}\\]", line):
            if line.isupper() or re.match(r"^\d+(?:\.\d+)*\s+[A-Z]", line):
                return line
    return ""


def _normalize_docling_text(text):
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
