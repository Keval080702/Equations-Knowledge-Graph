"""
Phase 2 — Optional LaTeX OCR Adapter (Nougat)
================================================

Provides a LaTeX OCR pass that can *repair* equation strings already found by
the PDF layout or text engines.  This is deliberately **not** an equation
finder — it only replaces the equation string for labels that the layout
engines have already identified.

The adapter is disabled by default and only activates when:
1. ``allow_latex_ocr=True`` is passed to ``parse_pdf()``.
2. A local Nougat-style CLI (``nougat`` or ``nougat_pdf``) is available on
   ``$PATH``.

Public API
----------
* ``extract_pdf_equations_with_latex_ocr(pdf_bytes, arxiv_id, limit)``
"""

import glob
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile

from src.sequence_validation import has_first_equation_prefix
from src.sequence_validation import normalize_equation_label


logger = logging.getLogger(__name__)
DEFAULT_OCR_PAGE_WINDOW = "1-4"


def extract_pdf_equations_with_latex_ocr(pdf_bytes, arxiv_id, limit=7, timeout=600):
    """Run local Nougat OCR and return numbered LaTeX equations if available."""
    nougat_cmd = _find_nougat_command()
    if not nougat_cmd:
        logger.info("arXiv:%s → LaTeX OCR skipped: nougat CLI not installed", arxiv_id)
        return []

    with tempfile.TemporaryDirectory(prefix="latexocr.") as tmp_dir:
        pdf_path = os.path.join(tmp_dir, "paper.pdf")
        out_dir = os.path.join(tmp_dir, "out")
        os.makedirs(out_dir, exist_ok=True)
        with open(pdf_path, "wb") as handle:
            handle.write(pdf_bytes)

        page_window = _ocr_page_window()
        command = [
            nougat_cmd,
            pdf_path,
            "-o",
            out_dir,
            "--no-skipping",
            "--batchsize",
            "1",
            "--pages",
            page_window,
        ]
        env = os.environ.copy()
        env.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
        if env.get("NOUGAT_FORCE_CPU") == "1":
            env["CUDA_VISIBLE_DEVICES"] = ""
            command.insert(-2, "--full-precision")
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
                env=env,
            )
        except Exception as exc:
            logger.info("arXiv:%s → LaTeX OCR failed to run: %s", arxiv_id, exc)
            return []

        if result.returncode != 0:
            logger.info(
                "arXiv:%s → LaTeX OCR returned %s: %s",
                arxiv_id,
                result.returncode,
                _short(result.stderr or result.stdout, 240),
            )
            return []

        markdown = _read_ocr_markdown(out_dir)
        if not markdown:
            logger.info("arXiv:%s → LaTeX OCR produced no markdown output", arxiv_id)
            return []

    equations = _extract_labeled_equations_from_markdown(markdown, limit=limit)
    if not equations:
        logger.info("arXiv:%s → LaTeX OCR found no numbered equations", arxiv_id)
        return []
    if not has_first_equation_prefix(equations):
        logger.info(
            "arXiv:%s → LaTeX OCR numbering is partial/non-prefix; keeping label-matched repairs only: %s",
            arxiv_id,
            [item.get("eq_number") for item in equations],
        )

    logger.info("arXiv:%s → LaTeX OCR found %d numbered equations", arxiv_id, len(equations))
    return equations[:limit]


def _find_nougat_command():
    """Find Nougat in PATH or beside the active Python executable."""
    nougat_cmd = shutil.which("nougat")
    if nougat_cmd:
        return nougat_cmd

    executable_dir = os.path.dirname(sys.executable)
    local_cmd = os.path.join(executable_dir, "nougat")
    if os.path.exists(local_cmd) and os.access(local_cmd, os.X_OK):
        return local_cmd
    return ""


def _ocr_page_window():
    """Return the first-page OCR window for PDF-only equation recovery."""
    return os.environ.get("NOUGAT_PAGE_WINDOW", DEFAULT_OCR_PAGE_WINDOW)


def _read_ocr_markdown(out_dir):
    candidates = []
    for pattern in ("**/*.mmd", "**/*.md", "**/*.tex"):
        candidates.extend(glob.glob(os.path.join(out_dir, pattern), recursive=True))
    candidates = sorted(set(candidates), key=lambda path: (os.path.getsize(path), path), reverse=True)
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                text = handle.read()
        except OSError:
            continue
        if text.strip():
            return text
    return ""


def _extract_labeled_equations_from_markdown(markdown, limit=7):
    equations = []
    seen = set()

    for match in _iter_labeled_equation_matches(markdown):
        label = normalize_equation_label(match.group("label"))
        if not label or label in seen:
            continue
        equation = _clean_ocr_equation(match.group("equation"))
        if not _usable_ocr_equation(equation):
            continue

        text_before, text_after = _ocr_context(markdown, match.start(), match.end())
        preview = _short(equation, 90)
        equations.append({
            "eq_number": label,
            "equation": equation,
            "equation_format": "latex_ocr",
            "equation_confidence": "medium",
            "text_before": text_before,
            "text_after": text_after,
            "section": _nearest_markdown_heading(markdown, match.start()),
            "audit_extract": (
                f"{preview} (Equation ({label}) from local Nougat LaTeX OCR, "
                f"pages {_ocr_page_window()})"
            ),
            "audit_context": (
                f"{len(text_before)} chars before, {len(text_after)} chars after "
                f"(Section: '{_nearest_markdown_heading(markdown, match.start())}')"
            ),
        })
        seen.add(label)
        if len(equations) >= limit:
            break

    return equations


def _iter_labeled_equation_matches(markdown):
    patterns = [
        re.compile(
            r"\\begin\{(?:equation|align|aligned|gather|multline)\*?\}"
            r"(?P<equation>.*?)"
            r"\\tag\{?\(?(?P<label>\d+(?:\.\d+)*(?:[A-Za-z](?:[-–—−][A-Za-z])?)?)\)?\}?"
            r".*?\\end\{(?:equation|align|aligned|gather|multline)\*?\}",
            re.DOTALL,
        ),
        re.compile(
            r"\$\$(?P<equation>.*?)"
            r"\\tag\{?\(?(?P<label>\d+(?:\.\d+)*(?:[A-Za-z](?:[-–—−][A-Za-z])?)?)\)?\}?"
            r".*?\$\$",
            re.DOTALL,
        ),
        re.compile(
            r"\\\[(?P<equation>.*?)"
            r"\\tag\{?\(?(?P<label>\d+(?:\.\d+)*(?:[A-Za-z](?:[-–—−][A-Za-z])?)?)\)?\}?"
            r".*?\\\]",
            re.DOTALL,
        ),
        re.compile(
            r"(?P<equation>[^\n]{8,500}?)\s*\((?P<label>\d+(?:\.\d+)*(?:[A-Za-z](?:[-–—−][A-Za-z])?)?)\)\s*$",
            re.MULTILINE,
        ),
    ]
    matches = []
    for pattern in patterns:
        matches.extend(pattern.finditer(markdown))
    return sorted(matches, key=lambda item: item.start())


def _clean_ocr_equation(equation):
    text = re.sub(r"\\tag\{?[^}]*\}?", "", equation or "")
    text = text.replace("$$", "")
    text = text.replace(r"\[", "").replace(r"\]", "")
    text = re.sub(r"\\begin\{(?:equation|align|aligned|gather|multline)\*?\}", "", text)
    text = re.sub(r"\\end\{(?:equation|align|aligned|gather|multline)\*?\}", "", text)
    text = re.sub(r"\s+", " ", text).strip(" ,")
    return text


def _usable_ocr_equation(equation):
    if len(equation) < 6:
        return False
    if _looks_like_ocr_prose_leak(equation):
        return False
    return any(token in equation for token in ("=", r"\frac", r"\sum", r"\int", r"\hat", r"\partial", r"\nabla"))


def _looks_like_ocr_prose_leak(equation):
    """Reject OCR matches that captured explanatory prose as equation text.

    Nougat sometimes emits an entire paragraph on one line and the final
    ``(n)`` equation label then gets attached to the wrong preceding text. We
    only use OCR as a text repair step, so a prose-heavy candidate is worse than
    the original PDF layout text and should be skipped.
    """
    text = equation or ""
    stripped = _strip_latex_commands_for_prose_check(text)
    words = re.findall(r"\b[A-Za-z]{3,}\b", stripped)
    if len(words) >= 8:
        return True
    prose_cues = (
        " where ",
        " since ",
        " consequently ",
        " therefore ",
        " represents ",
        " characterized ",
        " assumed ",
        " expressed ",
        " indicated ",
        " involving ",
        " reduces ",
    )
    lowered = f" {stripped.lower()} "
    return any(cue in lowered for cue in prose_cues)


def _strip_latex_commands_for_prose_check(text):
    text = re.sub(r"\\(?:begin|end)\{[^{}]*\}", " ", text or "")
    text = re.sub(r"\\[A-Za-z]+(?:\s*\{[^{}]*\})?", " ", text)
    text = re.sub(r"[_^]\{[^{}]*\}", " ", text)
    text = re.sub(r"[_^][A-Za-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text)


def _ocr_context(text, start, end, window=1200):
    before = text[max(0, start - window):start]
    after = text[end:min(len(text), end + window)]
    return _clean_context(before), _clean_context(after)


def _nearest_markdown_heading(text, position):
    prefix = text[:position]
    headings = re.findall(r"(?m)^\s{0,3}#{1,6}\s+(.+?)\s*$", prefix)
    return headings[-1].strip() if headings else "Unknown Section"


def _clean_context(text):
    text = re.sub(r"\s+", " ", text or "").strip()
    return _short(text, 1200)


def _short(text, limit):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


__all__ = ["extract_pdf_equations_with_latex_ocr"]
