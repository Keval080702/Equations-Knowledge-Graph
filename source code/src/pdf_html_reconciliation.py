"""PDF / HTML source reconciliation.

Compares PDF and HTML equation extractions for the same paper and selects
the best combination: PDF provides the canonical numbered sequence; HTML
provides higher-quality LaTeX when the similarity alignment is trusted.

Public API
----------
* ``reconcile_pdf_with_html(clean_id, html_content, html_equations, source_audit, reason)``
* ``resolve_pdf_equations_with_html_text(pdf_equations, html_content, arxiv_id)``
* ``should_use_pdf_sequence_over_html(html_equations, pdf_equations)``
* ``mark_equation_source(equations, order_source, text_source)``
"""

import logging
import re
from difflib import SequenceMatcher

from src.audit_trail import compact_audit_text, join_audit, preview_audit_text
from src.equation_extraction import parse_html, parse_pdf
from src.sequence_validation import (
    equation_sequence_label,
    has_first_equation_prefix,
    normalize_equation_label,
)
from src.source_downloader import download_pdf
from src.symbol_extraction import _normalize_symbol_key, extract_symbol_identifiers

logger = logging.getLogger(__name__)

MAX_EQUATIONS_PER_PAPER = 7


def mark_equation_source(equations, order_source, text_source):
    for equation in equations:
        equation.setdefault("equation_order_source", order_source)
        equation.setdefault("equation_text_source", text_source)
        equation.setdefault("context_source", text_source)


def reconcile_pdf_with_html(clean_id, html_content, html_equations, source_audit, reason):
    """Use PDF for canonical sequence order and HTML for LaTeX text where trusted."""
    pdf_bytes, pdf_audit = download_pdf(clean_id)
    combined_audit = join_audit(source_audit, pdf_audit)
    if not pdf_bytes:
        return html_equations, "html", combined_audit, (
            "HTML DOM equation extraction from allowed arXiv /html source; "
            "PDF fallback unavailable"
        ), reason

    pdf_equations = parse_pdf(pdf_bytes, clean_id, allow_latex_ocr=False)
    mark_equation_source(pdf_equations, order_source="pdf", text_source="pdf")
    if not pdf_equations or not has_first_equation_prefix(pdf_equations):
        return html_equations, "html", combined_audit, (
            "HTML DOM equation extraction from allowed arXiv /html source; "
            "PDF fallback did not provide a reliable first sequence"
        ), reason

    resolved_equations, resolved_count = resolve_pdf_equations_with_html_text(
        pdf_equations, html_content, clean_id,
    )
    extraction_method = (
        "PDF layout equation extraction from allowed arXiv /pdf fallback because "
        f"{reason}; PDF selected for canonical sequence/order"
    )
    if resolved_count:
        extraction_method += f"; {resolved_count} equation text(s) resolved from HTML LaTeX"
    return resolved_equations, "pdf", combined_audit, extraction_method, reason


def resolve_pdf_equations_with_html_text(pdf_equations, html_content, arxiv_id):
    """Keep PDF order/labels; substitute HTML LaTeX text where alignment is trusted."""
    html_candidates = parse_html(html_content, arxiv_id) if html_content else []
    mark_equation_source(html_candidates, order_source="html", text_source="html")

    resolved = []
    used_html = set()
    replaced = 0
    for pdf_equation in pdf_equations:
        match, score, method = _best_html_match(pdf_equation, html_candidates, used_html)
        if match is None:
            if _is_garbled_pdf_text(pdf_equation.get("equation", "")):
                match = _html_by_label(pdf_equation, html_candidates, used_html)
                if match is not None:
                    score, method = 0.0, "same equation label (garbled PDF → HTML)"
            if match is None:
                item = dict(pdf_equation)
                item["audit_extract"] = (
                    f"{item.get('audit_extract', '')} "
                    "(PDF canonical numbering/order; no reliable HTML text match)"
                ).strip()
                item["context_source"] = item.get("context_source", "pdf")
                resolved.append(item)
                continue

        if method == "core formula symbol overlap":
            item = dict(pdf_equation)
            item["text_before"] = match.get("text_before", item.get("text_before", ""))
            item["text_after"] = match.get("text_after", item.get("text_after", ""))
            item["section"] = match.get("section", item.get("section", ""))
            item["context_source"] = "html"
            item["audit_context"] = (
                f"{match.get('audit_context', item.get('audit_context', ''))} "
                "(matched by core formula symbol overlap)"
            ).strip()
            item["audit_extract"] = (
                f"{item.get('audit_extract', '')} "
                f"(PDF canonical numbering/order; HTML context matched by {method}, "
                f"similarity={score:.2f}; PDF equation text retained)"
            ).strip()
            resolved.append(item)
            continue

        used_html.add(id(match))
        item = dict(pdf_equation)
        item["equation"] = match["equation"]
        item["equation_format"] = "latex"
        item["equation_confidence"] = "high"
        item["text_before"] = match.get("text_before", item.get("text_before", ""))
        item["text_after"] = match.get("text_after", item.get("text_after", ""))
        item["section"] = match.get("section", item.get("section", ""))
        item["identifier_candidates"] = match.get("identifier_candidates", [])
        item["audit_identifiers"] = match.get("audit_identifiers", "")
        item["equation_order_source"] = "pdf"
        item["equation_text_source"] = "html"
        item["context_source"] = "html"
        item["audit_context"] = match.get("audit_context", item.get("audit_context", ""))
        preview = preview_audit_text(match["equation"], 90)
        item["audit_extract"] = (
            f"{preview} (Equation ({item['eq_number']}) uses PDF canonical "
            f"numbering/order; text resolved from HTML by {method}, "
            f"similarity={score:.2f})"
        )
        resolved.append(item)
        replaced += 1

    return resolved, replaced


def should_use_pdf_sequence_over_html(html_equations, pdf_equations):
    similarity = _equation_set_similarity(html_equations, pdf_equations)
    html_quality = _equation_set_quality(html_equations)
    pdf_quality = _equation_set_quality(pdf_equations)
    reason = (
        "HTML/PDF first-equation sequence mismatch "
        f"(similarity={similarity:.2f}, html_quality={html_quality:.2f}, "
        f"pdf_quality={pdf_quality:.2f})"
    )
    if pdf_quality < 0.50:
        return False, ""
    if html_quality > pdf_quality and _pdf_equations_appear_garbled(pdf_equations):
        return False, ""
    if similarity < 0.28:
        return True, reason
    if _sources_disagree(html_equations, pdf_equations) and similarity < 0.48:
        return True, reason
    return False, ""


# ── Internal helpers ──────────────────────────────────────────────────────────

def _best_html_match(pdf_equation, html_candidates, used_html):
    pdf_number = normalize_equation_label(pdf_equation.get("eq_number"))
    pdf_seq = equation_sequence_label(pdf_number)
    pdf_text = pdf_equation.get("equation", "")

    same_number = [
        item for item in html_candidates
        if id(item) not in used_html
        and equation_sequence_label(item.get("eq_number")) == pdf_seq
    ]
    best = _best_by_similarity(pdf_text, same_number)
    if best and best[1] >= 0.55:
        return best[0], best[1], "matching equation label"

    remaining = [item for item in html_candidates if id(item) not in used_html]
    best = _best_by_similarity(pdf_text, remaining)
    if best and best[1] >= 0.72:
        return best[0], best[1], "formula similarity"

    best = _best_by_core_symbol_overlap(pdf_text, remaining)
    if best and best[1] >= 0.82:
        return best[0], best[1], "core formula symbol overlap"

    return None, 0.0, ""


def _is_garbled_pdf_text(equation_text):
    text = str(equation_text or "")
    if "\x00" in text:
        return True
    tokens = text.split()
    short_plain = sum(1 for t in tokens if re.match(r"^[A-Za-z]{1,3}$", t))
    latex_cmds = len(re.findall(r"\\[A-Za-z]+", text))
    if short_plain > 5 and short_plain > latex_cmds:
        return True
    if re.search(r"\\partial[A-Za-z]", text):
        return True
    if re.search(r"[^\x00-\x7F]", text):
        return True
    if re.search(
        r"\\(?:hat|vec|bar|tilde|dot|ddot|breve|check)\{[A-Za-z0-9]\}[0-9]", text,
    ):
        return True
    if r"\\" in text:
        digit_tokens = sum(1 for t in tokens if re.match(r"^[0-9]+$", t))
        complex_structures = len(re.findall(
            r"\\(?:frac|hat|vec|bar|tilde|sum|int|sqrt|mathcal|mathbb|bra|ket)\b", text,
        ))
        if digit_tokens > 3 and complex_structures < 2:
            return True
    return False


def _pdf_equations_appear_garbled(pdf_equations):
    return any(
        _is_garbled_pdf_text(eq.get("equation", ""))
        for eq in pdf_equations
    )


def _html_by_label(pdf_equation, html_candidates, used_html):
    pdf_seq = equation_sequence_label(normalize_equation_label(pdf_equation.get("eq_number")))
    for item in html_candidates:
        if id(item) in used_html:
            continue
        if equation_sequence_label(item.get("eq_number")) == pdf_seq:
            return item
    return None


def _best_by_similarity(reference_text, candidates):
    best_item, best_score = None, 0.0
    for item in candidates:
        score = _source_equation_similarity(reference_text, item.get("equation", ""))
        if score > best_score:
            best_item, best_score = item, score
    return (best_item, best_score) if best_item else None


def _best_by_core_symbol_overlap(reference_text, candidates):
    reference_terms = _core_formula_terms(reference_text)
    if len(reference_terms) < 2:
        return None
    best_item, best_score = None, 0.0
    for item in candidates:
        candidate_terms = _core_formula_terms(item.get("equation", ""))
        if not candidate_terms:
            continue
        overlap = reference_terms & candidate_terms
        coverage = len(overlap) / len(reference_terms)
        precision = len(overlap) / len(candidate_terms)
        score = (0.75 * coverage) + (0.25 * precision)
        if _operator_profile(reference_text) & _operator_profile(item.get("equation", "")):
            score += 0.05
        if score > best_score:
            best_item, best_score = item, min(score, 1.0)
    return (best_item, best_score) if best_item else None


def _core_formula_terms(text):
    terms = set()
    for raw_symbol in extract_symbol_identifiers(None, str(text or "")):
        key = _normalize_symbol_key(raw_symbol)
        if not key:
            continue
        key = re.sub(r"_[a-z0-9]+$", "", key)
        if key in {"i", "j", "k", "l", "m", "delta", "pi"}:
            continue
        terms.add(key)
    return terms


def _operator_profile(text):
    text = str(text or "")
    profile = set()
    if "[" in text or r"\left[" in text:
        profile.add("bracket")
    if "=" in text:
        profile.add("equals")
    if r"\sum" in text:
        profile.add("sum")
    if r"\int" in text:
        profile.add("integral")
    if r"\frac{d" in text or r"\partial" in text:
        profile.add("differential")
    return profile


def _equation_set_similarity(left_equations, right_equations):
    count = min(3, len(left_equations), len(right_equations))
    if count == 0:
        return 0.0
    scores = [
        _source_equation_similarity(
            left_equations[i].get("equation", ""),
            right_equations[i].get("equation", ""),
        )
        for i in range(count)
    ]
    return sum(scores) / len(scores)


def _source_equation_similarity(left_text, right_text):
    left_sig = _equation_signature(left_text)
    right_sig = _equation_signature(right_text)
    if not left_sig or not right_sig:
        return 0.0
    left_tokens = set(left_sig.split())
    right_tokens = set(right_sig.split())
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
    sequence = SequenceMatcher(None, left_sig, right_sig).ratio()
    return (0.55 * jaccard) + (0.45 * sequence)


def _equation_signature(text):
    text = str(text or "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r"\\(?:hat|vec|bar|tilde|mathrm|operatorname|mathcal|mathbb)\s*\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    text = re.sub(r"[^A-Za-z0-9_+\-=*/^{}().,]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _equation_set_quality(equations):
    if not equations:
        return 0.0
    scores = [_equation_text_quality(item) for item in equations[:MAX_EQUATIONS_PER_PAPER]]
    return sum(scores) / len(scores)


def _equation_text_quality(equation):
    text = str(equation.get("equation", ""))
    if not text:
        return 0.0
    score = 0.25
    if len(text) >= 8:
        score += 0.15
    if any(token in text for token in ("=", "\\frac", "∂", "+", "-", "\\sum", "\\int", "[")):
        score += 0.20
    fmt = equation.get("equation_format")
    if fmt == "latex":
        score += 0.20
    elif fmt == "latex_ocr":
        score += 0.15
    elif fmt == "latex_like_from_pdf":
        score += 0.10
    conf = equation.get("equation_confidence")
    if conf == "high":
        score += 0.20
    elif conf == "medium":
        score += 0.10
    if looks_like_noisy_pdf_text(text):
        score -= 0.30
    return max(0.0, min(0.90, score))


def looks_like_noisy_pdf_text(text):
    if "\x00" in text:
        return True
    words = re.findall(r"[A-Za-z]{5,}", text)
    math_marks = re.findall(r"[=+\-*/∑∫√_^{}()[\]]", text)
    return len(words) > 5 and len(math_marks) < 3


def _sources_disagree(left_equations, right_equations):
    if not left_equations or not right_equations:
        return False
    left_lhs = _lhs_signature(left_equations[0].get("equation", ""))
    right_lhs = _lhs_signature(right_equations[0].get("equation", ""))
    if not left_lhs or not right_lhs:
        return False
    return _source_equation_similarity(left_lhs, right_lhs) < 0.35


def _lhs_signature(equation_text):
    text = str(equation_text or "")
    if "=" not in text:
        return text[:80]
    return text.split("=", 1)[0]
