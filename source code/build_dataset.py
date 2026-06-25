"""Pipeline entry point — builds the equations knowledge graph dataset.

Reads arXiv paper IDs from ``paper_list_1.txt``, downloads each paper,
extracts the first 7 numbered equations, enriches them with symbols /
meanings / relations, and writes the result to ``data/output/dataset.json``.

Run (full pipeline, 350 equations):
    python build_dataset.py

Quick smoke run (1 paper):
    python build_dataset.py --num-papers 1
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.audit_trail import (  # noqa: E402
    build_output_audit_trail,
    join_audit,
    validate_audit_trail,
)
from src.cross_equation_processing import (  # noqa: E402
    apply_cross_eq_symbol_inheritance,
    recover_definitions_paper_wide,
)
from src.equation_extraction import clean_equation_text, parse_html, parse_pdf  # noqa: E402
from src.nlp_enrichment import enrich_equations  # noqa: E402
from src.pdf_html_reconciliation import (  # noqa: E402
    looks_like_noisy_pdf_text,
    mark_equation_source,
    reconcile_pdf_with_html,
    resolve_pdf_equations_with_html_text,
    should_use_pdf_sequence_over_html,
)
from src.sequence_validation import (  # noqa: E402
    equation_numbers,
    equation_sequence_numbers,
    expected_prefix,
    has_first_equation_prefix,
    normalize_equation_label,
)
from src.source_downloader import (  # noqa: E402
    detect_best_format,
    download_html,
    download_pdf,
)


LOGGER = logging.getLogger(__name__)

MAX_EQUATIONS_PER_PAPER = 7
DEFAULT_TARGET_EQUATIONS = 350
OUTPUT_DIR = PROJECT_DIR / "data" / "output"
PAPER_LIST = PROJECT_DIR / "paper_list_1.txt"


def main():
    args = _parse_args()
    _configure_logging(args.verbose)

    paper_lines = _read_paper_list(args.paper_list)
    if args.num_papers is not None:
        paper_lines = paper_lines[: args.num_papers]

    target_equations = args.target_equations
    if target_equations is None and args.num_papers is None:
        target_equations = DEFAULT_TARGET_EQUATIONS

    output_data = {}
    total_equations = 0

    for paper_index, paper_line in enumerate(paper_lines, 1):
        clean_id = _clean_arxiv_id(paper_line)
        if not clean_id:
            continue

        LOGGER.info("--- Paper %d/target run: arXiv:%s ---", paper_index, clean_id)
        try:
            arxiv_id, paper_dict, status = process_paper(clean_id)
        except Exception as exc:
            LOGGER.exception("arXiv:%s failed: %s", clean_id, exc)
            arxiv_id = clean_id
            paper_dict = {}
            status = {"final_format": "failed", "equation_count": 0,
                      "equation_numbers": [], "suspicious_items": [str(exc)]}

        output_data[arxiv_id] = paper_dict
        total_equations += len(paper_dict)

        LOGGER.info(
            "    %s | equations=%d | total=%d | numbers=%s | suspicious=%d",
            status.get("final_format", "none"),
            status.get("equation_count", 0),
            total_equations,
            status.get("equation_numbers", []),
            len(status.get("suspicious_items", [])),
        )

        if target_equations is not None and total_equations >= target_equations:
            break

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _validate_dataset_schema(output_data)
    output_path = OUTPUT_DIR / "dataset.json"
    _write_json(output_path, output_data)
    LOGGER.info("Wrote dataset.json → %s", output_path)


def process_paper(arxiv_id):
    """Download, extract, enrich, and serialize one paper."""
    clean_id = _clean_arxiv_id(arxiv_id)
    initial_format, detection_audit = detect_best_format(clean_id)
    fallback_reason = ""

    if initial_format == "html":
        equations, final_format, source_audit, _, fallback_reason = (
            _process_html_first(clean_id, detection_audit)
        )
    else:
        equations, final_format, source_audit, _ = _process_pdf_only(
            clean_id,
            detection_audit,
        )

    if equations:
        _ensure_source_markers(equations, final_format)
        try:
            if not has_first_equation_prefix(equations):
                expected = expected_prefix(
                    len(equations),
                    equation_sequence_numbers(equations)[0]
                    if equation_sequence_numbers(equations)
                    else None,
                )
                LOGGER.warning(
                    "arXiv:%s numbering not first prefix: got %s, expected %s",
                    clean_id,
                    equation_numbers(equations),
                    expected,
                )
        except Exception as exc:
            LOGGER.warning("arXiv:%s sequence check failed: %s", clean_id, exc)

        enriched = enrich_equations(equations)
        recover_definitions_paper_wide(enriched)
        apply_cross_eq_symbol_inheritance(enriched)
    else:
        enriched = []

    paper_dict = _build_paper_dict(enriched, source_audit)
    status = _evaluate_paper(
        clean_id,
        initial_format,
        final_format,
        fallback_reason,
        enriched,
    )
    return clean_id, paper_dict, status


def _process_html_first(clean_id, detection_audit):
    html_content, html_audit = download_html(clean_id)
    source_audit = join_audit(detection_audit, html_audit)
    if not html_content:
        LOGGER.warning("arXiv:%s HTML unavailable after detection; trying PDF", clean_id)
        equations, final_format, pdf_audit, extraction_method = _process_pdf_only(
            clean_id, source_audit,
        )
        return equations, final_format, pdf_audit, extraction_method, "HTML unavailable after detection"

    html_equations = parse_html(html_content, clean_id)
    mark_equation_source(html_equations, order_source="html", text_source="html")
    if not html_equations:
        equations, final_format, pdf_audit, extraction_method = _process_pdf_only(
            clean_id, source_audit,
        )
        return equations, final_format, pdf_audit, extraction_method, "HTML contained no numbered equations"

    if not has_first_equation_prefix(html_equations):
        reason = "HTML numbering gapped; using PDF for sequence, HTML for LaTeX"
        LOGGER.warning("arXiv:%s -> %s", clean_id, reason)
        return reconcile_pdf_with_html(clean_id, html_content, html_equations, source_audit, reason)

    pdf_bytes, pdf_audit = download_pdf(clean_id)
    combined_audit = join_audit(source_audit, pdf_audit)
    if not pdf_bytes:
        return (
            html_equations, "html", source_audit,
            "HTML DOM equation extraction from allowed arXiv /html source", "",
        )

    pdf_equations = parse_pdf(pdf_bytes, clean_id, allow_latex_ocr=False)
    mark_equation_source(pdf_equations, order_source="pdf", text_source="pdf")
    if not pdf_equations or not has_first_equation_prefix(pdf_equations):
        return (
            html_equations, "html", combined_audit,
            "HTML DOM equation extraction from allowed arXiv /html source", "",
        )

    use_pdf, reason = should_use_pdf_sequence_over_html(html_equations, pdf_equations)
    if not use_pdf:
        return (
            html_equations, "html", combined_audit,
            "HTML DOM equation extraction from allowed arXiv /html source", "",
        )

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


def _process_pdf_only(clean_id, detection_audit):
    pdf_bytes, pdf_audit = download_pdf(clean_id)
    source_audit = join_audit(detection_audit, pdf_audit)
    if not pdf_bytes:
        return [], "pdf", source_audit, "PDF source unavailable"

    equations = parse_pdf(pdf_bytes, clean_id, allow_latex_ocr=True)
    mark_equation_source(equations, order_source="pdf", text_source="pdf")
    extraction_method = "PDF layout equation extraction from allowed arXiv /pdf source"
    if any(item.get("equation_text_source") == "latex_ocr" for item in equations):
        extraction_method += "; PDF-only equation text improved with local Nougat LaTeX OCR"
    return equations, "pdf", source_audit, extraction_method


def _build_paper_dict(equations, source_audit):
    paper = {}
    for equation in equations:
        eq_label = normalize_equation_label(equation.get("eq_number"))
        stored_equation = clean_equation_text(equation.get("equation", ""))
        audit_equation = dict(equation)
        audit_equation["stored_equation"] = stored_equation
        paper[eq_label] = {
            "equation": stored_equation,
            "meaning": _stored_meaning(equation),
            "symbols": equation.get("symbols", {}),
            "relations": _strict_relations(equation.get("relations", {})),
            "audit-trail": build_output_audit_trail(audit_equation, source_audit),
        }
    return paper


def _stored_meaning(equation):
    meaning = str(equation.get("meaning", "") or "").strip()
    confidence = str(equation.get("meaning_confidence", "") or "").lower()
    if confidence == "low":
        return ""
    return meaning


def _strict_relations(relations):
    cleaned = {}
    for key, relation in relations.items():
        relation_key = normalize_equation_label(key)
        cleaned[relation_key] = {
            "grade": relation.get("grade", "none"),
            "description": relation.get("description", ""),
        }
    return cleaned


def _evaluate_paper(arxiv_id, initial_format, final_format, fallback_reason, equations):
    numbers = equation_numbers(equations)
    sequence_numbers = equation_sequence_numbers(equations)
    expected = expected_prefix(
        len(sequence_numbers),
        sequence_numbers[0] if sequence_numbers else None,
    )
    consecutive = bool(sequence_numbers) and sequence_numbers == expected
    suspicious = _suspicious_items(equations, consecutive, expected, sequence_numbers)
    return {
        "arxiv_id": arxiv_id,
        "initial_format": initial_format,
        "final_format": final_format,
        "fallback_reason": fallback_reason,
        "equation_count": len(equations),
        "equation_numbers": numbers,
        "numbering_is_consecutive": consecutive,
        "expected_equation_numbers": expected,
        "suspicious_items": suspicious,
    }


def _suspicious_items(equations, consecutive, expected, sequence_numbers):
    items = []
    if equations and not consecutive:
        items.append(
            f"numbering gap: sequence labels {sequence_numbers}, expected {expected}"
        )
    for equation in equations:
        label = normalize_equation_label(equation.get("eq_number"))
        text = equation.get("equation", "")
        if not text:
            items.append(f"Eq({label}) empty equation text")
        if looks_like_noisy_pdf_text(text):
            items.append(f"Eq({label}) noisy PDF-like equation text")
    return items


def _ensure_source_markers(equations, final_format):
    source = "html" if final_format == "html" else "pdf"
    mark_equation_source(equations, order_source=source, text_source=source)


def _validate_dataset_schema(data):
    """Validate the final project JSON shape before writing it."""
    required_fields = {"equation", "meaning", "symbols", "relations", "audit-trail"}
    allowed_grades = {"none", "strong", "potential"}
    for arxiv_id, paper in data.items():
        if not isinstance(paper, dict):
            raise ValueError(f"{arxiv_id}: paper entry must be a dictionary")
        equation_keys = set(paper)
        for eq_number, equation in paper.items():
            fields = set(equation)
            if fields != required_fields:
                raise ValueError(
                    f"{arxiv_id} Eq({eq_number}) fields {sorted(fields)} "
                    f"!= {sorted(required_fields)}"
                )
            if not isinstance(equation["equation"], str):
                raise ValueError(f"{arxiv_id} Eq({eq_number}) equation must be string")
            if not isinstance(equation["meaning"], str):
                raise ValueError(f"{arxiv_id} Eq({eq_number}) meaning must be string")
            if not isinstance(equation["symbols"], dict):
                raise ValueError(f"{arxiv_id} Eq({eq_number}) symbols must be dict")
            if not isinstance(equation["relations"], dict):
                raise ValueError(f"{arxiv_id} Eq({eq_number}) relations must be dict")
            if not isinstance(equation["audit-trail"], dict) or not equation["audit-trail"]:
                raise ValueError(f"{arxiv_id} Eq({eq_number}) audit-trail must be non-empty dict")
            validate_audit_trail(equation["audit-trail"])

            expected_relations = equation_keys - {eq_number}
            if set(equation["relations"]) != expected_relations:
                raise ValueError(
                    f"{arxiv_id} Eq({eq_number}) relation keys "
                    f"{sorted(equation['relations'])} != {sorted(expected_relations)}"
                )
            for other_eq, relation in equation["relations"].items():
                grade = relation.get("grade")
                if grade not in allowed_grades:
                    raise ValueError(
                        f"{arxiv_id} Eq({eq_number})->Eq({other_eq}) invalid grade {grade!r}"
                    )
                if "description" not in relation:
                    raise ValueError(
                        f"{arxiv_id} Eq({eq_number})->Eq({other_eq}) missing description"
                    )


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _read_paper_list(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if _clean_arxiv_id(line)]


def _clean_arxiv_id(text):
    text = str(text or "").strip()
    if not text or text.startswith("#"):
        return ""
    text = text.replace("arXiv:", "").strip()
    match = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+/\d{7}(?:v\d+)?)", text)
    return match.group(1) if match else ""


def _parse_args():
    parser = argparse.ArgumentParser(description="Run equation extraction pipeline.")
    parser.add_argument("--paper-list", default=str(PAPER_LIST))
    parser.add_argument("--num-papers", type=int, default=None)
    parser.add_argument("--target-equations", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _configure_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )


if __name__ == "__main__":
    main()
