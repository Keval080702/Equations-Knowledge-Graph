"""
Phase 2 — Equation Extraction from HTML (Primary)
===================================================

Parses ar5iv HTML pages to extract numbered equations using the structured
document object model.

Key HTML structures used
------------------------
* ``<table class="ltx_equation">``         — numbered equation container
* ``<math alttext="...">``                 — MathML element carrying LaTeX
* ``<annotation encoding="...">``          — annotation with raw LaTeX string
* ``<span class="ltx_tag_equation">``      — equation number label, e.g. ``(1)``
* ``<p>`` / ``<section>``                  — surrounding context and headings

Extraction strategy
-------------------
1. Find all ``ltx_equation`` / ``ltx_equationgroup`` containers.
2. Prefer equations in the main body; filter appendix sections.
3. Select the first 7 sequentially numbered equations (1, 2, …).
4. Extract LaTeX from the annotation tag; fall back to alttext.
5. Collect up to 1 000 chars of context before and after each equation.

Public API
----------
* ``parse_html(html_content, arxiv_id)``
"""

import re
import logging

from bs4 import BeautifulSoup

from src.sequence_validation import equation_sequence_label, label_sort_key

logger = logging.getLogger(__name__)

__all__ = ["parse_html"]

# Maximum equations to extract per paper (project requirement)
MAX_EQUATIONS = 7
MAX_SCAN_EQUATIONS = MAX_EQUATIONS + 10
EQUATION_TAG_CLASSES = ("ltx_tag_equation", "ltx_tag_equationgroup")


# ── Public Functions ───────────────────────────────────────────────────────

def parse_html(html_content, arxiv_id):
    """Extract numbered equations from an ar5iv HTML page.

    Parameters
    ----------
    html_content : str
        Full HTML string from https://arxiv.org/html/{id}.
    arxiv_id : str
        The arXiv paper ID (for logging).

    Returns
    -------
    equations : list of dict
        Each dict has keys:
          - eq_number : str   (equation number as stated in paper, e.g. "1")
          - equation  : str   (math string from HTML annotation/alttext)
          - text_before : str (surrounding text before equation)
          - text_after  : str (surrounding text after equation)
          - section     : str (section heading)
          - audit_extract : str (audit trail for extraction)
          - audit_context : str (audit trail for surrounding text)
    """
    soup = BeautifulSoup(html_content, "html.parser")
    equations = []

    # Step 2a: Anchor extraction on visible equation numbers. This handles
    # arXiv HTML equation groups where one equation is split across several
    # table cells or row-spanned continuation rows.
    number_tags = soup.find_all(class_=_is_equation_tag_class)
    seen_numbers = set()

    for number_tag in number_tags:
        if len(equations) >= MAX_SCAN_EQUATIONS:
            break

        # ── Extract equation number ────────────────────────────────────
        eq_num = _extract_equation_number(number_tag)
        if eq_num is None:
            continue  # skip unnumbered equations
        if eq_num in seen_numbers:
            continue

        # ── Extract math string from HTML annotation/alttext ───────────
        rows, context_container = _equation_rows_for_tag(number_tag)
        math_text, math_source = _extract_math_from_rows(rows)
        identifiers = _identifiers_from_rows(rows)
        if not math_text:
            continue  # skip if no math content found
        if not _looks_like_equation(math_text):
            continue

        # ── Extract surrounding text ───────────────────────────────────
        text_before = _get_text_before(context_container)
        text_after = _get_text_after(context_container)
        section = _get_section_title(context_container)

        # ── Build audit trail messages ─────────────────────────────────
        math_preview = math_text[:80] + "..." if len(math_text) > 80 else math_text
        audit_extract = (
            f"{math_preview} (Equation ({eq_num}) parsed from {math_source})"
        )
        audit_context = (
            f"{len(text_before)} chars before, {len(text_after)} chars after "
            f"(Section: '{section}')"
        )

        equations.append({
            "eq_number": str(eq_num),
            "equation": math_text,
            "equation_format": "latex",
            "equation_confidence": "high",
            "text_before": text_before,
            "text_after": text_after,
            "section": section,
            "audit_extract": audit_extract,
            "audit_context": audit_context,
            "identifier_candidates": identifiers,
            "audit_identifiers": (
                f"Found {len(identifiers)} unique identifiers from HTML <mi> tags: "
                f"{', '.join(identifiers) if identifiers else 'none'}"
            ),
            "_html_table_id": context_container.get("id", "") if context_container else "",
        })
        seen_numbers.add(eq_num)

    equations = _recover_untagged_id_equations(soup, equations)
    equations = _recover_untagged_gap_equations(soup, equations)
    equations = _sort_equations_by_number(equations)
    equations = _repair_gapped_html_numbering_from_display_order(soup, equations)
    for equation in equations:
        equation.pop("_html_table_id", None)

    # When the main body already has ≥ MAX_EQUATIONS equations, prefer them
    # over appendix equations.  Papers sometimes restart numbering from (1)
    # inside an appendix, which would otherwise displace main-body equations.
    equations = _prefer_main_body_equations(equations)

    logger.info("Found %d numbered equations in HTML source", len(equations))
    return equations[:MAX_EQUATIONS]


# ── Internal helpers ───────────────────────────────────────────────────────

def _is_equation_tag_class(value):
    if not value:
        return False
    if isinstance(value, (list, tuple, set)):
        return any(item in EQUATION_TAG_CLASSES for item in value)
    return value in EQUATION_TAG_CLASSES


def _find_equation_tag(container):
    if not container:
        return None
    return container.find("span", class_=_is_equation_tag_class)


def _has_equation_tag(container):
    return _find_equation_tag(container) is not None


def _extract_equation_number(container):
    """Extract the equation number from a container element.

    Looks for <td class="ltx_eqn_num"> or arXiv equation/equationgroup tags.
    Returns the number as a string, or None if unnumbered.
    """
    text = container.get_text(strip=True) if container else ""
    match = re.search(r"\((\d+(?:\.\d+)*(?:[A-Za-z](?:\s*[-–—−]\s*[A-Za-z])?)?)\)", text)
    if match:
        return _normalize_html_equation_label(match.group(1))

    # Try <td class="ltx_eqn_num">
    num_elem = container.find(class_="ltx_eqn_num")
    if num_elem:
        text = num_elem.get_text(strip=True)
        match = re.search(r"\((\d+(?:\.\d+)*(?:[A-Za-z](?:\s*[-–—−]\s*[A-Za-z])?)?)\)", text)
        if match:
            return _normalize_html_equation_label(match.group(1))

    # Try <span class="ltx_tag ltx_tag_equation"> and equation groups like (1a-d).
    tag_elem = _find_equation_tag(container)
    if tag_elem:
        text = tag_elem.get_text(strip=True)
        match = re.search(r"\((\d+(?:\.\d+)*(?:[A-Za-z](?:\s*[-–—−]\s*[A-Za-z])?)?)\)", text)
        if match:
            return _normalize_html_equation_label(match.group(1))

    return None


def _normalize_html_equation_label(label):
    return re.sub(r"\s+", "", str(label)).replace("–", "-").replace("—", "-").replace("−", "-")


def _equation_rows_for_tag(tag_elem):
    """Return table rows that belong to a numbered equation tag."""
    row = tag_elem.find_parent("tr")
    table = tag_elem.find_parent(["table", "div"]) or row or tag_elem
    if row is None:
        return [tag_elem], table

    eqno_cell = tag_elem.find_parent(["td", "span"])
    rowspan = 1
    if eqno_cell and eqno_cell.name == "td":
        try:
            rowspan = int(eqno_cell.get("rowspan", "1"))
        except ValueError:
            rowspan = 1

    rows = _preceding_continuation_rows(row, table)
    rows.append(row)
    sibling = row.find_next_sibling("tr")
    while sibling is not None and len(rows) < rowspan:
        rows.append(sibling)
        sibling = sibling.find_next_sibling("tr")

    rows.extend(_following_continuation_rows(row, table))
    return _unique_nodes(rows), table


def _preceding_continuation_rows(row, table, max_rows=5):
    """Collect math rows above a number tag when the tag is on the final row."""
    if table is None or table.name != "table":
        return []

    continuations = []
    parent = row.parent
    sibling_row = row.find_previous_sibling("tr")
    while sibling_row is not None and len(continuations) < max_rows:
        if _has_equation_tag(sibling_row):
            break
        if sibling_row.find("math"):
            continuations.insert(0, sibling_row)
        sibling_row = sibling_row.find_previous_sibling("tr")

    sibling_group = parent.find_previous_sibling() if parent else None
    while sibling_group is not None and len(continuations) < max_rows:
        if sibling_group.name != "tbody":
            sibling_group = sibling_group.find_previous_sibling()
            continue
        group_rows = sibling_group.find_all("tr", recursive=False)
        if any(_has_equation_tag(group_row) for group_row in group_rows):
            break
        math_rows = [group_row for group_row in group_rows if group_row.find("math")]
        if not math_rows:
            break
        for math_row in reversed(math_rows):
            continuations.insert(0, math_row)
            if len(continuations) >= max_rows:
                break
        sibling_group = sibling_group.find_previous_sibling()
    return continuations


def _following_continuation_rows(row, table, max_rows=5):
    """Collect equation continuation rows in the same HTML equation table.

    arXiv HTML can split one display equation across several ``tbody`` blocks:
    the first row carries the visible equation number, and following rows carry
    operators such as ``= ...`` without their own number. Those rows still
    belong to the same equation.
    """
    if table is None or table.name != "table":
        return []

    parent = row.parent
    continuations = []

    # Continuations can be later rows in the same tbody.
    sibling_row = row.find_next_sibling("tr")
    while sibling_row is not None and len(continuations) < max_rows:
        if _has_equation_tag(sibling_row):
            return continuations
        if sibling_row.find("math"):
            continuations.append(sibling_row)
        sibling_row = sibling_row.find_next_sibling("tr")

    # Or later tbody groups within the same equation table.
    sibling_group = parent.find_next_sibling() if parent else None
    while sibling_group is not None and len(continuations) < max_rows:
        if sibling_group.name != "tbody":
            sibling_group = sibling_group.find_next_sibling()
            continue
        group_rows = sibling_group.find_all("tr", recursive=False)
        if any(_has_equation_tag(group_row) for group_row in group_rows):
            break
        math_rows = [group_row for group_row in group_rows if group_row.find("math")]
        if not math_rows:
            break
        continuations.extend(math_rows[: max_rows - len(continuations)])
        sibling_group = sibling_group.find_next_sibling()

    return continuations


def _unique_nodes(nodes):
    unique = []
    seen = set()
    for node in nodes:
        node_id = id(node)
        if node_id in seen:
            continue
        unique.append(node)
        seen.add(node_id)
    return unique


def _recover_untagged_gap_equations(soup, equations):
    """Recover untagged HTML display equations that fill a numbering gap.

    Some arXiv HTML pages render a numbered display equation as a standalone
    equation table between two tagged equations, but the visible number itself
    is not attached to that table. If the surrounding tagged equations are
    consecutive except for exactly those missing numbers, the intervening math
    table is the best source and avoids falling back to lossy PDF text.
    """
    by_number = {}
    for equation in equations:
        try:
            by_number[int(equation["eq_number"])] = equation
        except (KeyError, TypeError, ValueError):
            continue

    if not by_number:
        return equations

    recovered = []
    numbers = sorted(by_number)

    first_number = numbers[0]
    if first_number > 1:
        first_table = _table_for_equation(soup, by_number[first_number])
        if first_table is not None:
            leading_candidates = _untagged_math_tables_before(first_table)
            missing_count = first_number - 1
            for missing_number, table in zip(
                range(1, first_number), leading_candidates[-missing_count:]
            ):
                recovered_equation = _build_recovered_gap_equation(
                    missing_number,
                    table,
                    f"before Eq({first_number})",
                )
                if recovered_equation:
                    recovered.append(recovered_equation)

    for previous, following in zip(numbers, numbers[1:]):
        gap = following - previous
        if gap <= 1:
            continue

        previous_table = _table_for_equation(soup, by_number[previous])
        following_table = _table_for_equation(soup, by_number[following])
        if previous_table is None or following_table is None:
            continue

        candidates = _untagged_math_tables_between(previous_table, following_table)
        if not candidates:
            continue

        for offset, table in enumerate(candidates[: gap - 1], 1):
            missing_number = previous + offset
            recovered_equation = _build_recovered_gap_equation(
                missing_number,
                table,
                f"between Eq({previous}) and Eq({following})",
            )
            if recovered_equation:
                recovered.append(recovered_equation)

    if recovered:
        logger.info("Recovered %d untagged gap equation(s) from HTML", len(recovered))
    return equations + recovered


def _recover_untagged_id_equations(soup, equations):
    """Recover untagged equations whose HTML table id carries the equation number."""
    existing_numbers = _existing_equation_numbers(equations)
    recovered = []

    for table in soup.find_all("table"):
        if _has_equation_tag(table):
            continue
        if not table.find("math"):
            continue

        eq_num = _extract_equation_number_from_table_id(table)
        if eq_num is None:
            continue
        if eq_num in existing_numbers:
            continue
        if eq_num > MAX_SCAN_EQUATIONS:
            continue

        recovered_equation = _build_recovered_gap_equation(
            eq_num,
            table,
            "from HTML equation table id",
        )
        if recovered_equation:
            recovered.append(recovered_equation)
            existing_numbers.add(eq_num)

    if recovered:
        logger.info(
            "Recovered %d untagged equation(s) from HTML table ids", len(recovered)
        )
    return equations + recovered


def _existing_equation_numbers(equations):
    numbers = set()
    for equation in equations:
        try:
            numbers.add(int(equation_sequence_label(equation["eq_number"])))
        except (KeyError, TypeError, ValueError):
            continue
    return numbers


def _extract_equation_number_from_table_id(table):
    """Extract equation number from plain arXiv HTML ids like ``S2.E4``.

    We intentionally do not trust ``EGx`` ids here, because those can be display
    group counters rather than equation numbers. ``EGx`` tables are recovered
    only when they fill a verified gap between tagged equations.
    """
    table_id = table.get("id", "")
    match = re.search(r"(?:^|[.])E(\d+)$", table_id)
    if not match:
        return None
    return int(match.group(1))


def _sort_equations_by_number(equations):
    def sort_key(equation):
        return label_sort_key(equation.get("eq_number"))

    return sorted(equations, key=sort_key)


def _repair_gapped_html_numbering_from_display_order(soup, equations):
    """Use HTML display order when arXiv equation tags are gapped/misaligned.

    Some arXiv HTML pages expose high-quality TeX annotations but attach
    misleading visible numbers to split display tables. If the tagged sequence
    starts at 1 but has gaps, PDF fallback would be worse, so we segment the
    numbered display rows in DOM order and assign the required first-equation
    prefix.
    """
    numbers = []
    for equation in equations:
        try:
            if "." in str(equation["eq_number"]):
                return equations
            numbers.append(int(equation["eq_number"]))
        except (KeyError, TypeError, ValueError):
            return equations

    if not numbers:
        return equations
    if numbers == list(range(1, len(numbers) + 1)):
        return equations
    if min(numbers) != 1:
        return equations

    candidates = _html_display_order_candidates(soup)
    if len(candidates) < len(equations):
        return equations

    repaired = []
    for index, candidate in enumerate(candidates[:MAX_EQUATIONS], 1):
        item = dict(candidate)
        item["eq_number"] = str(index)
        preview = item["equation"][:80] + "..." if len(item["equation"]) > 80 else item["equation"]
        item["audit_extract"] = (
            f"{preview} (Equation ({index}) repaired from HTML display order; "
            f"original source {candidate.get('audit_source', 'row annotations')})"
        )
        item.pop("audit_source", None)
        repaired.append(item)

    if repaired:
        logger.info(
            "Repaired gapped HTML numbering from display order: %s -> %s",
            numbers[:MAX_EQUATIONS],
            [item["eq_number"] for item in repaired],
        )
        return repaired
    return equations


def _html_display_order_candidates(soup):
    """Return equation-like display segments in DOM order."""
    candidates = []
    seen_tables = set()
    for table in soup.find_all("table"):
        if id(table) in seen_tables:
            continue
        seen_tables.add(id(table))
        if not table.find("math"):
            continue
        if not _has_equation_tag(table) and _extract_equation_number_from_table_id(table) is None:
            continue
        candidates.extend(_segments_from_equation_table(table))
        if len(candidates) >= MAX_SCAN_EQUATIONS:
            break
    return candidates


def _segments_from_equation_table(table):
    """Split a numbered HTML equation table into display-level segments."""
    rows = table.find_all("tr")
    if not rows:
        return []

    tag_indices = [
        index for index, row in enumerate(rows)
        if _has_equation_tag(row)
    ]
    if not tag_indices:
        equation = _build_display_order_equation(
            table,
            rows,
            f"table {table.get('id', '<no-id>')}",
        )
        return [equation] if equation else []

    row_segments = []
    for position, tag_index in enumerate(tag_indices):
        next_tag_index = tag_indices[position + 1] if position + 1 < len(tag_indices) else len(rows)
        tagged_row = rows[tag_index]
        following_rows = rows[tag_index + 1:next_tag_index]

        if row_segments and not _row_starts_new_expression(tagged_row):
            row_segments[-1].extend(rows[tag_index:next_tag_index])
            continue

        if following_rows and _row_starts_new_expression(following_rows[0]):
            row_segments.append([tagged_row])
            row_segments.append(following_rows)
        else:
            row_segments.append(rows[tag_index:next_tag_index])

    segments = []
    for segment_rows in row_segments:
        source_kind = "tagged display"
        if segment_rows and not _has_equation_tag(segment_rows[0]):
            source_kind = "untagged continuation display"
        equation = _build_display_order_equation(
            table,
            segment_rows,
            f"{source_kind} in table {table.get('id', '<no-id>')}",
        )
        if equation:
            segments.append(equation)
    return segments


def _row_starts_new_expression(row):
    """Detect untagged rows that start a new display, not a continuation."""
    pieces = _math_pieces_from_row(row)
    if not pieces:
        return False
    first = pieces[0].strip()
    first = re.sub(r"^\\displaystyle\s*", "", first).strip()
    if not first:
        return False
    continuation_prefixes = (
        "=",
        "+",
        "-",
        r"\pm",
        r"\mp",
        r"\left",
        r"\right",
        r"\begin",
        r"\\",
    )
    return not first.startswith(continuation_prefixes)


def _math_pieces_from_row(row):
    pieces = []
    for math_elem in row.find_all("math"):
        annotation = math_elem.find("annotation", encoding="application/x-tex")
        if annotation:
            piece = annotation.get_text(strip=True)
        else:
            piece = math_elem.get("alttext", "").strip()
        if piece:
            pieces.append(piece)
    return pieces


def _build_display_order_equation(table, rows, source):
    math_text, math_source = _extract_math_from_rows(rows)
    if not math_text or not _looks_like_equation(math_text):
        return None
    identifiers = _identifiers_from_rows(rows)

    text_before = _get_text_before(table)
    text_after = _get_text_after(table)
    section = _get_section_title(table)
    audit_context = (
        f"{len(text_before)} chars before, {len(text_after)} chars after "
        f"(Section: '{section}')"
    )
    return {
        "eq_number": "",
        "equation": math_text,
        "equation_format": "latex",
        "equation_confidence": "high",
        "text_before": text_before,
        "text_after": text_after,
        "section": section,
        "audit_extract": "",
        "audit_context": audit_context,
        "audit_source": f"{source}; {math_source}",
        "identifier_candidates": identifiers,
        "audit_identifiers": (
            f"Found {len(identifiers)} unique identifiers from HTML <mi> tags: "
            f"{', '.join(identifiers) if identifiers else 'none'}"
        ),
        "_html_table_id": table.get("id", ""),
    }


def _table_for_equation(soup, equation):
    table_id = equation.get("_html_table_id")
    if not table_id:
        return None
    table = soup.find(id=table_id)
    if table and table.name == "table":
        return table
    return None


def _build_recovered_gap_equation(missing_number, table, location):
    rows = table.find_all("tr")
    math_text, math_source = _extract_math_from_rows(rows)
    if not math_text or not _looks_like_equation(math_text):
        return None
    identifiers = _identifiers_from_rows(rows)

    text_before = _get_text_before(table)
    text_after = _get_text_after(table)
    section = _get_section_title(table)
    math_preview = math_text[:80] + "..." if len(math_text) > 80 else math_text
    audit_extract = (
        f"{math_preview} (Equation ({missing_number}) recovered from "
        f"untagged HTML equation table {table.get('id', '<no-id>')} "
        f"{location}; source {math_source})"
    )
    audit_context = (
        f"{len(text_before)} chars before, {len(text_after)} chars after "
        f"(Section: '{section}')"
    )
    return {
        "eq_number": str(missing_number),
        "equation": math_text,
        "equation_format": "latex",
        "equation_confidence": "high",
        "text_before": text_before,
        "text_after": text_after,
        "section": section,
        "audit_extract": audit_extract,
        "audit_context": audit_context,
        "identifier_candidates": identifiers,
        "audit_identifiers": (
            f"Found {len(identifiers)} unique identifiers from HTML <mi> tags: "
            f"{', '.join(identifiers) if identifiers else 'none'}"
        ),
        "_html_table_id": table.get("id", ""),
    }


def _untagged_math_tables_before(following_table):
    candidates = []
    for table in following_table.find_all_previous("table"):
        if table.find("math") and not _has_equation_tag(table):
            candidates.append(table)
    return list(reversed(candidates))


def _untagged_math_tables_between(previous_table, following_table):
    candidates = []
    for table in previous_table.find_all_next("table"):
        if table is following_table:
            break
        if table.find("math") and not _has_equation_tag(table):
            candidates.append(table)
    return candidates


def _extract_math_from_rows(rows):
    """Collect all math annotations/alttext from equation table rows."""
    pieces = []
    for row in rows:
        for math_elem in row.find_all("math"):
            piece = ""
            annotation = math_elem.find("annotation", encoding="application/x-tex")
            if annotation:
                piece = annotation.get_text(strip=True)
            elif math_elem.get("alttext"):
                piece = math_elem["alttext"].strip()
            if piece:
                pieces.append(piece)

    if not pieces:
        return "", "none"

    math_text = " ".join(pieces)
    math_text = math_text.replace("%", "")
    math_text = re.sub(r"\\displaystyle\s*", "", math_text)
    math_text = re.sub(r"\s+", " ", math_text).strip()
    math_text = math_text.replace(" = ", "=").replace("= ", "=").replace(" =", "=")
    return math_text, "row_annotations"


def _identifiers_from_rows(rows):
    """Extract canonical identifiers from HTML MathML <mi> tags."""
    identifiers = []
    for row in rows:
        for mi in row.find_all("mi"):
            text = mi.get_text(strip=True)
            canonical = _canonical_identifier(text)
            if canonical and canonical not in identifiers:
                identifiers.append(canonical)
    return identifiers


def _canonical_identifier(text):
    mapping = {
        "ψ": "psi",
        "φ": "phi",
        "ϕ": "phi",
        "α": "alpha",
        "β": "beta",
        "γ": "gamma",
        "δ": "delta",
        "ε": "epsilon",
        "λ": "lambda",
        "μ": "mu",
        "ν": "nu",
        "ω": "omega",
        "ρ": "rho",
        "σ": "sigma",
        "θ": "theta",
        "π": "pi",
        "Ω": "Omega",
        "ℏ": "hbar",
    }
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    cleaned = mapping.get(cleaned, cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9]", "", cleaned)
    if not cleaned or cleaned in {"d", "i", "e"}:
        return ""
    return cleaned


def _looks_like_equation(math_text):
    """Reject tiny fragments accidentally produced by table-cell parsing."""
    if len(math_text) < 8:
        return False
    math_tokens = (
        "=",
        r"\sum",
        r"\frac",
        r"\int",
        r"\hat",
        r"\begin",
        r"\Psi",
        r"\psi",
        r"\mathbb",
        r"\boldsymbol",
        r"\ket",
        r"\bra",
        r"\left",
        r"\right",
        r"\pm",
        r"\to",
        r"\langle",
        r"\rangle",
        "^",
        "_",
    )
    return any(token in math_text for token in math_tokens)


def _get_text_before(container, max_chars=2000):
    """Get clean text from elements before the equation.

    Walks previous siblings collecting <p> text.
    """
    texts = []
    total = 0
    sibling = container.find_previous_sibling()
    while sibling and total < max_chars:
        # Only collect paragraph-like elements
        if sibling.name in ("p", "div"):
            text = sibling.get_text(separator=" ", strip=True)
            if text:
                texts.insert(0, text)
                total += len(text)
        sibling = sibling.find_previous_sibling()
    if total < max_chars // 3:
        for paragraph in container.find_all_previous("p", limit=6):
            text = paragraph.get_text(separator=" ", strip=True)
            if text and text not in texts:
                texts.insert(0, text)
                total += len(text)
            if total >= max_chars:
                break
    return " ".join(texts)[-max_chars:] if texts else ""


def _get_text_after(container, max_chars=2000):
    """Get clean text from elements after the equation.

    Walks next siblings collecting <p> text.
    """
    texts = []
    total = 0
    sibling = container.find_next_sibling()
    while sibling and total < max_chars:
        if sibling.name in ("p", "div"):
            text = sibling.get_text(separator=" ", strip=True)
            if text:
                texts.append(text)
                total += len(text)
        sibling = sibling.find_next_sibling()
    if total < max_chars // 3:
        for paragraph in container.find_all_next("p", limit=6):
            text = paragraph.get_text(separator=" ", strip=True)
            if text and text not in texts:
                texts.append(text)
                total += len(text)
            if total >= max_chars:
                break
    return " ".join(texts)[:max_chars] if texts else ""


_APPENDIX_RE = re.compile(
    r"\b(?:appendix|appendices|supplementary|supplemental|supporting\s+information"
    r"|supporting\s+material|supplement)\b",
    re.IGNORECASE,
)


def _prefer_main_body_equations(equations):
    """Filter appendix equations when the main body provides enough.

    Some papers restart equation numbering from (1) inside an appendix, so
    ``parse_html`` may collect a mixture of main-body and appendix equations
    that share the same numbers.  When the main body already yields
    MAX_EQUATIONS or more equations, discard the appendix ones entirely; the
    sort step has already placed main-body equations first by DOM order.

    Only removes appendix equations when the remaining main-body set is large
    enough to satisfy MAX_EQUATIONS, so papers with sparse main-body content
    still fall back to appendix equations.
    """
    main = [eq for eq in equations if not _APPENDIX_RE.search(eq.get("section", ""))]
    appendix = [eq for eq in equations if _APPENDIX_RE.search(eq.get("section", ""))]
    if not appendix:
        return equations  # nothing to filter
    if len(main) >= MAX_EQUATIONS:
        logger.info(
            "Dropped %d appendix equation(s); main body has %d",
            len(appendix), len(main),
        )
        return main
    return equations  # not enough main-body equations — keep everything


def _get_section_title(container):
    """Find the nearest section heading above this equation.

    Walks up the DOM looking for <section> parents with <h2>/<h3>/<h4>.
    """
    # Walk up to find a parent section
    parent = container.parent
    while parent:
        if parent.name == "section":
            heading = parent.find(["h2", "h3", "h4", "h5", "h6"])
            if heading:
                return heading.get_text(separator=" ", strip=True)
        parent = parent.parent

    # Fallback: find the closest preceding heading
    for heading_tag in ["h2", "h3", "h4"]:
        prev_heading = container.find_previous(heading_tag)
        if prev_heading:
            return prev_heading.get_text(separator=" ", strip=True)

    return "Unknown Section"
