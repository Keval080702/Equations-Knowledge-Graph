"""
Phase 2 — Equation Sequence Validation
========================================

Validation helpers for numbered equation sequences.

Papers usually number equations as ``1, 2, 3`` but some use labels such as
``2.1, 2.2`` or grouped sub-equation labels such as ``1a-d, 2a-b``.  The
project requirement is the **first numbered sequence** as stated in the paper,
so we preserve the original label while validating against its numeric
sequence.

Public API
----------
* ``has_first_equation_prefix(equations)``  — True if sequence starts at 1
* ``equation_sequence_numbers(equations)``  — canonical integer sequence
* ``normalize_equation_label(label)``       — normalise a raw label string
* ``equation_sequence_label(label)``        — reduce grouped label to base
* ``label_sort_key(label)``                 — sort key for mixed labels
* ``equation_numbers(equations)``           — list of eq_number values
* ``expected_prefix(equations)``            — expected first label string
"""

import re


GROUPED_LABEL_RE = re.compile(
    r"^(?P<base>\d+(?:\.\d+)*)(?P<start>[A-Za-z])(?:-(?P<end>[A-Za-z]))?$"
)

__all__ = [
    "equation_numbers",
    "equation_sequence_label",
    "equation_sequence_numbers",
    "expected_prefix",
    "has_first_equation_prefix",
    "is_grouped_equation_label",
    "label_sort_key",
    "normalize_equation_label",
    "validate_first_equation_prefix",
]


def equation_numbers(items):
    """Return original equation labels from a list or output dictionary."""
    if isinstance(items, dict):
        raw_numbers = items.keys()
    else:
        raw_numbers = [item.get("eq_number") for item in items]

    numbers = []
    for raw_number in raw_numbers:
        label = normalize_equation_label(raw_number)
        if not label:
            return []
        numbers.append(label)
    return numbers


def equation_sequence_numbers(items):
    """Return labels used to validate the first numeric equation sequence."""
    labels = equation_numbers(items)
    return [equation_sequence_label(label) for label in labels]


def normalize_equation_label(raw_number):
    """Normalize supported equation labels while preserving paper numbering."""
    if raw_number is None:
        return ""
    label = str(raw_number).strip().strip("()")
    label = label.replace("–", "-").replace("—", "-").replace("−", "-")
    label = re.sub(r"\s+", "", label)
    if not label:
        return ""
    return label


def equation_sequence_label(raw_number):
    """Return the numeric sequence label for a paper equation label.

    Examples
    --------
    ``1a-d`` -> ``1``
    ``2a-b`` -> ``2``
    ``2.1a-c`` -> ``2.1``
    ``2.1`` -> ``2.1``
    """
    label = normalize_equation_label(raw_number)
    match = GROUPED_LABEL_RE.fullmatch(label)
    if match:
        return match.group("base")
    return label


def is_grouped_equation_label(raw_number):
    """Return True for labels like ``1a-d`` or ``2.1a-c``."""
    label = normalize_equation_label(raw_number)
    match = GROUPED_LABEL_RE.fullmatch(label)
    return bool(match and match.group("start"))


def label_sort_key(label):
    """Natural sort key for integer, dotted, and grouped equation labels."""
    label = normalize_equation_label(label)
    sequence_label = equation_sequence_label(label)
    parts = sequence_label.split(".")
    if all(part.isdigit() for part in parts):
        suffix_match = GROUPED_LABEL_RE.fullmatch(label)
        suffix = suffix_match.group("start").lower() if suffix_match else ""
        return (0, tuple(int(part) for part in parts), suffix, label)
    return (1, label)


def expected_prefix(count, family=None):
    """Return the required first-equation prefix for ``count`` equations."""
    if family and "." in str(family):
        prefix = str(family).rsplit(".", 1)[0]
        return [f"{prefix}.{index}" for index in range(1, count + 1)]
    return [str(index) for index in range(1, count + 1)]


def has_first_equation_prefix(items):
    """Check that equations are the first sequence with no gaps."""
    labels = equation_sequence_numbers(items)
    if not labels:
        return False
    return labels == expected_prefix(len(labels), labels[0])


def validate_first_equation_prefix(items, arxiv_id):
    """Raise if a paper output is not the first numbered equation sequence."""
    raw_labels = equation_numbers(items)
    labels = [equation_sequence_label(label) for label in raw_labels]
    expected = expected_prefix(len(labels), labels[0] if labels else None)
    if labels != expected:
        raise ValueError(
            f"arXiv:{arxiv_id} equation numbering is not sequential: "
            f"got {raw_labels} (sequence labels {labels}), expected {expected}"
        )
