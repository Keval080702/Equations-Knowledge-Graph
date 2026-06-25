"""
Phase 6 — Pairwise Equation Relation Detection
================================================

Detects and grades pairwise semantic relations between equations extracted
from the same paper.

Relation grades (project vocabulary)
--------------------------------------
* ``"strong"``    — equations are directly linked (derivation, equivalence,
                    substitution, special case, same quantity).
* ``"potential"`` — weak co-occurrence evidence or shared symbol context.
* ``"none"``      — no detectable relationship.

Detection pipeline (applied in order; first matching evidence wins)
---------------------------------------------------------------------
1. **Explicit reference**  — equation number appears in the other equation's
   surrounding text (e.g. "from Eq. (2) we derive…").
2. **Derivation cue**      — transitional prose that explicitly connects two
   equations ("substituting (1) into (2)").
3. **Definition / usage**  — one equation defines a symbol that the other uses.
4. **Same-quantity rewrite** — both equations share the same LHS symbol.
5. **Parallel form**       — structural similarity (same operators, same
   symmetry pattern).
6. **Conservative context**— very weak shared-symbol signal; scored as
   ``"potential"`` only when no stronger evidence exists.

Public API
----------
* ``build_relations(equations)``
  Returns ``(relation_map, relation_audits)`` for all equation pairs.
"""

import itertools
import re

from src.enrichment_common import _clean_phrase
from src.enrichment_common import _context
from src.enrichment_common import _normalize_math_text
from src.enrichment_common import _reverse_description
from src.enrichment_common import _shorten
from src.sequence_validation import label_sort_key
from src.symbol_extraction import _normalize_symbol_key
from src.symbol_extraction import _symbols_from_equation


GENERIC_RELATION_SYMBOLS = {
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "alpha", "beta", "gamma", "delta", "epsilon", "eta", "theta",
    "kappa", "lambda", "mu", "nu", "omega", "phi", "pi", "rho",
    "sigma", "tau", "xi", "psi",
}

PROBABILITY_HEADS = {"p", "q", "P", "Q"}


def build_relations(equations):
    """Create graph-ready pairwise relation dictionaries for all equations."""
    relation_map = {eq["eq_number"]: {} for eq in equations}
    audits = {eq["eq_number"]: [] for eq in equations}

    for left, right in itertools.combinations(equations, 2):
        left_num = left["eq_number"]
        right_num = right["eq_number"]
        relation_lr = _classify_relation(left, right)
        relation_rl = _reverse_relation(relation_lr)
        relation_map[left_num][right_num] = relation_lr
        relation_map[right_num][left_num] = relation_rl
        audits[left_num].append(
            _relation_audit_text(right_num, relation_lr)
        )
        audits[right_num].append(
            _relation_audit_text(left_num, relation_rl)
        )

    return relation_map, {
        key: value[:6] if value else ["No pairwise equations"]
        for key, value in audits.items()
    }


def _relation_audit_text(other_eq_number, relation):
    """Format relation evidence for the audit trail."""
    return (
        f"Eq({other_eq_number}) grade={relation['grade']}; "
        f"method={relation['relation_type']}; "
        f"description='{relation['description']}'; "
        f"evidence='{relation['evidence']}'; "
        f"confidence={relation['confidence']}"
    )


def _classify_relation(left, right):
    left_symbols = set(left.get("symbols", {}))
    right_symbols = set(right.get("symbols", {}))
    shared_all = sorted(left_symbols & right_symbols)
    shared = _informative_shared_symbols(shared_all)
    left_context = _context(left)
    right_context = _context(right)

    explicit = _explicit_reference_relation(left, right, left_context, right_context)
    if explicit:
        return explicit

    local_text = _local_text_relation(left, right)
    if local_text:
        return local_text

    definition = _definition_usage_relation(left, right)
    if definition:
        return definition

    same_quantity = _same_quantity_relation(left, right, shared)
    if same_quantity:
        return same_quantity

    concept = _same_concept_relation(left, right, shared)
    if concept:
        return concept

    left_num = left["eq_number"]
    right_num = right["eq_number"]
    local = _same_section(left, right) or _labels_are_near(left_num, right_num, 1)

    # (1) A symbol DEFINED (topically) in BOTH equations is a genuine shared
    # concept — the strongest "potential" signal.  This bypasses the generic
    # single-letter/Greek filter, recovering links through H, L, S, rho, …
    # that carry a real meaning in both equations.
    shared_defined = _shared_defined_symbols(left, right, shared_all)
    if shared_defined:
        return _relation(
            "supporting_context",
            "potential",
            f"shares {'variables' if len(shared_defined) > 1 else 'variable'} {', '.join(shared_defined[:3])}",
            f"shared symbols: {', '.join(shared_defined[:8])}",
            0.5 if (len(shared_defined) >= 2 or local) else 0.4,
        )

    # (2) Meaningful (non-index) shared symbols — even if undefined — count when
    # there are several of them or the equations are local to each other.
    meaningful = _meaningful_shared_symbols(shared_all)
    if meaningful and (len(meaningful) >= 2 or local):
        return _relation(
            "supporting_context",
            "potential",
            f"shares {'variables' if len(meaningful) > 1 else 'variable'} {', '.join(meaningful[:3])}",
            f"shared symbols: {', '.join(meaningful[:8])}",
            0.4 if local else 0.3,
        )

    return _relation(
        "none",
        "none",
        "no shared symbols, references, or textual dependency evidence",
        "no relation evidence found",
        0.0,
    )


def _relation(relation_type, grade, description, evidence, confidence):
    return {
        "grade": grade,
        "description": description,
        "relation_type": relation_type,
        "evidence": evidence,
        "confidence": round(confidence, 2),
    }


def _explicit_reference_relation(left, right, left_context, right_context):
    left_num = left["eq_number"]
    right_num = right["eq_number"]

    left_mentions_right = _mention_side(left, right_num)
    right_mentions_left = _mention_side(right, left_num)
    if not left_mentions_right and not right_mentions_left:
        return None

    if left_mentions_right:
        context = _side_context(left, left_mentions_right) or left_context
        target = right_num
        direction = "dependency" if left_mentions_right == "before" else "source"
    else:
        context = _side_context(right, right_mentions_left) or right_context
        target = left_num
        direction = "source" if right_mentions_left == "before" else "dependency"

    window = _reference_window(context, target)

    # Check for directional substitution: "substituting Eq.(A) into Eq.(B)"
    subst = _substitution_direction(window, left_num, right_num)
    if subst:
        return subst

    semantic_desc = _semantic_reference_description(window)
    if _has_derivation_cue(window):
        if direction == "dependency":
            return _relation(
                "derivation_dependency",
                "strong",
                _dependency_description(semantic_desc),
                _shorten(window, 180),
                0.9,
            )
        return _relation(
            "used_by_derivation",
            "strong",
            _source_description(semantic_desc),
            _shorten(window, 180),
            0.9,
        )

    return _relation(
        "explicit_reference",
        "strong",
        semantic_desc or "explicitly referenced",
        _shorten(window, 180),
        0.75,
    )


def _local_text_relation(left, right):
    """Use adjacent sentence cues such as 'this gives' or 'hence ... becomes'.

    This catches common paper prose that links neighboring equations without
    writing an explicit equation number, while avoiding long-range guessing.
    """
    if not _labels_are_near(left["eq_number"], right["eq_number"], 1):
        return None
    if _label_precedes(left["eq_number"], right["eq_number"]):
        earlier, later = left, right
        left_to_right = True
    else:
        earlier, later = right, left
        left_to_right = False

    bridge = " ".join(
        part for part in (
            str(earlier.get("text_after", ""))[:500],
            str(later.get("text_before", ""))[-500:],
        )
        if part
    )
    if not bridge:
        return None
    if _has_local_equivalence_cue(bridge):
        return _relation(
            "same_quantity_or_rewrite",
            "strong",
            "equivalent formulation",
            _shorten(bridge, 180),
            0.82,
        )
    if _has_local_derivation_anaphora(bridge):
        relation_type = "used_by_derivation" if left_to_right else "derivation_dependency"
        description = "used in derivation" if left_to_right else "derived from"
        return _relation(
            relation_type,
            "strong",
            description,
            _shorten(bridge, 180),
            0.82,
        )
    return None


def _mention_side(equation, target_num):
    """Return whether target_num is mentioned before or after this equation."""
    before = equation.get("text_before", "")
    after = equation.get("text_after", "")
    if _mentions_equation(before, target_num):
        return "before"
    if _mentions_equation(after, target_num):
        return "after"
    context = _context(equation)
    if _mentions_equation(context, target_num):
        return "context"
    return ""


def _side_context(equation, side):
    if side == "before":
        return equation.get("text_before", "")
    if side == "after":
        return equation.get("text_after", "")
    return _context(equation)


def _substitution_direction(window, left_num, right_num):
    """Detect 'substituting Eq.(A) into Eq.(B)' and return a directed relation.

    Pattern: subst-verb ... Eq.(source) ... into ... Eq.(dest)
    or equivalently: into Eq.(dest) ... from Eq.(source).
    Returns a relation dict where the description reflects the direction,
    or None if the pattern is not found.
    """
    if not window:
        return None
    lower = window.lower()
    subst_re = re.compile(
        r"\b(?:substitut\w+|inserting|plug\w*)\b",
        re.IGNORECASE,
    )
    into_re = re.compile(r"\binto\b", re.IGNORECASE)
    if not subst_re.search(lower) or not into_re.search(lower):
        return None

    left_label = _label_pattern(left_num)
    right_label = _label_pattern(right_num)
    # "substituting Eq.(left) into Eq.(right)" → left is source, right is target
    pat_lr = re.compile(
        rf"\b(?:substitut\w+|inserting|plug\w*)\b[^.;]{{0,60}}"
        rf"{left_label}[^.;]{{0,40}}\binto\b[^.;]{{0,60}}{right_label}",
        re.IGNORECASE,
    )
    # "substituting Eq.(right) into Eq.(left)" → right is source, left is target
    pat_rl = re.compile(
        rf"\b(?:substitut\w+|inserting|plug\w*)\b[^.;]{{0,60}}"
        rf"{right_label}[^.;]{{0,40}}\binto\b[^.;]{{0,60}}{left_label}",
        re.IGNORECASE,
    )
    if pat_lr.search(window):
        return _relation(
            "substitution",
            "strong",
            "substituted into",
            _shorten(window, 180),
            0.92,
        )
    if pat_rl.search(window):
        return _relation(
            "substitution_target",
            "strong",
            "receives substitution from",
            _shorten(window, 180),
            0.92,
        )
    return None


def _definition_usage_relation(left, right):
    left_defined = _defined_lhs_symbols(left)
    right_defined = _defined_lhs_symbols(right)
    left_symbols = set(left.get("symbols", {}))
    right_symbols = set(right.get("symbols", {}))

    left_feeds_right = sorted(
        symbol for symbol in left_defined & right_symbols
        if _definition_symbol_can_link(symbol)
    )
    right_feeds_left = sorted(
        symbol for symbol in right_defined & left_symbols
        if _definition_symbol_can_link(symbol)
    )
    if left_feeds_right and not right_feeds_left:
        syms = ", ".join(left_feeds_right[:3])
        local = _definition_relation_is_local(left, right, left_feeds_right)
        return _relation(
            "definition_usage",
            "strong" if local else "potential",
            f"defines {syms}",
            (
                f"Eq({left['eq_number']}) LHS symbols {syms} appear in "
                f"Eq({right['eq_number']}); local_support={local}"
            ),
            0.8 if local else 0.45,
        )
    if right_feeds_left and not left_feeds_right:
        syms = ", ".join(right_feeds_left[:3])
        local = _definition_relation_is_local(right, left, right_feeds_left)
        return _relation(
            "uses_definition",
            "strong" if local else "potential",
            f"uses definition of {syms}",
            (
                f"Eq({right['eq_number']}) LHS symbols {syms} appear in "
                f"Eq({left['eq_number']}); local_support={local}"
            ),
            0.8 if local else 0.45,
        )
    return None


def _same_quantity_relation(left, right, shared):
    left_lhs = _defined_lhs_symbols(left)
    right_lhs = _defined_lhs_symbols(right)
    common_lhs = sorted(left_lhs & right_lhs)
    if common_lhs:
        sim = _equation_similarity(left["equation"], right["equation"])
        local = _same_section(left, right) or _labels_are_near(left["eq_number"], right["eq_number"], 2)
        can_be_strong = (
            sim >= 0.55
            and local
            and any(_definition_symbol_can_link(symbol) for symbol in common_lhs)
        )
        grade = "strong" if can_be_strong else "potential"
        desc = "equivalent formulation" if can_be_strong else "same defined quantity"
        return _relation(
            "same_quantity_or_rewrite",
            grade,
            desc,
            f"both equations define/transform {', '.join(common_lhs[:3])}",
            0.78 if grade == "strong" else 0.52,
        )
    if (
        left.get("meaning") == right.get("meaning")
        and len(shared) >= 2
        and (_same_section(left, right) or _labels_are_near(left["eq_number"], right["eq_number"], 1))
    ):
        return _relation(
            "parallel_form",
            "potential",
            "parallel formulation",
            f"same meaning label and shared symbols: {', '.join(shared[:6])}",
            0.55,
        )
    return None


def _same_concept_relation(left, right, shared):
    left_meaning = left.get("meaning", "")
    right_meaning = right.get("meaning", "")
    if not left_meaning or left_meaning != right_meaning:
        return None
    if left_meaning in {"mathematical relation", "enumerated mathematical expression"}:
        return None
    if shared and (_same_section(left, right) or _labels_are_near(left["eq_number"], right["eq_number"], 1)):
        return _relation(
            "same_formula_concept",
            "potential",
            "same concept",
            f"shared symbols: {', '.join(shared[:6])}",
            0.5,
        )
    return None


def _mentions_equation(context, eq_number):
    if not context or not eq_number:
        return False
    label = _label_pattern(eq_number)
    # ar5iv renders "(5)" as "( 5 )" — allow arbitrary whitespace inside parens.
    # Also handle \xa0 (non-breaking space) and   (thin space) between
    # the keyword and the opening paren, since \s covers both.
    # (?!\d) prevents "5" from matching inside "52"
    # _num matches the target with no trailing digit
    _num = label
    _any = _any_label_pattern()
    _kw  = r"\b(?:eqs?\.?|equations?|relations?)"
    patterns = [
        # Direct: "Eq. ( 5 )" / "Eq.( 5 )" / "Eq. 5"
        rf"{_kw}\s*{_num}",
        # Range lower bound: "Eqs. ( 3 )-( 5 )" matches 5
        rf"{_kw}\s*{_any}\s*[-–]\s*{_num}",
        # Range upper bound: "Eqs. ( 5 )-( 7 )" matches 5
        rf"{_kw}\s*{_num}\s*[-–]\s*{_any}",
        # List form: "Eqs. ( 21 ) and ( 23 )" or "Eqs. ( 3 ), ( 5 )"
        # keyword + any number of preceding terms + target
        rf"{_kw}\s*{_any}(?:\s*(?:,|and)\s*{_any}){{0,5}}\s*(?:,|and)\s*{_num}",
        # Range + extra: "Eqs. ( 15 )-( 17 ) and ( 28 )"
        rf"{_kw}\s*{_any}\s*[-–]\s*{_any}(?:\s*(?:,|and)\s*{_any}){{0,4}}\s*(?:,|and)\s*{_num}",
    ]
    if any(re.search(p, context, re.IGNORECASE) for p in patterns):
        return True
    # Check if target falls inside a numeric range reference "Eqs. ( A )-( B )"
    return _in_range_reference(context, eq_number)


def _in_range_reference(context, eq_number):
    """Return True if eq_number falls within a range like 'Eqs. ( 3 )-( 7 )'."""
    try:
        target = float(eq_number)
    except (TypeError, ValueError):
        return False
    range_pat = re.compile(
        r"\b(?:eqs?\.?|equations?)\s*\(?\s*(\d+(?:\.\d+)?)\s*\)?\s*[-–]\s*\(?\s*(\d+(?:\.\d+)?)\s*\)?",
        re.IGNORECASE,
    )
    for m in range_pat.finditer(context):
        try:
            lo, hi = float(m.group(1)), float(m.group(2))
        except ValueError:
            continue
        if lo <= target <= hi:
            return True
    return False


def _reference_window(context, eq_number):
    if not context:
        return ""
    candidates = [
        (index, sentence)
        for index, sentence in enumerate(_reference_sentences(context))
        if _mentions_equation(sentence, eq_number)
    ]
    if not candidates:
        return context[:220]
    scored = sorted(
        ((_reference_sentence_score(sentence), index, sentence) for index, sentence in candidates),
        reverse=True,
    )
    _, best_index, best_sentence = scored[0]
    sentences = _reference_sentences(context)
    window = best_sentence
    if _starts_with_anaphora(best_sentence) and best_index > 0:
        window = f"{sentences[best_index - 1]} {window}"
    if len(window) < 140 and best_index + 1 < len(sentences) and _has_relation_cue(sentences[best_index + 1]):
        window = f"{window} {sentences[best_index + 1]}"
    return window[:360]


def _reference_sentences(context):
    text = re.sub(r"\s+", " ", str(context or "")).strip()
    if not text:
        return []
    protected = (
        text.replace("Eq.", "Eq§")
        .replace("Eqs.", "Eqs§")
        .replace("Fig.", "Fig§")
        .replace("Sec.", "Sec§")
    )
    parts = re.split(r"(?<=[.!?;])\s+|\n+", protected)
    sentences = []
    for part in parts:
        restored = (
            part.replace("Eq§", "Eq.")
            .replace("Eqs§", "Eqs.")
            .replace("Fig§", "Fig.")
            .replace("Sec§", "Sec.")
        ).strip()
        if restored:
            sentences.append(restored)
    return sentences or [text]


def _reference_sentence_score(sentence):
    lower = sentence.lower()
    score = 1.0
    if _has_derivation_cue(sentence):
        score += 4.0
    if _semantic_reference_description(sentence):
        score += 3.0
    if re.search(r"\b(?:where|hence|therefore|thus|consequently|substitut\w+|using)\b", lower):
        score += 1.5
    if re.search(r"\b(?:fig(?:ure)?|table|appendix|section|supplementary)\b", lower):
        score -= 2.0
    if len(sentence) > 420:
        score -= 1.0
    return score


def _has_relation_cue(text):
    return bool(_semantic_reference_description(text) or _has_derivation_cue(text))


def _starts_with_anaphora(text):
    return bool(re.match(r"\s*(?:this|these|such|hence|therefore|thus|consequently)\b", text, re.IGNORECASE))


def _has_local_derivation_anaphora(text):
    lower = text.lower()
    patterns = [
        r"\b(?:this|these|the above|the previous|such)\s+(?:equations?|relations?|expressions?|forms?)?\b.{0,100}\b(?:gives?|yields?|leads?\s+to|results?\s+in|reduces?\s+to|simplif\w+|becomes?|become|takes?\s+the\s+form|can\s+be\s+written|is\s+written|are\s+written)\b",
        r"\b(?:hence|therefore|thus|consequently)\b.{0,100}\b(?:we\s+)?(?:obtain|get|find|have|arrive|derive)\b",
        r"\b(?:substitut\w+|inserting|plug\w*)\b.{0,120}\b(?:into|gives?|yields?|leads?\s+to|results?\s+in)\b",
        r"\b(?:using|from)\s+(?:this|these|the above|the previous)\b.{0,100}\b(?:obtain|get|derive|find|write)\b",
    ]
    return any(re.search(pattern, lower, re.IGNORECASE) for pattern in patterns)


def _has_local_equivalence_cue(text):
    lower = text.lower()
    patterns = [
        r"\b(?:equivalent|identical)\s+(?:to|form)\b",
        r"\b(?:rewritten|written|expressed|recast)\s+as\b",
        r"\b(?:same|equivalent)\s+(?:equation|expression|formulation|form)\b",
    ]
    return any(re.search(pattern, lower, re.IGNORECASE) for pattern in patterns)


def _semantic_reference_description(text):
    """Extract a short semantic description from the equation reference window.

    Searches for physics-relation cue phrases near the equation reference and
    returns a concise label. Returns empty string if no recognised cue is found.
    Ordered by specificity — first match wins.
    """
    lower = text.lower()
    cue_map = [
        ("special case", ["special case", "particular case", "limiting case"]),
        ("generalization", ["generalization", "general case", "more general form"]),
        ("reduces to", ["reduces to", "simplifies to", "collapses to"]),
        ("equivalent", ["equivalent to", "identical to", "same as", "is equivalent"]),
        ("negation", ["negation of", "opposite of", "complementary to"]),
        ("derived from", [
            "derived from", "follows from", "can be derived", "obtained from",
            "are obtained", "is obtained", "we obtain", "hence", "therefore",
            "becomes", "become",
        ]),
        ("substituted into", ["substituting", "by substitution", "inserting"]),
        ("generalized by", ["generalized by", "is extended", "extension of"]),
        ("approximation", ["approximation", "approximate form", "to leading order"]),
        ("combined with", ["combining", "combined with", "adding together"]),
        ("used in derivation", ["using eq", "using equation", "used to derive"]),
        ("see also", ["see eq", "see equation", "see relation"]),
        ("defined by", ["defined in eq", "defined by eq", "as defined"]),
        ("proportional to", ["proportional to"]),
        ("consistent with", ["consistent with"]),
    ]
    for label, cues in cue_map:
        if any(cue in lower for cue in cues):
            return label
    return ""


def _has_derivation_cue(text):
    lower = text.lower()
    return any(cue in lower for cue in (
        "using", "substituting", "substitution", "inserting", "from",
        "follows from", "follow from", "leads to", "lead to", "we obtain",
        "we get", "which gives", "which yields", "yields", "derived",
        "defined by", "given by", "reduce", "simplif", "therefore", "thus",
        "hence", "become", "obtained", "expressed", "written",
        "transformation", "result allows", "takes the form",
    ))


def _defined_lhs_symbols(equation):
    text = equation.get("equation", "")
    if "=" not in text:
        return set()
    lhs = text.split("=", 1)[0]
    if _lhs_is_complex_expression(lhs):
        return set()
    atoms = _symbols_from_equation(lhs)
    if not atoms:
        return set()
    symbol = _normalize_symbol_key(atoms[0])
    if _defined_symbol_is_safe(symbol, lhs, atoms[0]):
        return {symbol}
    return set()


def _lhs_is_complex_expression(lhs):
    compact = lhs.strip()
    if not compact:
        return True
    if any(token in compact for token in ("[", "]", "+", r"\frac", r"\sum", r"\int")):
        return True
    if "-" in compact and not re.search(r"\^\{?[-+]", compact):
        return True
    if len(_symbols_from_equation(compact)) > 5:
        return True
    return False


def _defined_symbol_is_safe(symbol, lhs, raw_atom):
    if not symbol:
        return False
    if _is_informative_relation_symbol(symbol):
        return True
    if symbol in PROBABILITY_HEADS:
        return False
    return _lhs_has_function_call(lhs, raw_atom)


def _definition_symbol_can_link(symbol):
    return _is_informative_relation_symbol(symbol) or symbol not in GENERIC_RELATION_SYMBOLS


def _definition_relation_is_local(definition_eq, usage_eq, symbols):
    if _labels_are_near(definition_eq["eq_number"], usage_eq["eq_number"], 2):
        return True
    if not _same_section(definition_eq, usage_eq):
        return False
    context = " ".join(
        str(part or "")
        for part in (
            definition_eq.get("text_after", ""),
            usage_eq.get("text_before", ""),
            usage_eq.get("text_after", ""),
        )
    )
    return any(_symbol_mentioned_in_text(symbol, context) for symbol in symbols)


def _symbol_mentioned_in_text(symbol, text):
    if not symbol or not text:
        return False
    compact_symbol = re.sub(r"[^A-Za-z0-9]+", "", str(symbol)).lower()
    compact_text = re.sub(r"[^A-Za-z0-9]+", "", _normalize_math_text(text)).lower()
    return bool(compact_symbol and compact_symbol in compact_text)


def _informative_shared_symbols(symbols):
    return sorted(symbol for symbol in symbols if _is_informative_relation_symbol(symbol))


# Pure index / coordinate letters: a bare one of these is a dummy variable, not a
# relation-bearing quantity.  (A subscripted/superscripted form is kept.)
_INDEX_LETTERS = {"i", "j", "k", "l", "m", "n", "t", "x", "y", "z"}

# Definitions that do NOT indicate a topical relationship: every equation has
# indices / a time variable, so sharing one of these is not evidence of a link.
_NONTOPICAL_DEFS = {
    "summation index", "differential variable", "Lorentz index",
    "time", "imaginary unit",
}


def _shared_defined_symbols(left, right, shared_all):
    """Shared symbols that carry a real (topical) definition in BOTH equations."""
    lsym = left.get("symbols", {})
    rsym = right.get("symbols", {})
    out = []
    for s in shared_all:
        ld = str(lsym.get(s, "")).strip()
        rd = str(rsym.get(s, "")).strip()
        if ld and rd and ld not in _NONTOPICAL_DEFS and rd not in _NONTOPICAL_DEFS:
            out.append(s)
    return sorted(out)


def _meaningful_shared_symbols(shared_all):
    """Shared symbols that are not bare index/coordinate letters."""
    out = []
    for s in shared_all:
        base = re.split(r"[_^]", s)[0]
        bare = "_" not in s and "^" not in s
        if bare and base.lower() in _INDEX_LETTERS:
            continue
        out.append(s)
    return sorted(out)


def _is_informative_relation_symbol(symbol):
    symbol = str(symbol or "").strip()
    if not symbol or symbol in GENERIC_RELATION_SYMBOLS:
        return False
    if symbol in PROBABILITY_HEADS:
        return False
    if len(symbol) == 1:
        return False
    return ("_" in symbol or "^" in symbol or len(symbol) > 2)


def _lhs_has_function_call(lhs, raw_atom):
    if not lhs or not raw_atom:
        return False
    pattern = re.escape(str(raw_atom).strip())
    return bool(re.search(rf"{pattern}\s*\(", lhs))


def _label_precedes(left_label, right_label):
    return label_sort_key(left_label) <= label_sort_key(right_label)


def _labels_are_near(left_label, right_label, max_gap):
    left_tuple = _label_numeric_tuple(left_label)
    right_tuple = _label_numeric_tuple(right_label)
    if not left_tuple or not right_tuple:
        return False
    if len(left_tuple) != len(right_tuple):
        return False
    if left_tuple[:-1] != right_tuple[:-1]:
        return False
    return abs(left_tuple[-1] - right_tuple[-1]) <= max_gap


def _label_numeric_tuple(label):
    key = label_sort_key(label)
    if not key or key[0] != 0:
        return ()
    return tuple(key[1])


def _label_pattern(eq_number):
    label = str(eq_number or "").strip()
    label = label.replace("–", "-").replace("—", "-")
    grouped = re.fullmatch(r"(\d+(?:\.\d+)?)([A-Za-z])\s*-\s*([A-Za-z])", label)
    if grouped:
        base, start, end = grouped.groups()
        body = rf"{re.escape(base)}\s*{re.escape(start)}\s*[-–]\s*{re.escape(end)}"
    else:
        lettered = re.fullmatch(r"(\d+(?:\.\d+)?)([A-Za-z])", label)
        if lettered:
            base, suffix = lettered.groups()
            body = rf"{re.escape(base)}\s*{re.escape(suffix)}"
        else:
            body = re.escape(label)
    return rf"\(?\s*{body}(?![\w.])\s*\)?"


def _any_label_pattern():
    return r"\(?\s*\d+(?:\.\d+)?(?:\s*[A-Za-z](?:\s*[-–]\s*[A-Za-z])?)?\s*\)?"


def _dependency_description(semantic_desc):
    if semantic_desc in {"", "used in derivation", "see also"}:
        return "derived from"
    return semantic_desc


def _source_description(semantic_desc):
    if semantic_desc in {"", "derived from", "defined by"}:
        return "used in derivation"
    return semantic_desc


def _equation_similarity(left_text, right_text):
    left_tokens = set(_relation_tokens(left_text))
    right_tokens = set(_relation_tokens(right_text))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _relation_tokens(text):
    normalized = _normalize_math_text(text).lower()
    normalized = re.sub(r"\\[a-z]+", " ", normalized)
    return re.findall(r"[a-z][a-z0-9]*|[0-9]+|[=+\-*/]", normalized)


def _same_section(left, right):
    return (
        left.get("section")
        and left.get("section") == right.get("section")
        and left.get("section") != "Unknown Section"
    )


def _reverse_relation(relation):
    relation_type = relation.get("relation_type", "unknown")
    reversed_type = {
        "definition_usage": "uses_definition",
        "uses_definition": "definition_usage",
        "used_by_derivation": "derivation_dependency",
        "derivation_dependency": "used_by_derivation",
        "substitution": "substitution_target",
        "substitution_target": "substitution",
    }.get(relation_type, relation_type)
    description = _reverse_description(relation["description"])
    if relation_type == "derivation_dependency":
        description = "used in derivation"
    elif relation_type == "used_by_derivation":
        description = "derived from"
    elif relation_type == "substitution":
        description = "receives substitution from"
    elif relation_type == "substitution_target":
        description = "substituted into"
    if relation_type == "definition_usage":
        # Forward: "defines X" → reverse: "uses definition of X"
        match = re.search(r"defines (.+?)$", relation["description"])
        if match:
            description = f"uses definition of {match.group(1)}"
    elif relation_type == "uses_definition":
        # Forward: "uses definition of X" → reverse: "defines X"
        match = re.search(r"uses definition of (.+?)$", relation["description"])
        if match:
            description = f"defines {match.group(1)}"
    return {
        "grade": relation["grade"],
        "description": description,
        "relation_type": reversed_type,
        "evidence": relation.get("evidence", ""),
        "confidence": relation.get("confidence", 0.0),
    }


__all__ = ["build_relations"]
