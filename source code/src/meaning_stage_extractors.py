"""Equation-meaning extraction stages 0–4 and phrase utilities.

Implements the five evidence-ordered extraction stages used by
``meaning_extraction.py``.  All logic is extractive (spaCy dependency
parser only — no text generation).

Stages
------
0  Local cue — production/copular verb in the sentence immediately before
   the equation labels the meaning with highest precision.
1  Definiendum — subject/object of a definition cue in ranked sentences.
2  Named term — proper-noun compound ("Dirac equation", "CHSH inequality").
3  LHS symbol — definition of the equation's principal left-hand symbol.
4  Role name — structural equation role (commutation relation, ODE, …).
"""

import re
from dataclasses import dataclass, field


# ── Lazy spaCy proxy (avoids circular import with meaning_extraction) ─────────

def _parse(text):
    """Parse a sentence via the shared cached parser in meaning_extraction."""
    from src.meaning_extraction import _parse as _cached_parse  # lazy import
    return _cached_parse(text)


# ── Text preprocessing ────────────────────────────────────────────────────────

_MATH_NOISE_RE = re.compile(
    r"\\[A-Za-z]+(?:_\{[^}]*\}|\^\{[^}]*\})*"
    r"|\b(?:start_|end_)?(?:POST(?:SUB|SUPER)SCRIPT|ARG)\b"
    r"|\b(?:subscript|superscript)\b"
    r"|\b(?:italic|bold|roman|cali?graphic|blackboard|fraktur|sans_serif|"
    r"typewriter|script|monospace|bold_italic)_\S+",
    re.IGNORECASE,
)
_MATH_WORDS_RE = re.compile(
    r"\b(?:ket|bra|langle|rangle|otimes|oplus|ominus|odot|dagger|"
    r"mathrm|mathcal|mathbb|mathbf|mathfrak|mathsf|hat|vec|tilde|overline|"
    r"underline|overrightarrow|boldsymbol)\b",
    re.IGNORECASE,
)


def _scrub(text):
    """Remove ar5iv math-conversion artifacts so prose stays parseable."""
    text = _MATH_NOISE_RE.sub(" ", str(text or ""))
    text = _MATH_WORDS_RE.sub(" ", text)
    text = re.sub(
        r"\b[A-Z]\S{1,25}(?:\s+\S{1,15})?\s+et[\s\xa0]+al\.?\s*\(\s*\d{4}[^)]*\)",
        " ", text,
    )
    text = re.sub(r"\b[A-Z][A-Za-zÀ-ɏ]{2,20}\s*\(\s*\d{4}[^)]{0,10}\)", " ", text)
    text = re.sub(r"[^\x00-\x7f]+", " ", text)
    text = re.sub(r"\[[^\]]{0,30}\]", " ", text)
    text = re.sub(r"[{}_^\\]", " ", text)
    text = re.sub(r"\b(?:[A-Za-z]\s+){2,}[A-Za-z]\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _sentences(text):
    """Scrub then split into sentences."""
    text = _scrub(text)
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?;:])\s+", text) if s.strip()]


# ── Candidate model ───────────────────────────────────────────────────────────

TIER_DEFINIENDUM = 1
TIER_NAMED = 2
TIER_LHS_SYMBOL = 3
TIER_ROLE = 4


@dataclass
class Candidate:
    phrase: str
    stage: str
    tier: int
    distance: int
    evidence: str
    score: float = field(default=0.0)
    embedding_score: float | None = field(default=None)
    embedding_selected: bool = field(default=False)


@dataclass
class LocalSentence:
    text: str
    side: str
    distance: int
    rank: float


# ── Stage 0: locality-first cue ───────────────────────────────────────────────

_SUBJECT_CUE_LEMMAS = {
    "define", "give", "write", "express", "represent", "derive", "obtain",
    "read", "formulate", "compute", "utilize", "model", "calculate",
    "introduce", "characterize", "approximate", "cast", "recast", "summarize",
    "encode", "capture", "govern", "describe", "set", "establish", "choose",
}
_PRODUCTION_CUE_LEMMAS = {
    "give", "make", "construct", "yield", "produce", "introduce", "consider",
    "take", "denote", "call", "define", "rewrite", "obtain", "simplify",
    "propose", "use", "apply", "write", "describe", "have",
}
_MEANING_PREPS = {"by", "under", "via", "to"}
_PRONOUNS = {"it", "they", "this", "these", "that", "which", "he", "she"}
_SELF_PRONOUNS = {"we", "one", "i", "us", "author", "authors"}


def _stage_local_cue(before, after):
    """Cue-typed meaning from the sentence that introduces the equation."""
    if not before:
        return []
    last = before[-1]
    prev = before[-2] if len(before) >= 2 else ""
    candidates = _local_cue_candidates(last, prev)
    if not candidates:
        return []
    _pos, phrase, evidence = max(candidates, key=lambda c: c[0])
    if not phrase:
        return []
    return [Candidate(phrase, "local cue", TIER_DEFINIENDUM, 0, evidence)]


def _local_cue_candidates(sentence, prev):
    doc = _parse(sentence)
    if doc is None or not doc.has_annotation("DEP"):
        return []
    out = []
    n = len(doc)
    for tok in doc:
        lemma = tok.lemma_.lower()
        if lemma in _SUBJECT_CUE_LEMMAS and _cue_takes_subject(tok):
            phrase = _subject_phrase(tok, doc, prev)
            if phrase:
                out.append((tok.i, phrase, f"subject of '{lemma}' cue"))
        elif lemma == "be" and tok.i >= n - 4:
            phrase = _subject_phrase(tok, doc, prev)
            if phrase:
                out.append((tok.i, phrase, "subject of sentence-final copula"))
        if lemma in _PRODUCTION_CUE_LEMMAS and tok.dep_ not in ("acl", "amod", "relcl", "advcl", "part"):
            obj = _direct_object(tok)
            if obj is not None and obj.i > tok.i:
                phrase = _meaning_np(obj)
                if phrase:
                    out.append((obj.i, phrase, f"object of '{lemma}' cue"))
    out += _trailing_prep_objects(doc)
    return out


def _cue_takes_subject(cue):
    window = cue.doc[cue.i + 1 : cue.i + 4]
    if any(t.lower_ == "as" for t in window):
        return True
    tail = [t for t in cue.doc[cue.i + 1 :] if t.is_alpha]
    return not tail and not any(t.lower_ == "by" for t in cue.doc[cue.i + 1 : cue.i + 3])


def _subject_phrase(cue, doc, prev):
    subj = next((c for c in cue.children if c.dep_ in ("nsubj", "nsubjpass")), None)
    if subj is None and cue.dep_ in ("acl", "relcl", "advcl"):
        subj = cue.head
    if subj is None:
        return ""
    low = subj.lower_
    if low in _SELF_PRONOUNS:
        return ""
    if low in _PRONOUNS or subj.pos_ == "PRON":
        return _resolve_pronoun(subj, doc, prev)
    return _meaning_np(subj)


def _meaning_np(token):
    span = _noun_span(token)
    if span:
        return span
    if token.pos_ == "ADJ" and any(c.dep_ == "det" for c in token.children):
        mods = [t.text for t in token.lefts if t.dep_ in ("amod", "compound")]
        return _clean(" ".join(mods + [token.text]))
    return ""


def _resolve_pronoun(subj, doc, prev):
    if subj.lower_ in ("which", "that"):
        for tok in reversed(doc[: subj.i]):
            if tok.pos_ in ("NOUN", "PROPN"):
                return _noun_span(tok)
    return _last_noun_phrase(prev)


def _direct_object(verb):
    return next((c for c in verb.children if c.dep_ in ("dobj", "obj", "attr")), None)


def _trailing_prep_objects(doc):
    out = []
    n = len(doc)
    for tok in doc:
        if tok.pos_ != "ADP" or tok.lower_ not in _MEANING_PREPS:
            continue
        if tok.i < n - 8:
            continue
        pobj = next((c for c in tok.children if c.dep_ == "pobj"), None)
        if pobj is not None and pobj.pos_ in ("NOUN", "PROPN"):
            phrase = _noun_span(pobj)
            if phrase:
                out.append((tok.i, phrase, f"object of '{tok.lower_}'"))
    return out


def _last_noun_phrase(text):
    doc = _parse(text)
    if doc is None:
        return ""
    chunks = [c for c in doc.noun_chunks if c.root.pos_ in ("NOUN", "PROPN")]
    if not chunks:
        return ""
    return _noun_span(chunks[-1].root)


# ── Stage 1: definiendum ──────────────────────────────────────────────────────

_DEFINITOR_LEMMAS = {
    "define", "give", "write", "express", "denote", "read", "represent",
    "derive", "obtain", "compute", "calculate", "introduce",
    "call", "term", "dub", "name", "label", "know", "relate", "connect",
}
_NAMING_LEMMAS = {"call", "term", "dub", "name", "label", "denote", "know"}
_STRONG_DEFINITORS = {"define", "call", "term", "dub", "denote", "name", "label"}
_PRODUCTION_LEMMAS = {"give", "make", "yield", "produce", "read", "take", "have"}
_RELATIVE_PRONOUNS = {"which", "that", "it", "this", "they", "these"}
_SELF_REFERENCE = {"we", "one", "i", "author", "authors"}
_DEFINITION_SENTENCE_CUE_RE = re.compile(
    r"\b(?:defined as|given by|written as|expressed as|called|known as|"
    r"denoted by|we define|we introduce|takes the form|is described by|"
    r"are described by|is given by|are given by|reads?|can be written|"
    r"can be expressed)\b"
    r"|\bwhere\b.{0,90}\bdenotes?\b"
    r"|:\s*$",
    re.IGNORECASE,
)
_NEGATIVE_CONTEXT_RE = re.compile(
    r"\b(?:figure|fig\.|table|tab\.|appendix|app\.|copyright|license|"
    r"email|affiliation|doi|arxiv|section|sect\.)\b",
    re.IGNORECASE,
)
_PHRASE_WORD = r"[A-Za-z][A-Za-z0-9'\-\/]*"
_PHRASE = rf"{_PHRASE_WORD}(?:\s+{_PHRASE_WORD}){{0,5}}"
_REGEX_CUE_PATTERNS = [
    (
        re.compile(
            rf"\b(?:the|a|an|this|these|those|our)\s+"
            rf"(?P<phrase>{_PHRASE})\s+"
            rf"(?:is|are|can be|may be|is then|are then)?\s*"
            rf"(?:given by|written as|defined as|expressed as|described by|"
            rf"denoted by|called|known as|takes the form)\b",
            re.IGNORECASE,
        ),
        "article noun phrase before definition cue",
    ),
    (
        re.compile(
            rf"\b(?:the|a|an|this|these|those|our)\s+"
            rf"(?P<phrase>{_PHRASE})\s+(?:is|are)\s*:?\s*$",
            re.IGNORECASE,
        ),
        "article noun phrase before colon",
    ),
    (
        re.compile(
            rf"\b(?:this|that|it|which)\s+"
            rf"(?:gives|yields|produces|defines|describes|introduces)\s+"
            rf"(?:the|a|an)?\s*(?P<phrase>{_PHRASE})\s*:?\s*$",
            re.IGNORECASE,
        ),
        "pronoun production cue",
    ),
    (
        re.compile(
            rf"\bwe\s+(?:define|introduce|call|denote|write|express)\s+"
            rf"(?:the|a|an)?\s*(?P<phrase>{_PHRASE})\b",
            re.IGNORECASE,
        ),
        "first-person definition cue",
    ),
    (
        re.compile(
            rf"(?:the|a|an)\s+following\s+(?P<phrase>{_PHRASE_WORD}(?:\s+{_PHRASE_WORD}){{0,2}})\b",
            re.IGNORECASE,
        ),
        "the following [equation name] introducer",
    ),
]


def _ranked_definition_contexts(before, after, limit=5):
    """Rank local sentences by definition/name evidence before extraction."""
    contexts = []
    for distance, sentence in enumerate(reversed(before)):
        contexts.append(LocalSentence(
            sentence, "before", distance,
            _sentence_rank(sentence, distance, "before"),
        ))
    for distance, sentence in enumerate(after):
        contexts.append(LocalSentence(
            sentence, "after", distance,
            _sentence_rank(sentence, distance, "after"),
        ))
    contexts = [ctx for ctx in contexts if not _is_boilerplate(ctx.text)]
    contexts.sort(key=lambda ctx: (-ctx.rank, ctx.distance, 0 if ctx.side == "before" else 1))
    return contexts[:limit]


def _sentence_rank(sentence, distance, side):
    score = max(0.0, 1.0 - 0.25 * distance)
    if _DEFINITION_SENTENCE_CUE_RE.search(sentence):
        score += 1.0
    if sentence.rstrip().endswith(":"):
        score += 0.45
    if re.search(r"\b(?:equation|relation|function|operator|matrix|state|rule)\b", sentence, re.I):
        score += 0.15
    if side == "before":
        score += 0.1
    if _NEGATIVE_CONTEXT_RE.search(sentence):
        score -= 0.8
    if re.search(r"\b(?:eqs?|refs?|sections?)\.?\s*\(?\d", sentence, re.I):
        score -= 0.25
    return round(score, 3)


def _stage_definiendum(contexts):
    """The subject/object of a definition cue in the best local sentences."""
    out = []
    for ctx in contexts:
        out += _definiendum_in_sentence(ctx.text, ctx.distance)
        out += _regex_cue_candidates(ctx.text, ctx.distance)
    return out


def _definiendum_in_sentence(sentence, distance):
    doc = _parse(sentence)
    if doc is None or not doc.has_annotation("DEP"):
        return []
    colon_intro = sentence.rstrip().endswith(":")
    out = []
    for token in doc:
        lemma = token.lemma_.lower()
        is_def = lemma in _DEFINITOR_LEMMAS and _is_definitional(token)
        is_prod = lemma in _PRODUCTION_LEMMAS and (colon_intro or _is_definitional(token))
        if not (is_def or is_prod):
            continue
        subject = _definitor_subject(token)
        if subject is not None and subject.pos_ in ("NOUN", "PROPN") \
                and subject.lower_ not in _RELATIVE_PRONOUNS | _SELF_REFERENCE:
            phrase = _noun_span(subject)
            if phrase:
                out.append(Candidate(phrase, "definiendum (cue subject)",
                                     TIER_DEFINIENDUM, distance,
                                     f"cue '{token.text}' -> subject '{phrase}'"))
        else:
            for child in token.children:
                if child.dep_ in ("dobj", "attr", "oprd"):
                    phrase = _noun_span(child)
                    if phrase:
                        out.append(Candidate(phrase, "definiendum (cue object)",
                                             TIER_DEFINIENDUM, distance,
                                             f"cue '{token.text}' -> object '{phrase}'"))
        if lemma in _NAMING_LEMMAS:
            out += _naming_complements(token, subject, distance)
    return out


def _regex_cue_candidates(sentence, distance):
    out = []
    for pattern, evidence in _REGEX_CUE_PATTERNS:
        for match in pattern.finditer(sentence):
            phrase = _clean(match.group("phrase"))
            if phrase:
                out.append(Candidate(
                    phrase, "definiendum (regex cue)", TIER_DEFINIENDUM, distance,
                    f"{evidence}: '{phrase}'",
                ))
    return out


def _is_definitional(verb):
    if verb.lemma_.lower() in _STRONG_DEFINITORS:
        return True
    children = list(verb.children)
    if any(c.dep_ == "auxpass" for c in children):
        return True
    return any(c.dep_ == "prep" and c.lower_ in ("as", "by") for c in children)


def _definitor_subject(verb):
    subject = next((c for c in verb.children if c.dep_ in ("nsubjpass", "nsubj")), None)
    if subject is None:
        return verb.head if verb.dep_ in ("acl", "relcl") else None
    if subject.lower_ in _RELATIVE_PRONOUNS:
        return verb.head if verb.dep_ in ("relcl", "acl") else None
    return subject


def _naming_complements(verb, subject, distance):
    out = []
    for child in verb.children:
        if child.dep_ not in ("attr", "oprd", "acomp", "dobj"):
            continue
        if child.pos_ in ("NOUN", "PROPN"):
            phrase = _noun_span(child)
        elif child.pos_ == "ADJ" and subject is not None:
            phrase = f"{child.text} {subject.text}"
        else:
            continue
        if phrase:
            out.append(Candidate(phrase, "alias (naming complement)", TIER_DEFINIENDUM,
                                 distance, f"naming '{verb.text} {phrase}'"))
    return out


# ── Stage 2: named term ───────────────────────────────────────────────────────

def _stage_alias_named(before, after):
    """Named concepts that contain a proper noun ('Dirac equation')."""
    out = []
    for distance, sentence in enumerate(reversed(before)):
        if not _is_boilerplate(sentence):
            out += _named_terms_in_sentence(sentence, distance)
    for distance, sentence in enumerate(after):
        if not _is_boilerplate(sentence):
            out += _named_terms_in_sentence(sentence, distance)
    return out


def _named_terms_in_sentence(sentence, distance):
    doc = _parse(sentence)
    if doc is None:
        return []
    out = []
    for chunk in doc.noun_chunks:
        if chunk.root.pos_ not in ("NOUN", "PROPN"):
            continue
        if not any(t.pos_ == "PROPN" for t in chunk):
            continue
        alpha = [t for t in chunk if t.is_alpha]
        if len(alpha) < 2 and not re.fullmatch(r"[A-Z]{2,}", chunk.root.text):
            continue
        out.append(Candidate(chunk.text, "named term", TIER_NAMED, distance,
                             f"named term '{chunk.text}'"))
    return out


# ── Stage 3: LHS symbol ───────────────────────────────────────────────────────

_LHS_SUPPRESS_ROLES = {
    "commutation relation", "differential equation", "inequality",
    "recurrence relation",
}


def _stage_lhs_symbol(equation, symbols, role):
    if not symbols or role in _LHS_SUPPRESS_ROLES:
        return []
    key = _principal_lhs_symbol(equation.get("equation", ""), symbols)
    if not key:
        return []
    definition = str(symbols.get(key, "")).strip()
    if not definition or definition.lower().startswith("definition not found"):
        return []
    definition = re.split(r"\s+(?:or|/)\s+", definition, maxsplit=1)[0].strip()
    if not (1 <= len(definition.split()) <= 6):
        return []
    return [Candidate(definition, "LHS symbol definition", TIER_LHS_SYMBOL, 0,
                      f"LHS symbol '{key}' = {definition}")]


def _principal_lhs_symbol(eq_text, symbols):
    eq = _normalize(eq_text)
    lhs = eq.split("=", 1)[0] if "=" in eq else eq[:80]
    if not lhs:
        return None
    best_key, best_pos = None, None
    for key in symbols:
        if not key:
            continue
        if len(key) > 1:
            match = re.search(r"\\(?:var)?" + re.escape(key) + r"(?![A-Za-z])", lhs)
        else:
            match = re.search(r"(?<![A-Za-z\\])" + re.escape(key) + r"(?![A-Za-z])", lhs)
        if match and (best_pos is None or match.start() < best_pos):
            best_key, best_pos = key, match.start()
    return best_key


# ── Stage 4: role name ────────────────────────────────────────────────────────

ROLE_MEANINGS = {
    "commutation relation": "commutation relation",
    "inequality": "inequality",
    "differential equation": "differential equation",
    "master equation": "master equation",
    "recurrence relation": "recurrence relation",
}


def _equation_role(equation):
    eq = _normalize(equation.get("equation", ""))
    ctx = _scrub(
        " ".join(
            equation.get(part, "") for part in ("text_before", "text_after", "section")
        )
    ).lower()
    if re.search(r"\[[^\]]+,[^\]]+\]\s*=", eq):
        return "commutation relation"
    if re.search(r"\\(?:leq?|geq?|ll|gg)\b|<=|>=", eq):
        return "inequality"
    if re.search(r"\\partial|\\frac\{d", eq) and ("master equation" in ctx or "lindblad" in ctx):
        return "master equation"
    if re.search(r"\\partial|\\frac\{d|\\dot", eq):
        return "differential equation"
    if any(w in ctx for w in ("recursion", "recurrence", "iterate", "update rule")):
        return "recurrence relation"
    return None


# ── Selection and scoring ─────────────────────────────────────────────────────

_ROLE_PHRASE_CUES = {
    "commutation relation": ("commutation", "commutator", "relation", "algebra"),
    "inequality": ("inequality", "bound"),
    "differential equation": ("equation", "dynamics", "evolution", "motion"),
    "master equation": ("master", "lindblad", "equation", "evolution"),
    "recurrence relation": ("recurrence", "recursion", "update", "relation"),
}


def _select(candidates, role, equation):
    """Pick the highest-precision tier; within it, the closest/best term."""
    if candidates:
        compatible = [c for c in candidates if _role_compatible(c, role)]
        if role in _ROLE_PHRASE_CUES:
            candidates = compatible
        else:
            candidates = compatible or candidates
    if candidates:
        best_tier = min(c.tier for c in candidates)
        pool = [c for c in candidates if c.tier == best_tier]
        local = [c for c in pool if c.stage == "local cue"]
        if local:
            best = local[0]
        else:
            embedded = [c for c in pool if c.embedding_selected]
            best = embedded[0] if embedded else min(pool, key=lambda c: (c.distance, -c.score))
        return {
            "meaning": best.phrase,
            "confidence": _confidence(best, equation),
            "stage": best.stage,
            "evidence": best.evidence,
        }
    role_meaning = ROLE_MEANINGS.get(role) if role else None
    if role_meaning:
        return {
            "meaning": role_meaning,
            "confidence": "medium",
            "stage": "role name",
            "evidence": f"structural role: {role}",
        }
    return {
        "meaning": "",
        "confidence": "low",
        "stage": "empty (no local naming evidence)",
        "evidence": "no definition cue, named term, or LHS-symbol definition found",
    }


def _role_compatible(candidate, role):
    if not role or role not in _ROLE_PHRASE_CUES:
        return True
    lower = candidate.phrase.lower()
    if any(cue in lower for cue in _ROLE_PHRASE_CUES[role]):
        return True
    return candidate.stage == "named term" and _looks_named(candidate.phrase)


def _score(candidate):
    score = max(0.0, 1.0 - 0.2 * candidate.distance)
    words = candidate.phrase.split()
    if 2 <= len(words) <= 5:
        score += 0.15
    if _looks_named(candidate.phrase):
        score += 0.1
    return round(min(1.0, score), 3)


def _dedupe_candidates(candidates):
    """Keep the strongest evidence for each phrase."""
    by_phrase = {}
    for cand in candidates:
        key = cand.phrase.lower()
        current = by_phrase.get(key)
        if current is None or (cand.tier, cand.distance) < (current.tier, current.distance):
            by_phrase[key] = cand
    return list(by_phrase.values())


def _confidence(candidate, equation):
    base = candidate.score
    if equation.get("equation_confidence") == "low":
        base -= 0.1
    if candidate.embedding_selected:
        base += 0.05
    if candidate.tier == TIER_DEFINIENDUM and candidate.distance == 0:
        return "high"
    if candidate.tier <= TIER_LHS_SYMBOL and base >= 0.6:
        return "high"
    if base >= 0.45:
        return "medium"
    return "low"


# ── Phrase cleaning and validity ──────────────────────────────────────────────

_LEAD_STRIP = re.compile(
    r"^(?:the|a|an|this|that|these|those|its|their|our|some|any|each|"
    r"following|above|given|resulting|corresponding|considered|associated)\s+",
    re.IGNORECASE,
)
_GENERIC_SINGLE = {
    "relation", "relations", "equation", "equations", "expression", "expressions",
    "transformation", "function", "functions", "state", "states", "matrix",
    "operator", "operators", "probability", "distribution", "quantity", "term",
    "value", "form", "definition", "parameter", "variable", "constant",
    "coefficient", "factor", "element", "system", "model", "time", "position",
    "coordinate", "index", "approach", "result", "case", "way", "part", "set",
    "number", "point", "order", "step", "family", "sequence", "output", "input",
    "sum", "summation", "product", "integral", "difference", "ratio", "series",
    "fact", "note", "thing", "idea", "notion", "object", "property",
}
_REFERENCE_WORDS = {
    "fig", "figs", "figure", "eq", "eqs", "ref", "refs", "sec", "sect",
    "section", "tab", "app", "appendix", "thm", "lemma", "prop", "ch", "cf",
    "preprint", "ii", "iii", "iv", "vi", "vii", "viii", "ix", "xi", "xii",
}
_NOUN_STOPWORDS = {
    "paper", "section", "result", "table", "figure", "appendix", "reference",
    "work", "case", "example", "note", "text", "term", "equation", "expression",
    "form", "way", "part", "number", "system", "approach",
    "herein", "hereof", "hereby", "thereof", "therein", "thereto", "therewith",
    "above", "below", "thus", "hence", "therefore", "moreover", "furthermore",
    "error", "errors", "overhead", "improvement", "improvements", "accuracy",
    "performance", "efficiency", "advantage", "disadvantage", "limitation",
    "comparison", "benchmark", "trade", "tradeoff",
}


def _noun_span(token):
    """Noun-chunk text headed by token, re-attaching a single of-complement."""
    if token.pos_ not in ("NOUN", "PROPN"):
        return ""
    head = next((c.text for c in token.doc.noun_chunks if c.root == token), token.text)
    of_prep = next((c for c in token.children if c.dep_ == "prep" and c.lower_ == "of"), None)
    pobj = of_prep and next((c for c in of_prep.children if c.dep_ == "pobj"), None)
    if pobj is not None and pobj.pos_ in ("NOUN", "PROPN"):
        tail = next((c.text for c in token.doc.noun_chunks if c.root == pobj), pobj.text)
        head = f"{head} of {tail}"
    return _clean(head)


def _clean(text):
    phrase = re.sub(r"\s+", " ", str(text or "")).strip(" :;,.()[]{}")
    phrase = _LEAD_STRIP.sub("", phrase)
    phrase = re.sub(r"(?<![\w/-])\d+(?:/\d+)?(?![\w/-])", " ", phrase)
    phrase = re.sub(r"\s+", " ", phrase)
    words = phrase.split()
    if len(words) > 6:
        phrase = " ".join(words[-6:])
        phrase = _LEAD_STRIP.sub("", phrase)
    return phrase.strip(" :;,.-/")


def _valid_phrase(phrase):
    lower = phrase.lower()
    if not phrase or len(phrase) < 3:
        return False
    if len(phrase.split()) == 1 and lower in _GENERIC_SINGLE:
        return False
    if any(w.lower().strip(".") in _REFERENCE_WORDS for w in phrase.split()):
        return False
    if re.fullmatch(r"[A-Za-z]{1,6}\d{1,4}[a-z]?", phrase):
        return False
    if re.search(r"[A-Za-z]\d{4}|\d{4}[A-Za-z]", phrase):
        return False
    if any(sym in phrase for sym in ("\\", "{", "}", "_", "^", "=")):
        return False
    if re.search(
        r"(?:SUPER|SUB)SCRIPT|start_ARG|end_ARG|italic_|roman_|"
        r"bold_|mathchar|relax|displaystyle",
        phrase, re.IGNORECASE,
    ):
        return False
    if re.search(
        r"\b(?:satisfy|satisfies|obey|obeys|become|becomes|yield|yields|"
        r"give|gives|read|reads|write|writes)\b",
        lower,
    ):
        return False
    if re.search(r"\bwhere\b", lower):
        return False
    if re.fullmatch(r"(?:LHS|RHS|lhs|rhs)", phrase):
        return False
    if re.search(
        r"\b(?:is|are|was|were|be|been|being|can|could|may|might|must|will|"
        r"would|shall|should|has|have|had|does|do|did|reduces?|constructed|"
        r"expressed|appended|applies|recovers?)\b",
        lower,
    ):
        return False
    if re.search(
        r"\b(?:and|or|of|in|on|for|to|by|with|as|that|which|the|a|an|"
        r"from|then|into|over|under|via|per|effectively|directly|split)\s*$",
        lower,
    ):
        return False
    if re.match(r"^(?:of|and|or|in|on|for|to|by|with|as|from|that|which)\b", lower):
        return False
    if re.match(r"^(?:I{1,3}|IV|V?I{0,3}|I?X|X{0,3}I{0,3})(?:\s+[A-Z])?$", phrase, re.I):
        return False
    if re.search(r"[^A-Za-z0-9 '\-/]", phrase):
        return False
    if len(re.findall(r"[A-Za-z]", phrase)) < 3:
        return False
    if len(phrase.split()) > 6:
        return False
    tokens = phrase.split()
    if len(tokens) >= 2 and sum(1 for t in tokens if len(t) <= 2) > len(tokens) / 2:
        return False
    return _has_noun_head(phrase) or _looks_named(phrase)


def _has_noun_head(phrase):
    doc = _parse(f"the {phrase} is")
    if doc is None:
        return len(phrase.split()) >= 2
    nouns = [t for t in doc if t.pos_ in ("NOUN", "PROPN")]
    return bool(nouns) and nouns[-1].lemma_.lower() not in _NOUN_STOPWORDS


def _looks_named(phrase):
    words = phrase.split()
    if len(words) == 1:
        return bool(re.fullmatch(r"[A-Z]{2,}", phrase))
    named = sum(1 for w in words if re.match(r"[A-Z][A-Za-z0-9-]{2,}$|[A-Z]{2,}$", w))
    return named >= 1


def _is_boilerplate(sentence):
    if re.search(
        r"\b(?:patents?|copyright|licen[sc]e|all rights reserved|arxiv:|doi:|"
        r"e-?mail|affiliation)\b",
        sentence, flags=re.IGNORECASE,
    ):
        return True
    return len(re.findall(r"\b[A-Z]{3,}\b", sentence)) >= 4


def _normalize(text):
    return (
        str(text or "")
        .replace("ˆ", r"\hat")
        .replace("†", r"\dagger")
        .replace("−", "-")
        .replace("ρ", r"\rho")
        .replace("σ", r"\sigma")
    )
