"""Phase 5 — Equation Meaning Extraction.

Evidence-ordered, extractive meaning assignment — **no text generation**.

Extraction stages (in decreasing precision)
--------------------------------------------
0  Local cue      — production/copular verb in the sentence before the equation
1  Definiendum    — subject/object of a definition cue in ranked sentences
2  Named term     — proper-noun compound ("Dirac equation", "CHSH inequality")
3  LHS symbol     — principal left-hand symbol's own definition (Phase 4)
4  Role name      — structural name inferred from equation syntax
5  Empty          — returned when no evidence is found

Public API
----------
* ``extract_meaning(equation, symbols=None)``
* ``extract_meaning_with_confidence(equation, symbols=None)``

References
----------
* Pagael & Schubotz (2014) — "Mathematical Language Processing"
* DEFT/Navigli-Velardi — definition extraction via dependency parses
* MathAlign — equation-to-description alignment
"""

import os
import re
from functools import lru_cache

from src.audit_trail import build_meaning_audit
from src.meaning_stage_extractors import (
    Candidate,
    LocalSentence,
    _clean,
    _dedupe_candidates,
    _equation_role,
    _is_boilerplate,
    _ranked_definition_contexts,
    _scrub,
    _select,
    _score,
    _sentences,
    _stage_alias_named,
    _stage_definiendum,
    _stage_local_cue,
    _stage_lhs_symbol,
    _valid_phrase,
    TIER_DEFINIENDUM,
    TIER_NAMED,
    TIER_LHS_SYMBOL,
    TIER_ROLE,
)

__all__ = [
    "extract_meaning",
    "extract_meaning_with_confidence",
]


# ── spaCy (parser only) ───────────────────────────────────────────────────────

_NLP = None


def _spacy():
    """Load spaCy once; fall back to a blank pipe (regex-only) if unavailable."""
    global _NLP
    if _NLP is not None:
        return _NLP or None
    try:
        import spacy
        try:
            _NLP = spacy.load("en_core_web_sm")
        except Exception:
            _NLP = spacy.blank("en")
    except Exception:
        _NLP = False
    return _NLP or None


@lru_cache(maxsize=8192)
def _parse(text):
    """Parse a sentence once (cached). Returns None without POS/DEP annotations."""
    nlp = _spacy()
    if not nlp:
        return None
    try:
        doc = nlp(str(text))
    except Exception:
        return None
    return doc if doc.has_annotation("TAG") else None


# ── Public API ────────────────────────────────────────────────────────────────

def extract_meaning(equation, symbols=None):
    """Return ``(meaning, confidence, audit)`` for the enrichment pipeline."""
    return extract_meaning_with_confidence(equation, symbols or {})


def extract_meaning_with_confidence(equation, symbols=None):
    """Select an equation meaning from local paper text + the LHS symbol."""
    symbols = symbols or {}
    role = _equation_role(equation)
    before = _sentences(equation.get("text_before", ""))[-5:]
    after = _sentences(equation.get("text_after", ""))[:5]
    definition_contexts = _ranked_definition_contexts(before, after)
    named_before = [ctx.text for ctx in sorted(
        (ctx for ctx in definition_contexts if ctx.side == "before"),
        key=lambda ctx: ctx.distance,
    )][-3:] or before[-3:]
    named_after = [ctx.text for ctx in sorted(
        (ctx for ctx in definition_contexts if ctx.side == "after"),
        key=lambda ctx: ctx.distance,
    )][:2] or after[:2]

    candidates = []
    candidates += _stage_local_cue(before, after)
    candidates += _stage_definiendum(definition_contexts)
    candidates += _stage_alias_named(named_before, named_after)
    candidates += _stage_lhs_symbol(equation, symbols, role)

    for cand in candidates:
        cand.phrase = _clean(cand.phrase)
    candidates = _dedupe_candidates([c for c in candidates if _valid_phrase(c.phrase)])
    for cand in candidates:
        cand.score = _score(cand)
    embedding_decision = _embedding_rerank(candidates, equation)

    selected = _select(candidates, role, equation)
    source_sentence = _source_sentence_for(
        selected.get("meaning", ""), before, after, definition_contexts
    )
    decision = {
        "selected_stage": selected["stage"],
        "role": role or "none",
        "evidence": selected.get("evidence", ""),
        "source_sentence": source_sentence,
        "embedding_rerank": embedding_decision,
        "ranked_sentences": [
            {"side": ctx.side, "distance": ctx.distance, "rank": round(ctx.rank, 3)}
            for ctx in definition_contexts[:4]
        ],
        "candidates": [
            {
                "phrase": c.phrase,
                "stage": c.stage,
                "score": c.score,
                "embedding_score": c.embedding_score,
            }
            for c in sorted(candidates, key=lambda c: (c.tier, c.distance, -c.score))[:5]
        ],
    }
    audit = build_meaning_audit(
        selected["meaning"], selected["confidence"], equation, decision
    )
    return selected["meaning"], selected["confidence"], audit


def _source_sentence_for(meaning, before, after, definition_contexts):
    """Return the local sentence the chosen meaning phrase was read from."""
    phrase = _clean(meaning or "").lower()
    if not phrase:
        return ""
    candidates = list(before) + list(after) + [c.text for c in definition_contexts]
    for sentence in candidates:
        if phrase in _clean(sentence).lower():
            return re.sub(r"\s+", " ", sentence).strip()
    return ""


# ── Optional embedding re-ranker ──────────────────────────────────────────────

_EMBEDDING_MODEL = None
_EMBEDDING_UNAVAILABLE = False
_EMBEDDING_MODEL_NAME = os.getenv("MEANING_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
_EMBEDDING_THRESHOLD = float(os.getenv("MEANING_EMBEDDING_THRESHOLD", "0.35"))
_EMBEDDING_MARGIN = float(os.getenv("MEANING_EMBEDDING_MARGIN", "0.03"))


def _embedding_model():
    """Load a local sentence-transformer model if available (never calls an API)."""
    global _EMBEDDING_MODEL, _EMBEDDING_UNAVAILABLE
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL
    if _EMBEDDING_UNAVAILABLE:
        return None
    try:
        from sentence_transformers import SentenceTransformer
        local_only = os.getenv("MEANING_EMBEDDING_LOCAL_ONLY", "1") != "0"
        _EMBEDDING_MODEL = SentenceTransformer(
            _EMBEDDING_MODEL_NAME, local_files_only=local_only,
        )
    except Exception:
        _EMBEDDING_UNAVAILABLE = True
        _EMBEDDING_MODEL = None
    return _EMBEDDING_MODEL


def _embedding_rerank(candidates, equation):
    """Use embeddings only to choose among same-tier extracted candidates."""
    if len(candidates) < 2:
        return {"used": False, "reason": "fewer than two candidates"}
    best_tier = min(c.tier for c in candidates)
    pool = [c for c in candidates if c.tier == best_tier]
    if len(pool) < 2:
        return {"used": False, "reason": "single candidate in best tier"}

    model = _embedding_model()
    if model is None:
        return {"used": False, "reason": "embedding model unavailable"}

    query = _embedding_query(equation)
    if not query:
        return {"used": False, "reason": "no context query"}

    try:
        texts = [query] + [c.phrase for c in pool]
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        query_vec = vectors[0]
        scored = []
        for cand, vector in zip(pool, vectors[1:]):
            score = float(query_vec @ vector)
            cand.embedding_score = round(score, 4)
            scored.append((score, cand))
    except Exception as exc:
        return {"used": False, "reason": f"embedding failed: {type(exc).__name__}"}

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else -1.0
    margin = best_score - second_score
    if best_score >= _EMBEDDING_THRESHOLD and margin >= _EMBEDDING_MARGIN:
        best.embedding_selected = True
        best.evidence = (
            f"{best.evidence}; embedding rerank selected against local context "
            f"(score={best_score:.3f}, margin={margin:.3f})"
        )
        return {
            "used": True,
            "selected": best.phrase,
            "score": round(best_score, 4),
            "margin": round(margin, 4),
            "model": _EMBEDDING_MODEL_NAME,
        }
    return {
        "used": False,
        "reason": "embedding score/margin below gate",
        "best": best.phrase,
        "score": round(best_score, 4),
        "margin": round(margin, 4),
        "model": _EMBEDDING_MODEL_NAME,
    }


def _embedding_query(equation):
    """Compact local-context query for candidate meaning ranking."""
    before = _sentences(equation.get("text_before", ""))[-3:]
    after = _sentences(equation.get("text_after", ""))[:2]
    section = _scrub(equation.get("section", ""))
    lhs = equation.get("equation", "")
    lhs = lhs.split("=", 1)[0][:80] if "=" in lhs else lhs[:80]
    parts = [section, *before, *after, lhs]
    query = " ".join(part for part in parts if part)
    return re.sub(r"\s+", " ", query).strip()[:700]
