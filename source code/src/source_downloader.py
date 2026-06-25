"""
Phase 1 — Source Acquisition
==============================

Downloads paper sources from arXiv in order of quality:

1. **HTML** (ar5iv mirror: ``/html/{id}``) — best quality; structured DOM with
   MathML tags that the HTML adapter can parse directly.
2. **PDF** (``/pdf/{id}``) — fallback when no HTML is available; raw text
   extracted and post-processed by the PDF engines.

robots.txt compliance (arXiv, verified 2026-06-15)
---------------------------------------------------
* ``Allow: /html``     — used for HTML download
* ``Allow: /pdf``      — used for PDF download
* ``Disallow: /src``   — source packages are blocked; never fetched
* ``Crawl-delay: 15``  — enforced between every request

Public API
----------
* ``detect_best_format(arxiv_id)``         — choose HTML or PDF
* ``download_html(arxiv_id)``              — fetch ar5iv HTML
* ``download_pdf(arxiv_id)``               — fetch arXiv PDF
"""

import logging
import os
import re
import time

import requests

logger = logging.getLogger(__name__)

__all__ = ["detect_best_format", "download_html", "download_pdf"]

# ── Configuration ──────────────────────────────────────────────────────────

USER_AGENT = "EquationKG-NLP-StudentProject/1.0 (academic research)"
CRAWL_DELAY = 15  # seconds — from arXiv robots.txt

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(PROJECT_DIR, "data", "sources")


# ── Public Functions ───────────────────────────────────────────────────────

def detect_best_format(arxiv_id):
    """Detect the best available source format for an arXiv paper.

    Tries HTML first, then falls back to PDF.

    Parameters
    ----------
    arxiv_id : str
        The arXiv paper ID (e.g. "2507.03587").

    Returns
    -------
    format_name : str
        "html" or "pdf".
    audit_message : str
        Short message for the audit trail.
    """
    clean_id = arxiv_id.replace("arXiv:", "").strip()

    html_cache = _html_cache_path(clean_id)
    if os.path.exists(html_cache) and _looks_like_paper_html(html_cache):
        return "html", f"HTML cache available at {html_cache}"

    html_url, audit = _discover_html_url(clean_id)
    if html_url:
        logger.info("arXiv:%s → HTML available at %s", clean_id, html_url)
        return "html", audit

    pdf_cache = _pdf_cache_path(clean_id)
    if os.path.exists(pdf_cache):
        return "pdf", f"Using cached PDF at {pdf_cache}; no valid HTML source found"

    # ── Fallback to PDF ────────────────────────────────────────────────
    logger.info("arXiv:%s → Falling back to PDF", clean_id)
    pdf_url = f"https://arxiv.org/pdf/{clean_id}"
    return "pdf", f"HTML unavailable, falling back to PDF at {pdf_url}"


def download_html(arxiv_id):
    """Download the HTML version of a paper from ar5iv.

    Parameters
    ----------
    arxiv_id : str
        The arXiv paper ID.

    Returns
    -------
    html_content : str or None
        The full HTML string, or None on failure.
    audit_message : str
        Short message for the audit trail.
    """
    clean_id = arxiv_id.replace("arXiv:", "").strip()
    cache_path = _html_cache_path(clean_id)
    if os.path.exists(cache_path) and _looks_like_paper_html(cache_path):
        with open(cache_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return content, f"HTML loaded from local cache {cache_path}"

    url, discovery_audit = _discover_html_url(clean_id)
    if not url:
        return None, discovery_audit

    headers = {"User-Agent": USER_AGENT}

    time.sleep(CRAWL_DELAY)
    try:
        response = requests.get(url, headers=headers, timeout=60)
        if response.status_code == 200:
            content = response.text
            os.makedirs(SOURCE_DIR, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(
                "arXiv:%s → HTML downloaded (%d chars)", clean_id, len(content)
            )
            return content, f"HTML downloaded from {url} ({len(content)} chars)"
        else:
            logger.warning(
                "arXiv:%s → HTML download failed (HTTP %d)",
                clean_id, response.status_code,
            )
            return None, f"HTML download failed (HTTP {response.status_code})"
    except requests.RequestException as exc:
        logger.warning("arXiv:%s → HTML download error: %s", clean_id, exc)
        return None, f"HTML download error: {exc}"


def download_pdf(arxiv_id):
    """Download the PDF version of a paper from arXiv.

    Parameters
    ----------
    arxiv_id : str
        The arXiv paper ID.

    Returns
    -------
    pdf_bytes : bytes or None
        The raw PDF bytes, or None on failure.
    audit_message : str
        Short message for the audit trail.
    """
    clean_id = arxiv_id.replace("arXiv:", "").strip()
    cache_path = _pdf_cache_path(clean_id)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            pdf_bytes = f.read()
        return pdf_bytes, f"PDF loaded from local cache {cache_path} ({len(pdf_bytes)} bytes)"

    url = f"https://arxiv.org/pdf/{clean_id}"
    headers = {"User-Agent": USER_AGENT}

    time.sleep(CRAWL_DELAY)
    try:
        response = requests.get(url, headers=headers, timeout=60)
        if response.status_code == 200:
            pdf_bytes = response.content
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                f.write(pdf_bytes)
            logger.info(
                "arXiv:%s → PDF downloaded (%d bytes)", clean_id, len(pdf_bytes)
            )
            return pdf_bytes, f"PDF downloaded from {url} ({len(pdf_bytes)} bytes)"
        else:
            logger.warning(
                "arXiv:%s → PDF download failed (HTTP %d)",
                clean_id, response.status_code,
            )
            return None, f"PDF download failed (HTTP {response.status_code})"
    except requests.RequestException as exc:
        logger.warning("arXiv:%s → PDF download error: %s", clean_id, exc)
        return None, f"PDF download error: {exc}"


def _html_cache_path(clean_id):
    """Return the canonical local HTML cache path for an arXiv ID."""
    return os.path.join(SOURCE_DIR, f"{clean_id}.html")


def _pdf_cache_path(clean_id):
    """Return the canonical local PDF cache path for an arXiv ID."""
    return os.path.join(SOURCE_DIR, clean_id.replace("/", "_"), "paper.pdf")


def _looks_like_paper_html(path):
    """Check whether cached HTML is the experimental paper HTML, not an abs page."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(50000)
    except OSError:
        return False
    return "ltx_equation" in sample or "ltx_document" in sample


def _discover_html_url(clean_id):
    """Find the best allowed arXiv HTML URL via /html and /abs pages.

    The direct /html/{id} endpoint is not always the final paper HTML. Some
    papers expose only a versioned link such as /html/2507.03587v2 on /abs.
    """
    headers = {"User-Agent": USER_AGENT}
    direct_url = f"https://arxiv.org/html/{clean_id}"

    time.sleep(CRAWL_DELAY)
    try:
        response = requests.get(direct_url, headers=headers, timeout=60)
        if response.status_code == 200 and "ltx_document" in response.text:
            return direct_url, f"HTML available at {direct_url}"
    except requests.RequestException as exc:
        logger.warning("arXiv:%s → direct HTML check failed: %s", clean_id, exc)

    abs_url = f"https://arxiv.org/abs/{clean_id}"
    time.sleep(CRAWL_DELAY)
    try:
        response = requests.get(abs_url, headers=headers, timeout=60)
    except requests.RequestException as exc:
        return None, f"HTML discovery failed from {abs_url}: {exc}"

    if response.status_code != 200:
        return None, f"HTML discovery failed from {abs_url} (HTTP {response.status_code})"

    match = re.search(r'href="(/html/[^"]+)"', response.text)
    if match:
        html_url = "https://arxiv.org" + match.group(1)
        return html_url, f"HTML discovered from {abs_url}: {html_url}"

    versions = sorted({int(v) for v in re.findall(r"\bv(\d+)\b", response.text)}, reverse=True)
    for version in versions:
        versioned_url = f"https://arxiv.org/html/{clean_id}v{version}"
        time.sleep(CRAWL_DELAY)
        try:
            versioned = requests.get(versioned_url, headers=headers, timeout=60)
        except requests.RequestException:
            continue
        if versioned.status_code == 200 and "ltx_document" in versioned.text:
            return versioned_url, f"HTML discovered by version probe: {versioned_url}"

    return None, f"No HTML link found on {abs_url}"
