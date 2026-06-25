# Equations Knowledge Graph - Final Submission

This folder contains the runnable source code and final dataset for Exam ID 1.
The pipeline extracts numbered equations from arXiv papers, enriches them with
meanings, symbol definitions, pairwise equation relations, and an audit trail.

## Final Output

The dataset is stored at:

```text
data/output/dataset.json
```

Current output:

- 60 papers
- 350 equations
- 5 fields per equation: `equation`, `meaning`, `symbols`, `relations`, `audit-trail`

## Folder Structure

```text
Final_submission/
├── README.md
├── build_dataset.py
├── paper_list_1.txt
├── requirements.txt
├── data/
│   ├── output/
│   │   └── dataset.json
│   └── sources/
│       └── cached arXiv HTML/PDF files
└── src/
    ├── __init__.py
    ├── audit_trail.py
    ├── cross_equation_processing.py
    ├── docling_adapter.py
    ├── enrichment_common.py
    ├── equation_extraction.py
    ├── html_adapter.py
    ├── latex_ocr_adapter.py
    ├── meaning_extraction.py
    ├── meaning_stage_extractors.py
    ├── nlp_enrichment.py
    ├── pdf_adapter.py
    ├── pdf_engines.py
    ├── pdf_html_reconciliation.py
    ├── pdf_math.py
    ├── pdf_text_helpers.py
    ├── relation_detection.py
    ├── sequence_validation.py
    ├── source_downloader.py
    ├── symbol_definition_extraction.py
    ├── symbol_definition_text.py
    ├── symbol_extraction.py
    └── symbol_structural_analysis.py
```

## Run

Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Run the full pipeline:

```bash
python build_dataset.py
```

Run only the first paper as a quick check:

```bash
python build_dataset.py --num-papers 1
```

## Pipeline Summary

- `source_downloader.py`: downloads or loads cached arXiv HTML/PDF sources.
- `equation_extraction.py`, `html_adapter.py`, `pdf_adapter.py`: extract numbered equations.
- `symbol_extraction.py`: extracts equation symbols.
- `symbol_definition_extraction.py`, `symbol_definition_text.py`, `symbol_structural_analysis.py`: extract symbol definitions.
- `meaning_extraction.py`, `meaning_stage_extractors.py`: extract equation meanings.
- `relation_detection.py`: builds pairwise equation relations.
- `audit_trail.py`: builds and validates audit-trail entries.
- `build_dataset.py`: runs the complete pipeline and writes `dataset.json`.

The extraction pipeline is deterministic and does not use LLM prompting or
external generative APIs.
