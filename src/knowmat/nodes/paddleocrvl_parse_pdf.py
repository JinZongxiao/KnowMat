"""
PDF parsing node using PaddleOCR-VL (with PaddleOCR fallback).

This parser renders each PDF page to an image, runs OCR, and returns a single
markdown text block for downstream extraction agents.  It also extracts
document-level metadata (DOI, etc.) from the first page header/footer area
and converts any residual HTML in pre-existing .txt files to clean markdown.
"""

import logging
import os
import json
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from knowmat.states import KnowMatState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DOI extraction helpers
# ---------------------------------------------------------------------------

_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b")
_DOI_URL_RE = re.compile(r"https?://doi\.org/(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)")


def _extract_doi_from_text(text: str) -> Optional[str]:
    """Return the first DOI found in *text*, or None."""
    if not text:
        return None
    m = _DOI_URL_RE.search(text)
    if m:
        return m.group(1).rstrip(".")
    m = _DOI_RE.search(text)
    if m:
        return m.group(1).rstrip(".")
    return None


# ---------------------------------------------------------------------------
# HTML ù?markdown conversion for .txt inputs
# ---------------------------------------------------------------------------

_HTML_IMG_RE = re.compile(r'<div[^>]*>\s*<img[^>]*/>\s*</div>', re.IGNORECASE)
_HTML_DIV_OPEN_RE = re.compile(r'<div[^>]*>', re.IGNORECASE)
_HTML_DIV_CLOSE_RE = re.compile(r'</div>', re.IGNORECASE)
_HTML_TABLE_BLOCK_RE = re.compile(
    r'<table[^>]*>.*?</table>', re.IGNORECASE | re.DOTALL
)
_HTML_TAG_RE = re.compile(r'</?[a-zA-Z][^>]*>')
_LATEX_DISPLAY_RE = re.compile(r'\$\$\s*(.*?)\s*\$\$', re.DOTALL)


def _html_table_to_markdown(html: str) -> str:
    """Best-effort conversion of a single <table> to a markdown table."""
    from html.parser import HTMLParser

    rows: List[List[str]] = []
    current_row: List[str] = []
    current_cell: List[str] = []
    in_cell = False

    class _TableParser(HTMLParser):
        def handle_starttag(self, tag, attrs):
            nonlocal in_cell, current_cell
            if tag in ("td", "th"):
                in_cell = True
                current_cell = []

        def handle_endtag(self, tag):
            nonlocal in_cell, current_row
            if tag in ("td", "th"):
                in_cell = False
                current_row.append(" ".join(current_cell).strip())
            elif tag == "tr":
                if current_row:
                    rows.append(current_row)
                current_row = []

        def handle_data(self, data):
            if in_cell:
                txt = data.strip()
                if txt:
                    current_cell.append(txt)

    try:
        _TableParser().feed(html)
    except (ValueError, AssertionError):
        return html

    if not rows:
        return html

    col_count = max(len(r) for r in rows)
    lines: List[str] = []
    for i, row in enumerate(rows):
        padded = row + [""] * (col_count - len(row))
        lines.append("| " + " | ".join(padded) + " |")
        if i == 0:
            lines.append("| " + " | ".join(["---"] * col_count) + " |")
    return "\n".join(lines)


def _convert_html_to_markdown(text: str) -> str:
    """Convert residual HTML markup in OCR/txt output to clean markdown."""
    if not text:
        return ""
    if "<" not in text and ">" not in text:
        return text.strip()

    converted = text
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError:
        # Fallback to regex-based stripping
        converted = _HTML_TABLE_BLOCK_RE.sub(lambda m: _html_table_to_markdown(m.group(0)), converted)
        converted = _HTML_IMG_RE.sub("", converted)
        converted = _HTML_DIV_OPEN_RE.sub("", converted)
        converted = _HTML_DIV_CLOSE_RE.sub("", converted)
        converted = _HTML_TAG_RE.sub("", converted)
    else:
        soup = BeautifulSoup(converted, "html.parser")
        for table in soup.find_all("table"):
            md_table = _html_table_to_markdown(str(table))
            table.replace_with(f"\n{md_table}\n")
        for br in soup.find_all("br"):
            br.replace_with("\n")
        block_tags = [
            "p",
            "div",
            "section",
            "article",
            "header",
            "footer",
            "li",
            "ul",
            "ol",
            "tr",
            "td",
            "th",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        ]
        for tag in soup.find_all(block_tags):
            if tag.string is None:
                tag.insert_before("\n")
                tag.insert_after("\n")
        converted = soup.get_text("\n")
        converted = _HTML_TAG_RE.sub("", converted)

    converted = re.sub(r"\n{3,}", "\n\n", converted)
    return converted.strip()


def _html_table_to_structured(html: str) -> Optional[Dict[str, Any]]:
    """Parse a <table> HTML string into a structured columns/rows dict."""
    if "<table" not in html.lower():
        return None
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError:
        return None

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return None

    # Extract header cells if present
    headers: List[str] = []
    thead = table.find("thead")
    if thead:
        header_cells = thead.find_all(["th", "td"])
        headers = [cell.get_text(" ", strip=True) for cell in header_cells if cell]

    # Fallback: use first row as header if it contains <th>
    if not headers:
        first_row = table.find("tr")
        if first_row:
            header_cells = first_row.find_all("th")
            if header_cells:
                headers = [cell.get_text(" ", strip=True) for cell in header_cells]

    rows: List[List[str]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        row = [cell.get_text(" ", strip=True) for cell in cells]
        rows.append(row)

    if headers and rows and rows[0] == headers:
        rows = rows[1:]

    if not headers and rows:
        max_len = max(len(r) for r in rows)
        headers = [f"col_{i+1}" for i in range(max_len)]

    if not headers:
        return None

    normalized_rows: List[Dict[str, str]] = []
    for row in rows:
        padded = row + [""] * (len(headers) - len(row))
        normalized_rows.append({headers[i]: padded[i] for i in range(len(headers))})

    return {
        "columns": [{"name": name, "type": "string"} for name in headers],
        "rows": normalized_rows,
    }


def _get_block_attr(block: Any, name: str, default: Any = None) -> Any:
    if hasattr(block, name):
        return getattr(block, name)
    if isinstance(block, dict):
        return block.get(name, default)
    return default


def _block_to_item(block: Any) -> Optional[Dict[str, Any]]:
    label = _get_block_attr(block, "label", None) or _get_block_attr(block, "block_label", None)
    content = _get_block_attr(block, "content", "") or _get_block_attr(block, "text", "")
    confidence = _get_block_attr(block, "score", None) or _get_block_attr(block, "confidence", None)

    if isinstance(content, str):
        content = content.strip()

    if label in ("table", "chart"):
        structured = _html_table_to_structured(content or "")
        item: Dict[str, Any] = {"typer": "table"}
        if structured:
            item.update(structured)
        else:
            item["text"] = _convert_html_to_markdown(content or "")
        if confidence is not None:
            item["confidence"] = confidence
        return item

    if label in ("image", "figure", "seal"):
        image_info = _get_block_attr(block, "image", None)
        image_path = image_info.get("path") if isinstance(image_info, dict) else None
        data: Dict[str, Any] = {"image_path": image_path}
        if content:
            data["caption"] = content
        if confidence is not None:
            data["confidence"] = confidence
        return {"typer": "image", "data": data}

    if content:
        item = {"typer": "paragraph", "text": _convert_html_to_markdown(content)}
        if confidence is not None:
            item["confidence"] = confidence
        return item
    return None


def _text_to_paragraph_items(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    return [{"typer": "paragraph", "text": block} for block in blocks]

# ---------------------------------------------------------------------------
# Section heading normalisation & noise filtering
# ---------------------------------------------------------------------------

_SPACED_TITLE_RE = re.compile(r'^(#+\s+)?([A-Z](?:\s[A-Z]){3,})$')

_SECTION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'^(?:#+\s*)?A\s*B\s*S\s*T\s*R\s*A\s*C\s*T\s*$', re.IGNORECASE), '## ABSTRACT'),
    (re.compile(r'^(?:#+\s*)?ABSTRACT\s*$', re.IGNORECASE), '## ABSTRACT'),
    (re.compile(r'^(?:#+\s*)?ARTICLE\s+INFO\s*$', re.IGNORECASE), '## ARTICLE INFO'),
    (re.compile(r'^(?:#+\s*)?KEYWORDS?\s*:?\s*$', re.IGNORECASE), '## Keywords'),
    (re.compile(r'^(?:#+\s*)?(ACKNOWLEDGEMENTS?|ACKNOWLEDGMENTS?)\s*$', re.IGNORECASE), '## Acknowledgements'),
    (re.compile(r'^(?:#+\s*)?CRediT\s+authorship\s+contribution\s+statement\s*$', re.IGNORECASE), '## CRediT Authorship'),
    (re.compile(r'^(?:#+\s*)?DECLARATION\s+OF\s+(COMPETING\s+)?INTEREST.*$', re.IGNORECASE), '## Declaration of Interest'),
    (re.compile(r'^(?:#+\s*)?DATA\s+AVAILABILITY.*$', re.IGNORECASE), '## Data Availability'),
    (re.compile(r'^(?:#+\s*)?SUPPLEMENTARY\s+(DATA|MATERIAL|INFORMATION).*$', re.IGNORECASE), '## Supplementary Material'),
    # Numbered sections: "1. Introduction", "2.1. Sample preparation", etc.
    (re.compile(r'^(?:#+\s*)?(\d+(?:\.\d+)*\.?)\s+(.+)$'), None),
]

_NOISE_LINE_PATTERNS: List[Tuple[re.Pattern, bool]] = [
    # (pattern, skip_if_contains_doi)
    # Journal name lines - require volume/year digits after the name to avoid
    # false-positives when the journal name also appears as a keyword.
    (re.compile(r'^Materials Science and Engineering\s+\d', re.IGNORECASE), True),
    (re.compile(r'^Acta Materialia\s+\d', re.IGNORECASE), True),
    (re.compile(r'^Journal of Materials Science\s+\d', re.IGNORECASE), True),
    (re.compile(r'^International Journal of Plasticity\s+\d', re.IGNORECASE), True),
    (re.compile(r'^Journal of Alloys and Compounds\s+\d', re.IGNORECASE), True),
    (re.compile(r'^Scripta Materialia\s+\d', re.IGNORECASE), True),
    (re.compile(r'^Additive Manufacturing\s+\d', re.IGNORECASE), True),
    # Lines that never contain DOI
    (re.compile(r'^Contents lists available at ScienceDirect', re.IGNORECASE), False),
    (re.compile(r'^Available online', re.IGNORECASE), False),
    (re.compile(r'^Received \d+', re.IGNORECASE), False),
    (re.compile(r'^Accepted \d+', re.IGNORECASE), False),
    (re.compile(r'^journal homepage', re.IGNORECASE), False),
    (re.compile(r'^https?://www\.(sciencedirect|elsevier|springer|wiley)', re.IGNORECASE), False),
    (re.compile(r'^Full length article\s*$', re.IGNORECASE), False),
    (re.compile(r'^\d{4}-\d{3}[\dX]/\s*ù', re.IGNORECASE), False),
    (re.compile(r'^ù\s*\d{4}', re.IGNORECASE), False),
    (re.compile(r'^Elsevier', re.IGNORECASE), False),
    (re.compile(r'^\* Corresponding author', re.IGNORECASE), False),
    (re.compile(r'^\*\* Corresponding author', re.IGNORECASE), False),
    (re.compile(r'^E-mail address', re.IGNORECASE), False),
    (re.compile(r'^\d+$'), False),  # bare page numbers
]


def _is_noise_line(line: str) -> bool:
    """Return True if *line* looks like a header/footer/metadata noise line.

    Lines that match a journal-name pattern but also contain a DOI are kept,
    because the DOI is valuable for downstream extraction.
    """
    stripped = line.strip()
    if not stripped:
        return False
    has_doi = bool(_DOI_RE.search(stripped)) or bool(_DOI_URL_RE.search(stripped))
    for pat, skip_if_doi in _NOISE_LINE_PATTERNS:
        if pat.match(stripped):
            if skip_if_doi and has_doi:
                return False  # keep the line ù?it contains a DOI
            return True
    return False


def _structure_sections(text: str) -> str:
    """Normalise heading levels and filter noise lines from parsed text.

    - Converts inconsistent ``###`` / ``####`` / ``#####`` headings to ``##``.
    - Recognises common section titles and standardises them.
    - Removes lines that are obviously journal headers, copyright, or page numbers.
    - Joins spaced-out titles like ``A B S T R A C T`` into ``ABSTRACT``.
    """
    output_lines: List[str] = []

    for line in text.splitlines():
        stripped = line.strip()

        # Skip noise lines
        if _is_noise_line(stripped):
            continue

        # Try to match known section titles
        matched = False
        for pat, replacement in _SECTION_PATTERNS:
            m = pat.match(stripped)
            if m:
                if replacement:
                    output_lines.append("")
                    output_lines.append(replacement)
                    output_lines.append("")
                else:
                    # Numbered section: normalise to ## level
                    sec_num = m.group(1).rstrip(".")
                    sec_title = m.group(2).strip()
                    output_lines.append("")
                    output_lines.append(f"## {sec_num}. {sec_title}")
                    output_lines.append("")
                matched = True
                break

        if matched:
            continue

        # Normalise spaced-out all-caps titles (e.g. "K E Y W O R D S")
        m = _SPACED_TITLE_RE.match(stripped)
        if m:
            collapsed = m.group(2).replace(" ", "")
            output_lines.append("")
            output_lines.append(f"## {collapsed}")
            output_lines.append("")
            continue

        # Normalise remaining headings to ## level (flatten deep nesting)
        heading_match = re.match(r'^(#{3,6})\s+(.*)', stripped)
        if heading_match:
            title = heading_match.group(2).strip()
            output_lines.append(f"## {title}")
            continue

        # Remove "## Page N" markers from PDF OCR output (not useful for LLM)
        if re.match(r'^##\s+Page\s+\d+\s*$', stripped):
            output_lines.append("")
            continue

        output_lines.append(line)

    result = "\n".join(output_lines)
    # Collapse excessive blank lines
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _default_model_dir() -> Path:
    """Return the default local model directory in this project."""
    repo_root = Path(__file__).resolve().parents[3]
    model_dir = os.getenv("PADDLEOCRVL_MODEL_DIR", str(repo_root / "models" / "paddleocrvl1_5"))
    return Path(model_dir).expanduser().resolve()


def _prepare_ocr_home(model_dir: Path) -> None:
    """Ensure local model directory exists and is used by PaddleOCR."""
    model_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PADDLEOCR_HOME"] = str(model_dir)


def _create_ocr_engine(model_dir: Path) -> Tuple[Any, str]:
    """Create a PaddleOCR-VL engine, or fallback to PaddleOCR."""
    _prepare_ocr_home(model_dir)

    try:
        from paddleocr import PaddleOCRVL  # type: ignore

        for kwargs in ({}, {"lang": "en"}, {"model_dir": str(model_dir)}, {"lang": "en", "model_dir": str(model_dir)}):
            try:
                return PaddleOCRVL(**kwargs), "paddleocrvl"
            except TypeError:
                continue
    except ImportError:
        logger.info("PaddleOCR-VL not available, falling back to PaddleOCR")

    try:
        from paddleocr import PaddleOCR  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PaddleOCR is not installed. Install with: pip install knowmat[ocr]"
        ) from exc

    init_candidates = (
        {"use_angle_cls": True, "lang": "en"},
        {"lang": "en"},
        {},
    )
    last_error: Exception | None = None
    for kwargs in init_candidates:
        try:
            return PaddleOCR(**kwargs), "paddleocr"
        except TypeError as exc:
            last_error = exc

    if last_error:
        raise RuntimeError(f"Failed to initialize PaddleOCR engine: {last_error}") from last_error
    raise RuntimeError("Failed to initialize PaddleOCR engine.")


def _run_ocr(engine: Any, image_path: Path) -> Any:
    """Run OCR for a single page image with broad API compatibility."""
    img = str(image_path)

    if hasattr(engine, "predict"):
        try:
            return engine.predict(img)
        except TypeError:
            return engine.predict(input=img)
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning("engine.predict() failed for %s: %s", image_path.name, exc, exc_info=True)

    if hasattr(engine, "ocr"):
        try:
            return engine.ocr(img, cls=True)
        except TypeError:
            return engine.ocr(img)

    if callable(engine):
        return engine(img)

    raise RuntimeError("OCR engine does not expose a known inference method.")


def _collect_text(obj: Any, out: List[str]) -> None:
    """Recursively collect OCR text from mixed response structures."""
    if obj is None:
        return

    if isinstance(obj, str):
        text = obj.strip()
        if text:
            out.append(text)
        return

    if isinstance(obj, dict):
        for key in ("text", "rec_text", "ocr_text", "content", "transcription"):
            val = obj.get(key)
            if isinstance(val, str):
                val = val.strip()
                if val:
                    out.append(val)
        for val in obj.values():
            _collect_text(val, out)
        return

    if isinstance(obj, (list, tuple)):
        # PaddleOCR classic format: [box, (text, score)]
        if (
            len(obj) == 2
            and isinstance(obj[1], (list, tuple))
            and len(obj[1]) > 0
            and isinstance(obj[1][0], str)
        ):
            text = obj[1][0].strip()
            if text:
                out.append(text)
            return
        for item in obj:
            _collect_text(item, out)


def _normalize_lines(lines: List[str]) -> List[str]:
    """Clean and deduplicate line-level OCR text."""
    normalized: List[str] = []
    prev = ""
    for line in lines:
        text = re.sub(r"\s+", " ", line).strip()
        if not text or text == prev:
            continue
        normalized.append(text)
        prev = text
    return normalized


def _strip_references_section(text: str) -> str:
    """Drop references/bibliography section from parsed text."""
    out: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^#+\s*(references?|bibliography|citations?)\s*$", stripped, re.IGNORECASE):
            break
        if re.match(r"^(references?|bibliography|citations?)\s*$", stripped, re.IGNORECASE):
            break
        out.append(line)
    return "\n".join(out).strip()


def _extract_doi_from_pdf_metadata(pdf_path: str) -> Optional[str]:
    """Try to extract DOI from the PDF's built-in metadata fields."""
    try:
        import fitz  # type: ignore
    except ImportError:
        return None
    try:
        doc = fitz.open(pdf_path)
        meta = doc.metadata or {}
        doc.close()
        for key in ("doi", "subject", "keywords", "title"):
            val = meta.get(key, "")
            if val:
                doi = _extract_doi_from_text(val)
                if doi:
                    return doi
    except (RuntimeError, ValueError, KeyError) as exc:
        logger.debug("Could not read PDF metadata from %s: %s", pdf_path, exc)
    return None


def _render_page(args: Tuple[str, int, int, str]) -> Path:
    """Render a single PDF page to a PNG image.

    Designed to run inside a thread pool.  PyMuPDF releases the GIL during
    pixmap creation so threads provide genuine parallelism here.
    """
    import fitz  # type: ignore

    pdf_path_str, page_idx, dpi, image_dir_str = args
    image_dir = Path(image_dir_str)
    pdf_stem = Path(pdf_path_str).stem
    image_path = image_dir / f"{pdf_stem}-page-{page_idx:04d}.png"

    doc = fitz.open(pdf_path_str)
    try:
        page = doc[page_idx - 1]
        page.get_pixmap(dpi=dpi, alpha=False).save(str(image_path))
    finally:
        doc.close()
    return image_path


def _save_ocr_json(raw: Any, dest: Path) -> None:
    """Write OCR raw result to JSON (runs in background thread)."""
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2, default=str)


_MAX_RENDER_WORKERS = 4


def _extract_pdf_with_paddleocrvl(pdf_path: str, output_dir: str, model_dir: Path, save_intermediate: bool = True) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """Render PDF pages, run OCR, and return merged text + metadata + structured items.

    Page rendering is parallelised across threads (PyMuPDF releases the GIL).
    OCR remains serial to avoid GPU memory contention.  Intermediate JSON
    writes are submitted to a background thread pool so they don't block the
    main OCR loop.
    """
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for PaddleOCR parsing. Install with: pip install pymupdf"
        ) from exc

    pdf = Path(pdf_path)
    out_dir = Path(output_dir)
    temp_dir = None
    if save_intermediate:
        image_dir = out_dir / "page_images"
        raw_dir = out_dir / "ocr_raw"
        image_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.TemporaryDirectory()
        image_dir = Path(temp_dir.name)
        raw_dir = None

    engine, backend = _create_ocr_engine(model_dir=model_dir)

    _HEADER_LINES = 5
    _FOOTER_LINES = 3

    # --- Phase 1: parallel page rendering ---
    doc = fitz.open(str(pdf))
    total_pages = len(doc)
    first_page_pymupdf_text = doc[0].get_text("text") if total_pages > 0 else ""
    doc.close()

    render_args = [
        (str(pdf), idx, 300, str(image_dir)) for idx in range(1, total_pages + 1)
    ]
    num_workers = min(_MAX_RENDER_WORKERS, max(1, os.cpu_count() or 1), total_pages)
    with ThreadPoolExecutor(max_workers=num_workers) as render_pool:
        image_paths: List[Path] = list(render_pool.map(_render_page, render_args))

    # --- Phase 2: serial OCR + background JSON writes ---
    page_blocks: List[str] = []
    page_level_meta: List[Dict[str, Any]] = []
    ocr_items: List[Dict[str, Any]] = []
    first_page_full_text = first_page_pymupdf_text

    bg_io_pool = ThreadPoolExecutor(max_workers=2)
    bg_futures: List[Future] = []

    try:
        vl_results: List[Any] = []
        for page_idx, image_path in enumerate(image_paths, start=1):
            raw = _run_ocr(engine, image_path)

            if raw_dir is not None:
                dest = raw_dir / f"page-{page_idx:04d}.json"
                bg_futures.append(bg_io_pool.submit(_save_ocr_json, raw, dest))

            if backend == "paddleocrvl":
                if isinstance(raw, list):
                    vl_results.extend(raw)
                else:
                    vl_results.append(raw)
            else:
                lines: List[str] = []
                _collect_text(raw, lines)
                lines = _normalize_lines(lines)

                if not lines:
                    doc = fitz.open(str(pdf))
                    try:
                        fallback = doc[page_idx - 1].get_text("text")
                    finally:
                        doc.close()
                    lines = [x.strip() for x in fallback.splitlines() if x.strip()]

                page_text = "\n".join(lines).strip()
                page_text = _convert_html_to_markdown(page_text)
                page_blocks.append(f"## Page {page_idx}\n\n{page_text}")

                page_level_meta.append({
                    "page": page_idx,
                    "header_text": "\n".join(lines[:_HEADER_LINES]),
                    "footer_text": "\n".join(lines[-_FOOTER_LINES:]) if len(lines) >= _FOOTER_LINES else "",
                    "line_count": len(lines),
                })

            if page_idx == 1:
                if backend == "paddleocrvl":
                    first_page_full_text = first_page_pymupdf_text
                else:
                    first_page_full_text = page_text + "\n" + first_page_pymupdf_text
    finally:
        if temp_dir is not None:
            # Wait for background writes before cleaning up temp dir
            for fut in bg_futures:
                fut.result()
            temp_dir.cleanup()

    # --- Phase 3: PaddleOCR-VL batch post-processing ---
    if backend == "paddleocrvl" and vl_results:
        try:
            restructured = engine.restructure_pages(
                vl_results,
                merge_tables=True,
                relevel_titles=True,
                concatenate_pages=True,
            )
            restructured = list(restructured)
        except (RuntimeError, ValueError, TypeError) as exc:
            logger.warning("restructure_pages failed for %s: %s", pdf, exc, exc_info=True)
            restructured = []

        for idx, res in enumerate(restructured, start=1):
            try:
                md_info = res._to_markdown(pretty=True, show_formula_number=False)
                page_text = md_info.get("markdown_texts", "")
            except (AttributeError, TypeError, KeyError) as exc:
                logger.warning("_to_markdown failed for page %d of %s: %s", idx, pdf, exc)
                page_text = ""
            page_text = _convert_html_to_markdown(page_text)
            if page_text:
                page_blocks.append(page_text)

            lines = [ln for ln in page_text.splitlines() if ln.strip()]
            page_level_meta.append({
                "page": idx,
                "header_text": "\n".join(lines[:_HEADER_LINES]),
                "footer_text": "\n".join(lines[-_FOOTER_LINES:]) if len(lines) >= _FOOTER_LINES else "",
                "line_count": len(lines),
            })

            try:
                blocks = res["parsing_res_list"]
            except (KeyError, TypeError, IndexError):
                blocks = getattr(res, "parsing_res_list", [])
            for block in blocks or []:
                item = _block_to_item(block)
                if item:
                    item["page"] = idx
                    ocr_items.append(item)

    # Wait for any remaining background writes
    for fut in bg_futures:
        fut.result()
    bg_io_pool.shutdown(wait=False)

    merged = "\n\n".join(page_blocks).strip()

    if not ocr_items and merged:
        ocr_items = _text_to_paragraph_items(merged)

    doi = _extract_doi_from_pdf_metadata(str(pdf))
    if not doi:
        doi = _extract_doi_from_text(first_page_full_text)

    metadata = {
        "backend": backend,
        "model_dir": str(model_dir),
        "pages": len(page_blocks),
        "image_dir": str(image_dir),
        "ocr_raw_dir": str(raw_dir) if raw_dir is not None else None,
        "doi": doi,
        "page_level_metadata": page_level_meta,
        "ocr_items": len(ocr_items),
    }
    return merged, metadata, ocr_items

def _read_txt_file(path: Path) -> str:
    """Read plain text file with robust fallback encoding handling."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def parse_pdf_with_paddleocrvl(state: KnowMatState) -> dict:
    """Parse PDF/TXT and return cleaned paper text + document metadata.

    Returns
    -------
    dict
        ``paper_text``: cleaned markdown of the paper body (references stripped).
        ``document_metadata``: dict with at least ``{"doi": str|None}``.
    """
    input_path = state.get("pdf_path")
    if not input_path:
        raise ValueError("No input file path provided in state for parse_pdf_with_paddleocrvl node.")
    save_intermediate = bool(state.get("save_intermediate", True))
    output_dir = state.get("output_dir", ".")
    source_path = Path(input_path)
    suffix = source_path.suffix.lower()

    if suffix in (".txt", ".md"):
        parse_output_dir = Path(output_dir) / "txt_parse" if save_intermediate else Path(output_dir)
        if save_intermediate:
            parse_output_dir.mkdir(parents=True, exist_ok=True)

        raw_text = _read_txt_file(source_path)
        # Convert any residual HTML to clean markdown, then structure sections
        md_text = _convert_html_to_markdown(raw_text)
        md_text = _structure_sections(md_text)
        cleaned_text = _strip_references_section(md_text)
        stem = source_path.stem

        # Try to extract DOI from the text itself (first ~5000 chars)
        doi = _extract_doi_from_text(cleaned_text[:5000])

        # Inject DOI at the top so the LLM can see it
        if doi and doi not in cleaned_text:
            cleaned_text = f"DOI: {doi}\n\n{cleaned_text}"

        if save_intermediate:
            final_md_path = parse_output_dir / f"{stem}_final_output.md"
            with open(final_md_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)
            print(f"Saved txt parsed output to: {final_md_path}")

        doc_meta: Dict[str, Any] = {
            "backend": "txt-direct",
            "source_file": str(source_path),
            "doi": doi,
        }
        if save_intermediate:
            meta_path = parse_output_dir / f"{stem}_parse_metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(doc_meta, f, ensure_ascii=False, indent=2)
            print(f"Saved parser metadata to: {meta_path}")

        return {"paper_text": cleaned_text, "document_metadata": doc_meta, "ocr_items": _text_to_paragraph_items(cleaned_text)}

    if suffix != ".pdf" and suffix not in (".txt", ".md"):
        raise ValueError(f"Unsupported file type: {source_path.suffix}. Only .pdf, .txt, and .md are supported.")
    parse_output_dir = Path(output_dir) / "paddleocrvl_parse" if save_intermediate else Path(output_dir)
    if save_intermediate:
        parse_output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = _default_model_dir()

    try:
        extracted_text, metadata, ocr_items = _extract_pdf_with_paddleocrvl(
            str(source_path),
            str(parse_output_dir),
            model_dir,
            save_intermediate=save_intermediate,
        )
        structured_text = _structure_sections(extracted_text)
        cleaned_text = _strip_references_section(structured_text)

        # Inject DOI at the top of paper_text so the LLM can also see it
        doi_from_ocr = metadata.get("doi")
        if doi_from_ocr and doi_from_ocr not in cleaned_text:
            cleaned_text = f"DOI: {doi_from_ocr}\n\n{cleaned_text}"

        pdf_name = source_path.stem
        if save_intermediate:
            final_md_path = parse_output_dir / f"{pdf_name}_final_output.md"
            with open(final_md_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)
            print(f"Saved final markdown output to: {final_md_path}")

        doc_meta = {
            "backend": metadata.get("backend", "paddleocrvl"),
            "model_dir": metadata.get("model_dir"),
            "pages": metadata.get("pages"),
            "doi": doi_from_ocr,
            "page_level_metadata": metadata.get("page_level_metadata"),
        }

        if save_intermediate:
            meta_path = parse_output_dir / f"{pdf_name}_parse_metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            print(f"Saved parser metadata to: {meta_path}")

        return {"paper_text": cleaned_text, "document_metadata": doc_meta, "ocr_items": _text_to_paragraph_items(cleaned_text)}
    except Exception as exc:
        raise RuntimeError(f"Failed to parse PDF with PaddleOCR-VL: {str(exc)}") from exc






