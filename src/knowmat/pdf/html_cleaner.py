"""HTML/table cleanup utilities for OCR output."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
BLOCK_TAGS = [
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


def _html_table_to_markdown(html: str) -> str:
    """Best-effort conversion of a single <table> to markdown table."""
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


def convert_html_to_markdown(text: str) -> str:
    """Convert residual HTML markup in OCR/txt output to clean markdown."""
    if not text:
        return ""
    if "<" not in text and ">" not in text:
        return text.strip()

    from bs4 import BeautifulSoup  # type: ignore

    soup = BeautifulSoup(text, "html.parser")

    for table in soup.find_all("table"):
        md_table = _html_table_to_markdown(str(table))
        table.replace_with(f"\n{md_table}\n")

    for div in soup.find_all("div"):
        if div.find("img") and not div.get_text(strip=True):
            div.decompose()

    for br in soup.find_all("br"):
        br.replace_with("\n")
    for tag in soup.find_all(BLOCK_TAGS):
        if tag.string is None:
            tag.insert_before("\n")
            tag.insert_after("\n")

    converted = soup.get_text("\n")
    converted = HTML_TAG_RE.sub("", converted)
    converted = re.sub(r"\n{3,}", "\n\n", converted)
    return converted.strip()


def html_table_to_structured(html: str) -> Optional[Dict[str, Any]]:
    """Parse a <table> HTML string into structured columns/rows dict."""
    if "<table" not in html.lower():
        return None

    from bs4 import BeautifulSoup  # type: ignore

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return None

    headers: List[str] = []
    thead = table.find("thead")
    if thead:
        header_cells = thead.find_all(["th", "td"])
        headers = [cell.get_text(" ", strip=True) for cell in header_cells if cell]

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
        headers = [f"col_{i + 1}" for i in range(max_len)]

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

