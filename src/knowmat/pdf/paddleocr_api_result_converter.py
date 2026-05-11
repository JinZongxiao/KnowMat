"""Convert PaddleOCR cloud API JSONL output to KnowMat internal format."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

_FORMULA_PATTERN = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE_FORMULA_PATTERN = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")
_IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_TABLE_LINE_PATTERN = re.compile(r"^\|.+\|$")
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")


def _parse_markdown_blocks(
    markdown_text: str,
    page_num: int,
    images_dir: Optional[Path],
    image_urls: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Parse a page's markdown into structured ocr_items."""
    items: List[Dict[str, Any]] = []
    lines = markdown_text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]

        # Display formula block ($$...$$)
        if line.strip().startswith("$$"):
            formula_lines = [line]
            if not line.strip().endswith("$$") or line.strip() == "$$":
                i += 1
                while i < len(lines):
                    formula_lines.append(lines[i])
                    if lines[i].strip().endswith("$$"):
                        i += 1
                        break
                    i += 1
            else:
                i += 1
            formula_text = "\n".join(formula_lines)
            latex = formula_text.strip().strip("$").strip()
            if latex:
                items.append({
                    "typer": "formula",
                    "data": {"text": latex, "latex": latex},
                    "page": page_num,
                    "block_label": "formula",
                })
            continue

        # Table block (consecutive | lines)
        if _TABLE_LINE_PATTERN.match(line.strip()):
            table_lines = []
            while i < len(lines) and _TABLE_LINE_PATTERN.match(lines[i].strip()):
                table_lines.append(lines[i])
                i += 1
            table_md = "\n".join(table_lines)
            if table_md.strip():
                items.append({
                    "typer": "table",
                    "data": {"text": table_md, "raw_html": ""},
                    "page": page_num,
                    "block_label": "table",
                })
            continue

        # Image reference
        img_match = _IMAGE_PATTERN.match(line.strip())
        if img_match:
            caption = img_match.group(1)
            img_ref = img_match.group(2)
            resolved_path = ""
            if image_urls and img_ref in image_urls and images_dir:
                img_url = image_urls[img_ref]
                local_name = Path(img_ref).name or f"page{page_num}_img.jpg"
                local_path = images_dir / local_name
                if not local_path.exists():
                    try:
                        resp = requests.get(img_url, timeout=60)
                        resp.raise_for_status()
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        local_path.write_bytes(resp.content)
                        resolved_path = str(local_path)
                    except Exception as exc:
                        logger.warning("Failed to download image %s: %s", img_ref, exc)
                else:
                    resolved_path = str(local_path)
            items.append({
                "typer": "image",
                "data": {"image_path": resolved_path, "caption": caption},
                "page": page_num,
                "block_label": "figure",
            })
            i += 1
            continue

        # Heading
        heading_match = _HEADING_PATTERN.match(line.strip())
        if heading_match:
            text = heading_match.group(2).strip()
            if text:
                items.append({
                    "typer": "paragraph",
                    "text": text,
                    "page": page_num,
                    "block_label": "title",
                })
            i += 1
            continue

        # Regular paragraph (collect consecutive non-empty non-special lines)
        para_lines = []
        while i < len(lines):
            curr = lines[i]
            if not curr.strip():
                i += 1
                break
            if curr.strip().startswith("$$"):
                break
            if _TABLE_LINE_PATTERN.match(curr.strip()):
                break
            if _IMAGE_PATTERN.match(curr.strip()):
                break
            if _HEADING_PATTERN.match(curr.strip()):
                break
            para_lines.append(curr)
            i += 1

        text = " ".join(para_lines).strip()
        if text:
            # Check for inline formulas and mark them
            items.append({
                "typer": "paragraph",
                "text": text,
                "page": page_num,
            })
        elif not para_lines:
            i += 1

    return items


def convert_paddleocr_api_to_knowmat(
    pages_data: List[Dict[str, Any]],
    pdf_path: str,
    images_dir: Optional[Path] = None,
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """Convert PaddleOCR API JSONL page results to KnowMat format.

    Parameters
    ----------
    pages_data : list
        Parsed JSONL lines from PaddleOCR API (one dict per JSONL line).
    pdf_path : str
        Path to the source PDF.
    images_dir : Path, optional
        Directory to save downloaded images.

    Returns
    -------
    tuple of (extracted_text, metadata, ocr_items)
    """
    all_ocr_items: List[Dict[str, Any]] = []
    page_texts: List[str] = []
    total_pages = 0

    for line_idx, page_data in enumerate(pages_data):
        result = page_data.get("result", {})
        layout_results = result.get("layoutParsingResults", [])

        for layout_res in layout_results:
            total_pages += 1
            page_num = total_pages

            markdown_info = layout_res.get("markdown", {})
            md_text = markdown_info.get("text", "")
            image_urls = markdown_info.get("images", {})

            if md_text.strip():
                page_texts.append(md_text)

            page_items = _parse_markdown_blocks(
                md_text, page_num, images_dir, image_urls
            )
            all_ocr_items.extend(page_items)

            # Handle outputImages (rendered layout images)
            output_images = layout_res.get("outputImages", {})
            if output_images and images_dir:
                for img_name, img_url in output_images.items():
                    local_path = images_dir / f"{img_name}_{page_num}.jpg"
                    if not local_path.exists():
                        try:
                            resp = requests.get(img_url, timeout=60)
                            if resp.status_code == 200:
                                local_path.parent.mkdir(parents=True, exist_ok=True)
                                local_path.write_bytes(resp.content)
                        except Exception as exc:
                            logger.debug("Failed to download outputImage %s: %s", img_name, exc)

    extracted_text = "\n\n".join(page_texts)

    table_count = sum(1 for it in all_ocr_items if it.get("typer") == "table")
    formula_count = sum(1 for it in all_ocr_items if it.get("typer") == "formula")
    image_count = sum(1 for it in all_ocr_items if it.get("typer") == "image")

    metadata: Dict[str, Any] = {
        "backend": "paddleocr_api",
        "pages": total_pages,
        "ocr_items": len(all_ocr_items),
        "ocr_quality": {
            "ocr_avg_confidence": None,
            "ocr_low_confidence_pages": [],
            "table_count": table_count,
            "formula_count": formula_count,
            "image_count": image_count,
            "ppstructure_status": "not_applicable",
            "ppstructure_detail": "parsed by PaddleOCR cloud API",
            "ppstructure_replacements": 0,
        },
    }

    return extracted_text, metadata, all_ocr_items


def extract_formulas_per_page(
    pages_data: List[Dict[str, Any]],
) -> Dict[int, List[str]]:
    """Extract formula LaTeX strings per page from PP-StructureV3 API result.

    Returns dict of {page_number: [latex1, latex2, ...]} in reading order.
    Used for PP-StructureV3 formula refinement on other backends (MinerU, etc.).
    """
    formulas_by_page: Dict[int, List[str]] = {}
    page_num = 0

    for page_data in pages_data:
        result = page_data.get("result", {})
        layout_results = result.get("layoutParsingResults", [])

        for layout_res in layout_results:
            page_num += 1
            md_text = layout_res.get("markdown", {}).get("text", "")

            page_formulas: List[str] = []
            # Find display formulas
            for match in _FORMULA_PATTERN.finditer(md_text):
                latex = match.group(1).strip()
                if latex:
                    page_formulas.append(latex)

            if page_formulas:
                formulas_by_page[page_num] = page_formulas

    return formulas_by_page


def extract_tables_per_page(
    pages_data: List[Dict[str, Any]],
) -> Dict[int, List[str]]:
    """Extract markdown table strings per page from PP-StructureV3 API result.

    Returns dict of {page_number: [table_md_1, table_md_2, ...]} in reading order.
    """
    tables_by_page: Dict[int, List[str]] = {}
    page_num = 0

    for page_data in pages_data:
        result = page_data.get("result", {})
        layout_results = result.get("layoutParsingResults", [])

        for layout_res in layout_results:
            page_num += 1
            md_text = layout_res.get("markdown", {}).get("text", "")

            page_tables: List[str] = []
            lines = md_text.split("\n")
            i = 0
            while i < len(lines):
                if _TABLE_LINE_PATTERN.match(lines[i].strip()):
                    table_lines = []
                    while i < len(lines) and _TABLE_LINE_PATTERN.match(lines[i].strip()):
                        table_lines.append(lines[i])
                        i += 1
                    table_md = "\n".join(table_lines)
                    if table_md.strip():
                        page_tables.append(table_md)
                else:
                    i += 1

            if page_tables:
                tables_by_page[page_num] = page_tables

    return tables_by_page
