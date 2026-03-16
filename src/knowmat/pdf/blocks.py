"""Block adaptation helpers for PaddleOCR-VL outputs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .html_cleaner import convert_html_to_markdown, html_table_to_structured


def _get_block_attr(block: Any, name: str, default: Any = None) -> Any:
    if hasattr(block, name):
        return getattr(block, name)
    if isinstance(block, dict):
        return block.get(name, default)
    return default


def block_to_item(block: Any) -> Optional[Dict[str, Any]]:
    label = _get_block_attr(block, "label", None) or _get_block_attr(block, "block_label", None)
    content = _get_block_attr(block, "content", "") or _get_block_attr(block, "text", "")
    confidence = _get_block_attr(block, "score", None) or _get_block_attr(block, "confidence", None)

    if isinstance(content, str):
        content = content.strip()

    if label in ("table", "chart"):
        structured = html_table_to_structured(content or "")
        item: Dict[str, Any] = {"typer": "table"}
        if structured:
            item.update(structured)
        else:
            item["text"] = convert_html_to_markdown(content or "")
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
        item = {"typer": "paragraph", "text": convert_html_to_markdown(content)}
        if confidence is not None:
            item["confidence"] = confidence
        return item
    return None


def text_to_paragraph_items(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    return [{"typer": "paragraph", "text": block} for block in blocks]

