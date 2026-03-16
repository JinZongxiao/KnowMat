"""Section/title normalization and noise filtering for OCR text."""

from __future__ import annotations

import re
from typing import List, Tuple

from .doi_extractor import DOI_RE, DOI_URL_RE

ALLOY_OCR_FIXES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bNb15\s*Ta1[oO]\s*W75\b"), "Nb15Ta10W75"),
    (re.compile(r"\bNb1[sS]\s*Ta10\s*W75\b"), "Nb15Ta10W75"),
    (re.compile(r"\bNb15\s*Ta1[oO]0\s*W75\b"), "Nb15Ta10W75"),
    (re.compile(r"\bNb15\s+Ta10\s*W75\b"), "Nb15Ta10W75"),
    (re.compile(r"\bNb15\s*Ta10\s+W75\b"), "Nb15Ta10W75"),
    (re.compile(r"\bNb15\s*Ta10\s*W\s*,"), "Nb15Ta10W75,"),
    (re.compile(r"\(Nb15\s*Ta1[oO]\s*W75\)"), "(Nb15Ta10W75)"),
    (re.compile(r"\(Nb15\s*Ta10\s+W75\)"), "(Nb15Ta10W75)"),
    (re.compile(r"\(Nb15\s+Ta10\s*W75\)"), "(Nb15Ta10W75)"),
]

SPACED_TITLE_RE = re.compile(r"^(#+\s+)?([A-Z](?:\s[A-Z]){3,})$")

SECTION_PATTERNS: List[Tuple[re.Pattern, str | None]] = [
    (re.compile(r"^(?:#+\s*)?A\s*B\s*S\s*T\s*R\s*A\s*C\s*T\s*$", re.IGNORECASE), "## ABSTRACT"),
    (re.compile(r"^(?:#+\s*)?ABSTRACT\s*$", re.IGNORECASE), "## ABSTRACT"),
    (re.compile(r"^(?:#+\s*)?ARTICLE\s+INFO\s*$", re.IGNORECASE), "## ARTICLE INFO"),
    (re.compile(r"^(?:#+\s*)?KEYWORDS?\s*:?\s*$", re.IGNORECASE), "## Keywords"),
    (re.compile(r"^(?:#+\s*)?(ACKNOWLEDGEMENTS?|ACKNOWLEDGMENTS?)\s*$", re.IGNORECASE), "## Acknowledgements"),
    (re.compile(r"^(?:#+\s*)?CRediT\s+authorship\s+contribution\s+statement\s*$", re.IGNORECASE), "## CRediT Authorship"),
    (re.compile(r"^(?:#+\s*)?DECLARATION\s+OF\s+(COMPETING\s+)?INTEREST.*$", re.IGNORECASE), "## Declaration of Interest"),
    (re.compile(r"^(?:#+\s*)?DATA\s+AVAILABILITY.*$", re.IGNORECASE), "## Data Availability"),
    (re.compile(r"^(?:#+\s*)?SUPPLEMENTARY\s+(DATA|MATERIAL|INFORMATION).*$", re.IGNORECASE), "## Supplementary Material"),
    (re.compile(r"^(?:#+\s*)?(\d+(?:\.\d+)*\.?)\s+(.+)$"), None),
]

NOISE_LINE_PATTERNS: List[Tuple[re.Pattern, bool]] = [
    (re.compile(r"^Materials Science and Engineering\s+\d", re.IGNORECASE), True),
    (re.compile(r"^Acta Materialia\s+\d", re.IGNORECASE), True),
    (re.compile(r"^Journal of Materials Science\s+\d", re.IGNORECASE), True),
    (re.compile(r"^International Journal of Plasticity\s+\d", re.IGNORECASE), True),
    (re.compile(r"^Journal of Alloys and Compounds\s+\d", re.IGNORECASE), True),
    (re.compile(r"^Scripta Materialia\s+\d", re.IGNORECASE), True),
    (re.compile(r"^Additive Manufacturing\s+\d", re.IGNORECASE), True),
    (re.compile(r"^Contents lists available at ScienceDirect", re.IGNORECASE), False),
    (re.compile(r"^Available online", re.IGNORECASE), False),
    (re.compile(r"^Received \d+", re.IGNORECASE), False),
    (re.compile(r"^Accepted \d+", re.IGNORECASE), False),
    (re.compile(r"^journal homepage", re.IGNORECASE), False),
    (re.compile(r"^https?://www\.(sciencedirect|elsevier|springer|wiley)", re.IGNORECASE), False),
    (re.compile(r"^Full length article\s*$", re.IGNORECASE), False),
    (re.compile(r"^\d{4}-\d{3}[\dX]/\s*�", re.IGNORECASE), False),
    (re.compile(r"^�\s*\d{4}", re.IGNORECASE), False),
    (re.compile(r"^Elsevier", re.IGNORECASE), False),
    (re.compile(r"^\* Corresponding author", re.IGNORECASE), False),
    (re.compile(r"^\*\* Corresponding author", re.IGNORECASE), False),
    (re.compile(r"^E-mail address", re.IGNORECASE), False),
    (re.compile(r"^\d+$"), False),
    (re.compile(r"^[A-Za-z]:\\"), False),
    (re.compile(r"^/tmp/"), False),
    (re.compile(r"^min\s*$", re.IGNORECASE), False),
    (re.compile(r"^general\s*$", re.IGNORECASE), False),
    (re.compile(r"^(STRUCTURE|PROPERTIES|MODELLING|MODELIINC|PROCESSING)\s*$"), False),
    (re.compile(r"^Check\s+for\s*$", re.IGNORECASE), False),
    (re.compile(r"^updates?\s*$", re.IGNORECASE), False),
    (re.compile(r"^Acta\s+Materialia\s*$", re.IGNORECASE), False),
    (re.compile(r"^Actd\s+Materialia\b", re.IGNORECASE), False),
    (re.compile(r"^Materialia\s*$", re.IGNORECASE), False),
    (re.compile(r"^Acta\s*$", re.IGNORECASE), False),
    (re.compile(r"^[A-Z]\.\s+\w+\s+et\s+al\.?\s*$", re.IGNORECASE), False),
]


def normalize_alloy_strings(text: str) -> str:
    for pat, repl in ALLOY_OCR_FIXES:
        text = pat.sub(repl, text)
    return text


def is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    numeric_unit = re.sub(r"^#+\s*", "", stripped)
    if re.match(r"^\d{1,4}(?:\.\d+)?\s*(um|nm|pm|°C|C)\b", numeric_unit):
        return True
    if re.match(r"^\d{1,4}(?:\.\d+)?\s*(?:°C|C)\s*[~\-]\s*\d{1,4}(?:\.\d+)?\s*(?:°C|C)\b", numeric_unit):
        return True

    has_doi = bool(DOI_RE.search(stripped)) or bool(DOI_URL_RE.search(stripped))
    for pat, skip_if_doi in NOISE_LINE_PATTERNS:
        if pat.match(stripped):
            if skip_if_doi and has_doi:
                return False
            return True
    return False


def structure_sections(text: str) -> str:
    output_lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if is_noise_line(stripped):
            continue
        matched = False
        for pat, replacement in SECTION_PATTERNS:
            m = pat.match(stripped)
            if m:
                if replacement:
                    output_lines.append("")
                    output_lines.append(replacement)
                    output_lines.append("")
                else:
                    sec_num = m.group(1).rstrip(".")
                    sec_title = m.group(2).strip()
                    output_lines.append("")
                    output_lines.append(f"## {sec_num}. {sec_title}")
                    output_lines.append("")
                matched = True
                break
        if matched:
            continue
        m = SPACED_TITLE_RE.match(stripped)
        if m:
            collapsed = m.group(2).replace(" ", "")
            output_lines.append("")
            output_lines.append(f"## {collapsed}")
            output_lines.append("")
            continue
        heading_match = re.match(r"^(#{3,6})\s+(.*)", stripped)
        if heading_match:
            title = heading_match.group(2).strip()
            output_lines.append(f"## {title}")
            continue
        if re.match(r"^##\s+Page\s+\d+\s*$", stripped):
            output_lines.append("")
            continue
        output_lines.append(line)

    result = "\n".join(output_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def strip_references_section(text: str) -> str:
    out: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^#+\s*(references?|bibliography|citations?)\s*$", stripped, re.IGNORECASE):
            break
        if re.match(r"^(references?|bibliography|citations?)\s*$", stripped, re.IGNORECASE):
            break
        out.append(line)
    return "\n".join(out).strip()

