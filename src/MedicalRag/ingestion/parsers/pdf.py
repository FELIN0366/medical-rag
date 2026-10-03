"""使用 PyMuPDF 的 PDF Parser，支持标题、双栏、表格与扫描件 warning。"""
from __future__ import annotations

import hashlib
import statistics
from pathlib import Path

import fitz

from ..models import ParsedDocument, Section
from ..tree import append_section


def _table_markdown(table) -> str:
    rows = table.extract() or []
    rows = [["" if value is None else str(value).replace("\n", " ") for value in row] for row in rows]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header = normalized[0]
    return "\n".join(["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"] + ["| " + " | ".join(row) + " |" for row in normalized[1:]])


def parse_pdf(path: Path, *, source_name: str | None = None) -> ParsedDocument:
    roots: list[Section] = []
    stack: list[Section] = []
    warnings: list[str] = []
    document = fitz.open(path)
    title = (document.metadata or {}).get("title") or path.stem
    body_sizes: list[float] = []
    page_lines: list[list[tuple[float, float, float, str, float]]] = []
    page_tables: list[list[tuple[fitz.Rect, str]]] = []
    for page in document:
        try:
            found_tables = page.find_tables().tables
        except Exception as exc:
            warnings.append(f"page {page.number + 1}: table detection failed ({exc})")
            found_tables = []
        tables = [(table.bbox, _table_markdown(table)) for table in found_tables]
        page_tables.append(tables)
        lines: list[tuple[float, float, float, str, float]] = []
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(span.get("text", "") for span in spans).strip()
                if not text:
                    continue
                bbox = fitz.Rect(line["bbox"])
                if any(bbox.intersects(fitz.Rect(table_bbox)) for table_bbox, _ in tables):
                    continue
                size = max((float(span.get("size", 0)) for span in spans), default=0.0)
                if size:
                    body_sizes.append(size)
                lines.append((bbox.y0, bbox.x0, bbox.width, text, size))
        page_lines.append(lines)
    body_size = statistics.median(body_sizes) if body_sizes else 0.0
    if not body_size:
        warnings.append("no text layer detected; suspected scanned PDF (OCR is not enabled in Stage 1)")
    for page_number, (page, lines, tables) in enumerate(zip(document, page_lines, page_tables), start=1):
        if not lines and not tables:
            warnings.append(f"page {page_number}: suspected scanned or empty page")
            continue
        widths = [line[2] for line in lines]
        is_two_column = bool(widths and statistics.median(widths) < 0.62 * page.rect.width)
        if is_two_column:
            midpoint = page.rect.width / 2
            lines.sort(key=lambda item: (0 if item[1] < midpoint else 1, item[0], item[1]))
        else:
            lines.sort(key=lambda item: (item[0], item[1]))
        for _, _, _, text, size in lines:
            if body_size and size >= body_size * 1.15 and len(text) <= 120:
                level = 1 if size >= body_size * 1.45 else 2 if size >= body_size * 1.28 else 3
                section = Section(level=level, title=text, page=page_number)
                append_section(roots, stack, section)
            else:
                if stack:
                    stack[-1].text = "\n".join((stack[-1].text, text)).strip()
                else:
                    section = Section(level=1, title=f"第 {page_number} 页", text=text, page=page_number)
                    append_section(roots, stack, section)
        for _, table_text in tables:
            if table_text:
                section = Section(level=(stack[-1].level + 1 if stack else 1), title="表格", text=table_text, page=page_number, is_table=True)
                append_section(roots, stack, section)
    document.close()
    return ParsedDocument(
        doc_id=hashlib.sha256(path.read_bytes()).hexdigest(), title=title, source_name=source_name or path.stem,
        sections=roots, warnings=warnings,
    )
