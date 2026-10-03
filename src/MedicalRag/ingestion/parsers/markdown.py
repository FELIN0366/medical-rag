"""支持标题、表格与代码围栏识别的本地 Markdown Parser。"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ..models import ParsedDocument, Section
from ..tree import append_section

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def parse_markdown(path: Path, *, source_name: str | None = None) -> ParsedDocument:
    text = path.read_text(encoding="utf-8", errors="replace")
    roots: list[Section] = []
    stack: list[Section] = []
    current: Section | None = None
    buffer: list[str] = []
    in_code = False

    def flush() -> None:
        nonlocal buffer
        if buffer:
            if current is None:
                section = Section(level=1, title="正文", text="\n".join(buffer).strip())
                append_section(roots, stack, section)
            else:
                current.text = "\n".join((current.text, "\n".join(buffer))).strip()
            buffer = []

    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.strip().startswith("```") or line.strip().startswith("~~~"):
            in_code = not in_code
            buffer.append(line)
            index += 1
            continue
        heading = None if in_code else HEADING.match(line)
        if heading:
            flush()
            current = Section(level=len(heading.group(1)), title=heading.group(2).strip())
            append_section(roots, stack, current)
            index += 1
            continue
        if not in_code and line.lstrip().startswith("|"):
            flush()
            table = [line]
            index += 1
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                table.append(lines[index])
                index += 1
            table_section = Section(level=(current.level + 1 if current else 1), title="表格", text="\n".join(table), is_table=True)
            append_section(roots, stack, table_section)
            continue
        buffer.append(line)
        index += 1
    flush()
    title = next((s.title for s in roots if s.title != "正文"), path.stem)
    return ParsedDocument(
        doc_id=hashlib.sha256(path.read_bytes()).hexdigest(), title=title, source_name=source_name or path.stem,
        sections=roots,
    )
