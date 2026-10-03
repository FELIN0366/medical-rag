"""离线 HTML Parser：trafilatura 提取正文后复用 Markdown Parser。"""
from __future__ import annotations

import hashlib
import re
import tempfile
from html.parser import HTMLParser
from pathlib import Path

import trafilatura

from ..models import ParsedDocument
from .markdown import parse_markdown


class _HeadingCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._level: int | None = None
        self._parts: list[str] = []
        self.headings: list[tuple[int, str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if re.fullmatch(r"h[1-6]", tag.lower()):
            self._level = int(tag[1])
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._level is not None:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._level is not None and tag.lower() == f"h{self._level}":
            title = " ".join("".join(self._parts).split())
            if title:
                self.headings.append((self._level, title))
            self._level = None


def parse_html(path: Path, *, source_name: str | None = None) -> ParsedDocument:
    raw = path.read_text(encoding="utf-8", errors="replace")
    warnings: list[str] = []
    extracted = trafilatura.extract(raw, output_format="markdown", include_tables=True, include_links=False)
    if not extracted or not extracted.strip():
        warnings.append("trafilatura main-content extraction returned no text; using local HTML text fallback")
        extracted = re.sub(r"<[^>]+>", " ", raw)
    collector = _HeadingCollector()
    collector.feed(raw)
    # trafilatura 通常会输出标题；若未输出，则恢复原始层级。
    if collector.headings and not re.search(r"^#{1,6}\s", extracted, flags=re.M):
        prefix = "\n".join("#" * level + " " + title for level, title in collector.headings)
        extracted = prefix + "\n\n" + extracted
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".md", delete=False) as temp:
        temp.write(extracted)
        temp_path = Path(temp.name)
    try:
        parsed = parse_markdown(temp_path, source_name=source_name or path.stem)
    finally:
        temp_path.unlink(missing_ok=True)
    parsed.doc_id = hashlib.sha256(path.read_bytes()).hexdigest()
    parsed.source_name = source_name or path.stem
    parsed.warnings.extend(warnings)
    if not parsed.sections:
        parsed.warnings.append("HTML parser produced no sections")
    return parsed
