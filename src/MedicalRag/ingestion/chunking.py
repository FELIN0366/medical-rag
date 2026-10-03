"""确定性的 structural、fixed 与 recursive Chunker。"""
from __future__ import annotations

import re

from .models import Chunk, ParsedDocument, Section
from .representations import make_literature_chunk
from .tree import flatten_text, walk_sections

SENTENCE_RE = re.compile(r"[^。；！？\n]+[。；！？\n]?", re.S)


def _hard_split(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    result, start = [], 0
    step = max(1, max_chars - overlap)
    while start < len(text):
        result.append(text[start:start + max_chars])
        if start + max_chars >= len(text):
            break
        start += step
    return result


def _sentence_chunks(text: str, max_chars: int, min_chars: int, overlap: int) -> list[str]:
    segments: list[str] = []
    for sentence in SENTENCE_RE.findall(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        segments.extend(_hard_split(sentence, max_chars, overlap) if len(sentence) > max_chars else [sentence])
    if not segments:
        return []
    chunks: list[str] = []
    current = ""
    for segment in segments:
        if current and len(current) + len(segment) > max_chars:
            chunks.append(current)
            current = (current[-overlap:] + segment) if overlap else segment
            if len(current) > max_chars:
                chunks.extend(_hard_split(current, max_chars, overlap)[:-1])
                current = _hard_split(current, max_chars, overlap)[-1]
        else:
            current += segment
    if current:
        if chunks and len(current) < min_chars and len(chunks[-1]) + len(current) <= max_chars:
            chunks[-1] += current
        else:
            chunks.append(current)
    return chunks


def _table_chunks(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return _hard_split(text, max_chars, 0)
    header = "\n".join(lines[:2])
    output, current = [], header
    for line in lines[2:]:
        if len(current) + len(line) + 1 > max_chars and current != header:
            output.append(current)
            current = header + "\n" + line
        else:
            current += "\n" + line
    if current:
        output.append(current)
    return output


RECURSIVE_SEPARATORS = ("\n\n", "\n", "。", "；", "，")


def _recursive_chunks(text: str, max_chars: int, overlap: int, separator_index: int = 0) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    if separator_index >= len(RECURSIVE_SEPARATORS):
        return _hard_split(text, max_chars, overlap)

    separator = RECURSIVE_SEPARATORS[separator_index]
    if separator not in text:
        return _recursive_chunks(text, max_chars, overlap, separator_index + 1)

    parts = text.split(separator)
    pieces = [part + separator for part in parts[:-1]] + [parts[-1]]
    result: list[str] = []
    current = ""
    for piece in pieces:
        if not piece:
            continue
        # 当前层分隔后仍过长时，必须继续使用下一优先级分隔符。
        if len(piece) > max_chars:
            if current:
                result.append(current)
                current = ""
            result.extend(_recursive_chunks(piece, max_chars, overlap, separator_index + 1))
            continue
        if current and len(current) + len(piece) > max_chars:
            result.append(current)
            prefix = current[-overlap:] if overlap else ""
            # overlap 加上新片段仍超长时，优先保证长度不变量。
            current = prefix + piece if len(prefix) + len(piece) <= max_chars else piece
        else:
            current += piece
    if current:
        result.append(current)
    return result


def chunk_document(document: ParsedDocument, strategy: str = "structural", max_chars: int = 600,
                   min_chars: int = 100, overlap: int = 80, table_max_chars: int = 4000) -> list[Chunk]:
    if strategy not in {"structural", "fixed", "recursive"}:
        raise ValueError("strategy must be structural, fixed, or recursive")
    raw_units: list[tuple[str, str, int | None, bool]] = []
    if strategy == "fixed":
        raw_units = [(flatten_text(document.sections), "", None, False)]
    else:
        for section, path in walk_sections(document.sections):
            if section.text.strip():
                raw_units.append((section.text.strip(), path, section.page, section.is_table))
    chunks: list[Chunk] = []
    for text, path, page, is_table in raw_units:
        if is_table:
            pieces = _table_chunks(text, table_max_chars)
        elif strategy == "fixed":
            pieces = _hard_split(text, max_chars, overlap)
        elif strategy == "recursive":
            pieces = _recursive_chunks(text, max_chars, overlap)
        else:
            pieces = _sentence_chunks(text, max_chars, min_chars, overlap)
        for piece in pieces:
            if piece.strip():
                chunks.append(make_literature_chunk(
                    doc_id=document.doc_id, chunk_id=len(chunks), source_name=document.source_name,
                    department=document.department, title=document.title, section_path=path,
                    page=page, document=piece,
                ))
    return chunks
