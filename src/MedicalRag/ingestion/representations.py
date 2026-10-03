"""确定性的 Stage 1 检索表示与安全的 Milvus 字符串处理。"""
from __future__ import annotations

import hashlib
import re

from .models import Chunk


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fit_varchar(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def lead_text(text: str, limit: int = 150) -> str:
    first = re.split(r"(?<=[。；！？.!?])|\n", text.strip(), maxsplit=1)[0].strip()
    return (first or text.strip())[:limit]


def make_literature_chunk(*, doc_id: str, chunk_id: int, source_name: str, department: str,
                          title: str, section_path: str, page: int | None, document: str) -> Chunk:
    document = document.strip()
    summary = "\n".join(part for part in (title.strip(), section_path.strip(), lead_text(document)) if part)
    text = f"文档：{title}\n章节：{section_path}\n正文：\n{document}"
    return Chunk(
        pk=stable_hash(f"{doc_id}{chunk_id}{text}"), doc_id=doc_id, chunk_id=chunk_id,
        source="literature", source_name=source_name, department=department, title=title,
        section_path=section_path, page=page, summary=summary, document=document, text=text,
    )


def make_qa_chunk(record: dict, ordinal: int) -> Chunk:
    question = str(record.get("question") or "").strip()
    answer = str(record.get("answer") or "").strip()
    text = f"问题: {question}\n\n答案: {answer}"
    doc_id = stable_hash(f"huatuo_qa:{record.get('question_hash') or question or ordinal}")
    return Chunk(
        pk=stable_hash(f"{doc_id}0{text}"), doc_id=doc_id, chunk_id=0, source="qa",
        source_name="huatuo_qa", department="", title="", section_path="", page=0,
        summary=question, document=answer, text=text,
    )
