"""构建 Stage 1 语料单元，不产生 embedding 或 Milvus 副作用。"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import chunk_document
from .models import Chunk, ParsedDocument
from .parsers import parse_html, parse_markdown, parse_pdf
from .registry import scan_data
from .representations import make_qa_chunk


@dataclass
class CorpusResult:
    qa_chunks: list[Chunk] = field(default_factory=list)
    literature_chunks: list[Chunk] = field(default_factory=list)
    documents: list[ParsedDocument] = field(default_factory=list)
    inventory: list[dict] = field(default_factory=list)
    parse_failures: list[dict] = field(default_factory=list)

    @property
    def chunks(self) -> list[Chunk]:
        return self.qa_chunks + self.literature_chunks


def sample_huatuoqa(path: Path, sample_size: int = 200, seed: int = 42) -> list[dict]:
    if not 100 <= sample_size <= 300:
        raise ValueError("qa_sample_size must be between 100 and 300")
    records: list[dict] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid HuatuoQA JSON on line {line_number}: {exc}") from exc
            if record.get("question") and record.get("answer"):
                records.append(record)
    if len(records) < sample_size:
        raise ValueError(f"HuatuoQA has only {len(records)} valid records; requested {sample_size}")
    return random.Random(seed).sample(records, sample_size)


def build_corpus(data_dir: Path, *, qa_sample_size: int = 200, seed: int = 42,
                 strategy: str = "structural") -> CorpusResult:
    result = CorpusResult()
    sources, result.inventory = scan_data(data_dir)
    for source in sources:
        if source.kind == "qa":
            sampled = sample_huatuoqa(source.path, qa_sample_size, seed)
            result.qa_chunks = [make_qa_chunk(record, ordinal) for ordinal, record in enumerate(sampled)]
            continue
        try:
            if source.kind == "pdf":
                document = parse_pdf(source.path)
            elif source.kind == "html":
                document = parse_html(source.path)
            elif source.kind == "markdown":
                document = parse_markdown(source.path)
            else:  # defensive: scanner owns the allowed set
                continue
            result.documents.append(document)
            result.literature_chunks.extend(chunk_document(document, strategy=strategy))
        except Exception as exc:
            result.parse_failures.append({"path": source.path.relative_to(data_dir).as_posix(), "error": str(exc)})
    return result
