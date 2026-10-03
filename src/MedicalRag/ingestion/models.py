"""Stage 1 使用的轻量、与存储无关的中间表示。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Section(BaseModel):
    level: int
    title: str
    text: str = ""
    page: int | None = None
    is_table: bool = False
    children: list["Section"] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    doc_id: str
    title: str
    source: str = "literature"
    source_name: str
    department: str = ""
    sections: list[Section] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    pk: str
    doc_id: str
    chunk_id: int
    source: str
    source_name: str
    department: str = ""
    title: str = ""
    section_path: str = ""
    page: int | None = None
    summary: str
    document: str
    text: str


Section.model_rebuild()
