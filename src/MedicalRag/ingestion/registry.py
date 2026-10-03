"""仅发现 Stage 1 明确支持的本地数据源。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceFile:
    path: Path
    kind: str


def scan_data(data_dir: Path) -> tuple[list[SourceFile], list[dict]]:
    supported: list[SourceFile] = []
    inventory: list[dict] = []
    for path in sorted(candidate for candidate in data_dir.rglob("*") if candidate.is_file()):
        relative = path.relative_to(data_dir).as_posix()
        suffix = path.suffix.lower()
        # 数据目录说明和评测集不是医学语料，但仍必须出现在 inventory 中。
        if relative == "README.md" or relative.startswith("eval/"):
            kind, status = "non_corpus", "ignored_non_corpus"
        # 已知 HuatuoQA 是独立 QA 入口，不属于通用 JSON Parser。
        elif path.name == "qa_50000.jsonl":
            kind, status = "qa", "supported"
        elif suffix == ".pdf":
            kind, status = "pdf", "supported"
        elif suffix in {".html", ".htm"}:
            kind, status = "html", "supported"
        elif suffix in {".md", ".markdown"}:
            kind, status = "markdown", "supported"
        else:
            kind, status = "unsupported", "ignored_unsupported"
        record = {"path": relative, "kind": kind, "status": status, "bytes": path.stat().st_size}
        inventory.append(record)
        if status == "supported":
            supported.append(SourceFile(path, kind))
    return supported, inventory
