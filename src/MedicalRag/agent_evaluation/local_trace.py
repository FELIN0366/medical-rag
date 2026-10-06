"""只用于 Stage 4 Eval 的内存轨迹收集器，不连接任何云端平台。"""
from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _safe_value(value: Any) -> Any:
    """轨迹只保留行为元数据，避免写入检索 chunk 或完整模型上下文。"""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:500]
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value[:20]]
    if isinstance(value, dict):
        return {str(key): _safe_value(item) for key, item in value.items() if key != "documents"}
    return str(value)[:500]


class LocalTraceCollector:
    """线程安全的 JSONL 轨迹收集器，由 Eval Runner 显式创建与落盘。"""

    def __init__(self, incremental_path: Path | None = None) -> None:
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._incremental_path = incremental_path
        if incremental_path is not None:
            incremental_path.parent.mkdir(parents=True, exist_ok=True)
            incremental_path.write_text("", encoding="utf-8")

    def emit(self, context: dict[str, Any] | None, event: str, **data: Any) -> None:
        if not context:
            return
        row = {
            "timestamp_ms": round(time.time() * 1000),
            "case_id": context.get("case_id", ""),
            "primary_category": context.get("primary_category", ""),
            "session_id": context.get("session_id", ""),
            "turn_index": context.get("turn_index", 0),
            "event": event,
            "data": _safe_value(data),
        }
        with self._lock:
            self._events.append(row)
            if self._incremental_path is not None:
                with self._incremental_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    @contextmanager
    def span(self, context: dict[str, Any] | None, name: str, **data: Any):
        """增量记录 start/end/error 与真实耗时，不写入模型上下文或 chunk。"""
        started = time.perf_counter()
        self.emit(context, "span", name=name, phase="start", **data)
        try:
            yield
        except Exception as error:
            self.emit(context, "span", name=name, phase="error",
                      duration_ms=round((time.perf_counter() - started) * 1000, 3),
                      error_type=type(error).__name__)
            raise
        else:
            self.emit(context, "span", name=name, phase="end",
                      duration_ms=round((time.perf_counter() - started) * 1000, 3))

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            for event in self.events():
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
