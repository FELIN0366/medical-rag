"""与 Parser 解耦的章节树构建与遍历逻辑。"""
from __future__ import annotations

from collections.abc import Iterator

from .models import Section


def append_section(roots: list[Section], stack: list[Section], section: Section) -> None:
    """追加标题、表格或正文 Section，并维护标题层级。"""
    while stack and stack[-1].level >= section.level:
        stack.pop()
    if stack:
        stack[-1].children.append(section)
    else:
        roots.append(section)
    stack.append(section)


def walk_sections(sections: list[Section], ancestors: tuple[str, ...] = ()) -> Iterator[tuple[Section, str]]:
    for section in sections:
        title = section.title.strip() or "正文"
        path = " > ".join((*ancestors, title))
        yield section, path
        yield from walk_sections(section.children, (*ancestors, title))


def flatten_text(sections: list[Section]) -> str:
    return "\n\n".join(section.text for section, _ in walk_sections(sections) if section.text.strip())
