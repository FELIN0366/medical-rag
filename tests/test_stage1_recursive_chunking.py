from MedicalRag.ingestion.chunking import chunk_document
from MedicalRag.ingestion.models import ParsedDocument, Section


def _document(text: str) -> ParsedDocument:
    return ParsedDocument(
        doc_id="recursive-test",
        title="递归切分测试",
        source_name="test",
        sections=[Section(level=1, title="正文", text=text)],
    )


def test_recursive_chunks_descend_separator_priority() -> None:
    text = "甲" * 350 + "，" + "乙" * 350 + "\n\n" + "丙" * 350 + "；" + "丁" * 350
    chunks = chunk_document(_document(text), strategy="recursive", max_chars=600, overlap=80)
    assert len(chunks) > 1
    assert all(len(chunk.document) <= 600 for chunk in chunks)


def test_recursive_chunks_hard_split_unbroken_text() -> None:
    chunks = chunk_document(_document("甲" * 1400), strategy="recursive", max_chars=600, overlap=80)
    assert len(chunks) == 3
    assert all(len(chunk.document) <= 600 for chunk in chunks)
