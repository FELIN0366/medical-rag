#!/usr/bin/env python
"""执行真实的 Stage 1 Milvus 检索 Smoke Test。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from MedicalRag.config.loader import ConfigLoader
from MedicalRag.config.models import FusionSpec, SearchRequest, SingleSearchRequest
from MedicalRag.core.KnowledgeBase import MedicalHybridKnowledgeBase

FIELDS = ["pk", "text", "summary", "document", "source", "source_name", "doc_id", "chunk_id", "department", "title", "section_path", "page"]


def compact(doc) -> dict:
    metadata = doc.metadata
    return {"pk": metadata.get("pk"), "source": metadata.get("source"), "source_name": metadata.get("source_name"),
            "title": metadata.get("title"), "section_path": metadata.get("section_path"),
            "distance": metadata.get("distance"), "document_preview": metadata.get("document", "")[:240]}


def run_route(kb, config, query: str, name: str, requests: list[SingleSearchRequest]) -> dict:
    docs = kb.search(SearchRequest(query=query, collection_name=config.milvus.collection_name, requests=requests,
                                  output_fields=FIELDS, fuse=FusionSpec(method="rrf", k=60), limit=5))
    if not docs:
        raise RuntimeError(f"Smoke route returned no results: {name}")
    return {"query": query, "route": name, "results": [compact(doc) for doc in docs]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", required=True)
    args = parser.parse_args()
    config = ConfigLoader().config
    kb = MedicalHybridKnowledgeBase(config)
    if not kb.client.has_collection(config.milvus.collection_name):
        raise RuntimeError("Stage 1 collection does not exist; run scripts/02_ingest_data.py --recreate first")
    qa_query = "硫马唑的副作用"
    literature_query = "高血压防治指南"
    dense = lambda field, expr="": SingleSearchRequest(anns_field=field, metric_type="COSINE", search_params={"ef": 64}, limit=5, expr=expr)
    sparse = lambda expr="": SingleSearchRequest(anns_field="text_sparse", metric_type="BM25", search_params={"drop_ratio_search": 0.0}, limit=5, expr=expr)
    routes = [
        run_route(kb, config, qa_query, "summary_dense", [dense("summary_dense")]),
        run_route(kb, config, qa_query, "text_dense", [dense("text_dense")]),
        run_route(kb, config, literature_query, "text_sparse_bm25", [sparse()]),
        run_route(kb, config, literature_query, "hybrid", [dense("summary_dense"), sparse()]),
        run_route(kb, config, qa_query, "source_filter_qa", [dense("summary_dense", 'source == "qa"')]),
        run_route(kb, config, literature_query, "source_filter_literature", [sparse('source == "literature"')]),
    ]
    source_sets = {route["route"]: sorted({item["source"] for item in route["results"]}) for route in routes}
    if source_sets["source_filter_qa"] != ["qa"] or source_sets["source_filter_literature"] != ["literature"]:
        raise RuntimeError(f"Source filter smoke failed: {source_sets}")
    report = {"collection_name": config.milvus.collection_name, "routes": routes, "route_sources": source_sets}
    output = Path("artifacts/stage1")
    output.mkdir(parents=True, exist_ok=True)
    (output / "smoke_search.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
