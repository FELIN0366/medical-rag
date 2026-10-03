"""Stage 2 可复现 Eval Set 的模型、人工文献标签与 QA 改写器。"""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Iterable, Literal

import httpx
from pydantic import BaseModel, Field

from ..config.models import AppConfig


class EvalRecord(BaseModel):
    """一条检索评测记录；rel=2 为 direct evidence，rel=1 为 partial。"""
    query_id: str
    query: str
    dataset: str
    query_type: str = "semantic"
    label_quality: Literal["direct", "best_effort"] = "direct"
    original_question: str | None = None
    gold_pk: str | None = None
    relevant_pks: list[str] = Field(default_factory=list)
    partial_relevant_pks: list[str] = Field(default_factory=list)
    hard_negative_pks: list[str] = Field(default_factory=list)
    gold_source_name: str = ""
    gold_section_path: str = ""
    gold_evidence_text: str = ""
    curation_note: str = ""

    def direct_gold_pks(self) -> set[str]:
        return set(self.relevant_pks) | ({self.gold_pk} if self.gold_pk else set())

    def relevance(self, pk: str) -> int:
        if pk in self.direct_gold_pks():
            return 2
        if pk in self.partial_relevant_pks:
            return 1
        return 0


def write_eval_records(path: Path, records: Iterable[EvalRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            # 当前 Pydantic 版本的 model_dump_json 不接收 ensure_ascii，统一交由标准库编码。
            stream.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")


def load_eval_records(path: Path) -> list[EvalRecord]:
    with path.open(encoding="utf-8") as stream:
        return [EvalRecord.model_validate_json(line) for line in stream if line.strip()]


# 下面 20 条由人工逐段核对后冻结；PK 对应 Stage 1 冻结 collection，不能由模型自动替代。
MANUAL_LITERATURE_CASES = (
    ("lit-01", "不服用降压药时，诊室血压达到什么标准可定义为高血压？", "numeric_or_table", "d80823760616ff70adb0f7e1f95465c9992b57cbbd269149c60b68458349ac39", "ff37f4cb89401730c477d6ff9cdc8e25d611b6caec43fc39e77d8a7e09200114", "32830de0e566ed6e6be2624d4f6de76c4b95e7865cea5efa37a3742823a85a27"),
    ("lit-02", "收缩压 165、舒张压 105 属于几级高血压？", "numeric_or_table", "3f82090c8ff6a4128ddef8cc6d4570f18508d07f5fd5f9f39a353ce1271f867b", "d80823760616ff70adb0f7e1f95465c9992b57cbbd269149c60b68458349ac39", "0642306e29d55561f08bbefe16a274c62479597254cd6667329abff805046ac0"),
    ("lit-03", "高血压患者即使没有不适，也可能出现哪些常见症状和风险？", "semantic", "d265b13dde054024cae3f23ea0b2877fe04a794b1265d0421670edfa40b1a18a", "3ed79e2b7849b65e88d7f480dcc4b8757160d4921c69037f104cdf040b4c2102", "0ac8738a4387ed6a5fb790d0f2c3d92978a5f170ebd147f0c6e87625efd7060f"),
    ("lit-04", "哪些生活方式因素会增加高血压风险？", "semantic", "e0078f76cb3e2be2ca361ad87413e27c08de1ec61e71bd4c0ba05b7b5fa7bc14", "ff37f4cb89401730c477d6ff9cdc8e25d611b6caec43fc39e77d8a7e09200114", "1db483cfaaacca9c42a83f0ed50b37dba738f622853fc6595838bc79c78403fd"),
    ("lit-05", "高血压饮食中每日食盐摄入建议是多少？", "numeric_or_table", "31f13efc59e0a7bf84c0e9b83a947d7c8a6a689d2a859e0f9d8dae14c3f771e1", "e0078f76cb3e2be2ca361ad87413e27c08de1ec61e71bd4c0ba05b7b5fa7bc14", "32830de0e566ed6e6be2624d4f6de76c4b95e7865cea5efa37a3742823a85a27"),
    ("lit-06", "DASH 饮食模式对高血压患者的食物选择有什么建议？", "semantic", "31f13efc59e0a7bf84c0e9b83a947d7c8a6a689d2a859e0f9d8dae14c3f771e1", "511c40cc3ccfc4b4372162ef9271d328c5acb1474ca1a22fe8f33e379348fd67", "0ac8738a4387ed6a5fb790d0f2c3d92978a5f170ebd147f0c6e87625efd7060f"),
    ("lit-07", "高血压患者每周应进行多长时间、何种强度的运动？", "numeric_or_table", "32830de0e566ed6e6be2624d4f6de76c4b95e7865cea5efa37a3742823a85a27", "c93396c98faac5c8538af169300f9f108b160c8c6593809c2885a67af04ad58d", "31f13efc59e0a7bf84c0e9b83a947d7c8a6a689d2a859e0f9d8dae14c3f771e1"),
    ("lit-08", "血压超过什么数值时应暂缓运动？", "numeric_or_table", "32830de0e566ed6e6be2624d4f6de76c4b95e7865cea5efa37a3742823a85a27", "3f82090c8ff6a4128ddef8cc6d4570f18508d07f5fd5f9f39a353ce1271f867b", "0fbc6f91b7aeb8b095e040cef502eabc0b1e23d652e0e454d2cb70a72db325c8"),
    ("lit-09", "ACEI 类降压药有哪些代表药，常见不良反应是什么？", "lexical", "0ac8738a4387ed6a5fb790d0f2c3d92978a5f170ebd147f0c6e87625efd7060f", "1db483cfaaacca9c42a83f0ed50b37dba738f622853fc6595838bc79c78403fd", "848e0dc387b299f492bbf82b44309b7a86527d7e785ae2b2e1afe9a6f87ca4a4"),
    ("lit-10", "高血压合并心力衰竭时可优先选择哪些降压药？", "lexical", "1db483cfaaacca9c42a83f0ed50b37dba738f622853fc6595838bc79c78403fd", "0ac8738a4387ed6a5fb790d0f2c3d92978a5f170ebd147f0c6e87625efd7060f", "e0078f76cb3e2be2ca361ad87413e27c08de1ec61e71bd4c0ba05b7b5fa7bc14"),
    ("lit-11", "高血压合并糖尿病肾病或蛋白尿时，优先考虑哪类药物？", "lexical", "1db483cfaaacca9c42a83f0ed50b37dba738f622853fc6595838bc79c78403fd", "0ac8738a4387ed6a5fb790d0f2c3d92978a5f170ebd147f0c6e87625efd7060f", "32830de0e566ed6e6be2624d4f6de76c4b95e7865cea5efa37a3742823a85a27"),
    ("lit-12", "怀孕期间高血压可以使用哪些药物，哪些药物要避免？", "lexical", "2ca38c90e16e19256435497b6e74af3f982b41264859b46f8eda1c77cf822a2c", "0ac8738a4387ed6a5fb790d0f2c3d92978a5f170ebd147f0c6e87625efd7060f", "848e0dc387b299f492bbf82b44309b7a86527d7e785ae2b2e1afe9a6f87ca4a4"),
    ("lit-13", "发生高血压危象应如何处理？", "lexical", "2ca38c90e16e19256435497b6e74af3f982b41264859b46f8eda1c77cf822a2c", "d265b13dde054024cae3f23ea0b2877fe04a794b1265d0421670edfa40b1a18a", "31f13efc59e0a7bf84c0e9b83a947d7c8a6a689d2a859e0f9d8dae14c3f771e1"),
    ("lit-14", "哪些常见药物可能让血压升高，需要在用药时特别留意？", "lexical", "0642306e29d55561f08bbefe16a274c62479597254cd6667329abff805046ac0", "4f0972964456de0ece3803a11066bc659f4f177c1e77c6426a53d96d14e1a26f", "32830de0e566ed6e6be2624d4f6de76c4b95e7865cea5efa37a3742823a85a27"),
    ("lit-15", "为什么比索洛尔不能突然停药？", "lexical", "4f0972964456de0ece3803a11066bc659f4f177c1e77c6426a53d96d14e1a26f", "0ac8738a4387ed6a5fb790d0f2c3d92978a5f170ebd147f0c6e87625efd7060f", "31f13efc59e0a7bf84c0e9b83a947d7c8a6a689d2a859e0f9d8dae14c3f771e1"),
    ("lit-16", "哪些降压药联合方案不建议同时使用，原因是什么？", "lexical", "78925e7f6165f44c1acfdaf56f43ebe7711c49c27fb01faf87909562e7f5fde7", "0ac8738a4387ed6a5fb790d0f2c3d92978a5f170ebd147f0c6e87625efd7060f", "e0078f76cb3e2be2ca361ad87413e27c08de1ec61e71bd4c0ba05b7b5fa7bc14"),
    ("lit-17", "肝阳上亢型高血压有哪些症状及推荐中成药？", "semantic", "3ed79e2b7849b65e88d7f480dcc4b8757160d4921c69037f104cdf040b4c2102", "0521653ff587b6705fca8bff39643e812094a08e36caca99c9e621e0b1dfe230", "1db483cfaaacca9c42a83f0ed50b37dba738f622853fc6595838bc79c78403fd"),
    ("lit-18", "哪些营养素有助于血压控制，以及推荐摄入量是多少？", "numeric_or_table", "511c40cc3ccfc4b4372162ef9271d328c5acb1474ca1a22fe8f33e379348fd67", "31f13efc59e0a7bf84c0e9b83a947d7c8a6a689d2a859e0f9d8dae14c3f771e1", "0ac8738a4387ed6a5fb790d0f2c3d92978a5f170ebd147f0c6e87625efd7060f"),
    ("lit-19", "H 型高血压为什么建议补充叶酸和维生素 B 族？", "lexical", "cb877685fec34e2b144a25ede33c729b57f2739aae54f59892c0fb3ec565dc86", "511c40cc3ccfc4b4372162ef9271d328c5acb1474ca1a22fe8f33e379348fd67", "32830de0e566ed6e6be2624d4f6de76c4b95e7865cea5efa37a3742823a85a27"),
    ("lit-20", "血压正常后能自行停用降压药吗？", "topic_or_section", "848e0dc387b299f492bbf82b44309b7a86527d7e785ae2b2e1afe9a6f87ca4a4", "4f0972964456de0ece3803a11066bc659f4f177c1e77c6426a53d96d14e1a26f", "31f13efc59e0a7bf84c0e9b83a947d7c8a6a689d2a859e0f9d8dae14c3f771e1"),
)

# 当前冻结语料只有极少数 QA 与文献具备严格 direct evidence。
# 此表是在用户明确要求下保留的“最佳努力”诊断样本；它们只标为 partial，绝不混入主指标。
BEST_EFFORT_CROSS_CASES = (
    ("cross-01", "服用氨酚伪麻美芬片Ⅱ须注意的事项", "lexical", "0642306e29d55561f08bbefe16a274c62479597254cd6667329abff805046ac0",
     "文献明确提醒含伪麻黄碱的感冒药可致血压骤升；与该复方药的注意事项存在成分级关联。"),
    ("cross-02", "请描述卡托普利的历史", "lexical", "0ac8738a4387ed6a5fb790d0f2c3d92978a5f170ebd147f0c6e87625efd7060f",
     "文献给出卡托普利所属药物类别、作用机制与常见不良反应，但不提供药物历史。"),
    ("cross-03", "胃溃疡喝中药多少时间", "topic_or_section", "c99efd8c31a0da13e7c66b756e4c86ebd819c9dcbb4a88b0cb420badd88efe3d",
     "文献给出幽门螺杆菌根除疗程与抗生素组合，不能替代针对中药疗程的直接建议。"),
    ("cross-04", "幽门螺杆菌检测", "topic_or_section", "f4e526a15c878cced85b2e967d00740b5b09bbbf3167a6f2cf968b0d8807c42f",
     "文献为幽门螺杆菌根除治疗指南，覆盖疾病管理背景，但不直接列出检测方法。"),
    ("cross-05", "血糖高吃什么好食物好", "semantic", "31f13efc59e0a7bf84c0e9b83a947d7c8a6a689d2a859e0f9d8dae14c3f771e1",
     "文献提供低盐、DASH 等一般膳食建议；其目标疾病为高血压，不能作为高血糖饮食的直接证据。"),
)


def build_literature_records(rows_by_pk: dict[str, dict]) -> list[EvalRecord]:
    """将人工冻结标签与当前 collection 元数据组合为完整 Literature Eval。"""
    records: list[EvalRecord] = []
    for query_id, query, query_type, relevant, partial, hard_negative in MANUAL_LITERATURE_CASES:
        evidence = rows_by_pk.get(relevant)
        if evidence is None:
            raise RuntimeError(f"Literature Eval Gold PK 不在当前 collection：{relevant}")
        for pk in (partial, hard_negative):
            if pk not in rows_by_pk:
                raise RuntimeError(f"Literature Eval 标注 PK 不在当前 collection：{pk}")
        records.append(EvalRecord(
            query_id=query_id, query=query, dataset="literature", query_type=query_type,
            relevant_pks=[relevant], partial_relevant_pks=[partial], hard_negative_pks=[hard_negative],
            gold_source_name=evidence.get("source_name", ""), gold_section_path=evidence.get("section_path", ""),
            gold_evidence_text=evidence.get("document", ""), curation_note="人工核对 Stage 1 文献块文本后冻结。",
        ))
    return records


def build_best_effort_cross_records(rows_by_pk: dict[str, dict], qa_rows: list[dict]) -> list[EvalRecord]:
    """构建用户允许的低置信 Cross 诊断集，所有标签均保持 partial。"""
    qa_questions = {str(row["summary"]) for row in qa_rows}
    records: list[EvalRecord] = []
    for query_id, question, query_type, partial_pk, note in BEST_EFFORT_CROSS_CASES:
        if question not in qa_questions:
            raise RuntimeError(f"最佳努力 Cross 原题不在冻结 QA：{question}")
        evidence = rows_by_pk.get(partial_pk)
        if evidence is None:
            raise RuntimeError(f"最佳努力 Cross 文献 PK 不在当前 collection：{partial_pk}")
        records.append(EvalRecord(
            query_id=query_id, query=question, original_question=question, dataset="cross_corpus",
            query_type=query_type, label_quality="best_effort", partial_relevant_pks=[partial_pk],
            gold_source_name=evidence.get("source_name", ""), gold_section_path=evidence.get("section_path", ""),
            gold_evidence_text=evidence.get("document", ""), curation_note=note,
        ))
    return records


def sample_qa_rows(rows: list[dict], seed: int = 42, size: int = 30) -> list[dict]:
    """按 PK 排序后固定抽样，避免 Milvus 返回顺序影响 Eval Set。"""
    if len(rows) < size:
        raise RuntimeError(f"QA 数量不足：需要 {size}，实际 {len(rows)}")
    return random.Random(seed).sample(sorted(rows, key=lambda row: row["pk"]), size)


def _extract_json(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def _question_key(text: str) -> str:
    """去除空白与标点后比较，阻止只改符号的伪改写进入 QA Eval。"""
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text).lower()


def paraphrase_qa_batch(questions: list[str], config: AppConfig) -> list[str]:
    """每批十条调用一次 DeepSeek，生成不复述原句的用户式查询。"""
    api_key = os.getenv(config.llm.env_key_name or "")
    if not api_key:
        raise RuntimeError(f"未配置 QA 改写所需环境变量：{config.llm.env_key_name}")
    prompt = (
        "将每个医学问题改写为自然的用户检索问法。保持原医学意图，不增加医学事实，"
        "不得复制原句。只输出 JSON：{\"items\":[{\"index\":0,\"query\":\"...\"}]}。\n"
        + json.dumps([{"index": index, "question": question} for index, question in enumerate(questions)], ensure_ascii=False)
    )
    url = (config.llm.base_url or "").rstrip("/") + "/chat/completions"
    response = httpx.post(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=30.0,
                          json={"model": config.llm.model, "temperature": 0, "messages": [
                              {"role": "system", "content": "你是严谨的医学检索查询改写器。"},
                              {"role": "user", "content": prompt},
                          ]})
    response.raise_for_status()
    payload = _extract_json(response.json()["choices"][0]["message"]["content"])
    items = payload.get("items", [])
    by_index = {int(item["index"]): str(item["query"]).strip() for item in items}
    if sorted(by_index) != list(range(len(questions))) or any(not item for item in by_index.values()):
        raise RuntimeError("QA 改写响应缺少条目或包含空 query")
    if any(_question_key(by_index[index]) == _question_key(questions[index]) for index in by_index):
        raise RuntimeError("QA 改写器返回了原始 question，拒绝写入 Eval Set")
    return [by_index[index] for index in range(len(questions))]


def build_qa_records(qa_rows: list[dict], config: AppConfig) -> list[EvalRecord]:
    """按十条一批改写固定的三十条 QA，并保存其原始问题和唯一 Gold PK。"""
    sampled = sample_qa_rows(qa_rows)
    queries: list[str] = []
    for start in range(0, len(sampled), 10):
        queries.extend(paraphrase_qa_batch([row["summary"] for row in sampled[start:start + 10]], config))
    return [EvalRecord(query_id=f"qa-{index + 1:02d}", query=query, dataset="qa",
                       original_question=row["summary"], gold_pk=row["pk"], relevant_pks=[row["pk"]],
                       curation_note="固定 seed=42 抽样；DeepSeek 批量改写。")
            for index, (row, query) in enumerate(zip(sampled, queries))]
