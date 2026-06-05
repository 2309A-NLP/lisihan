# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-Query 理解优化任务。

本文件用于验证工单五多轮对话与指代消解规则。
"""

from __future__ import annotations

from src.coreference_resolver import CoreferenceResolver
from src.rag_engine import RAGEngine


def _turn(result):
    return {
        "question": result.original_question,
        "resolved_question": result.resolved_question,
        "mentioned_companies": result.mentioned_companies,
        "current_company": result.current_company,
    }


def test_pronouns_resolve_to_latest_company():
    resolver = CoreferenceResolver()
    history = [
        _turn(resolver.resolve("报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"))
    ]

    q2 = resolver.resolve("他参与的哪个工程荣获了国家科技进步一等奖？", history)
    history.append(_turn(q2))
    q3 = resolver.resolve("这个公司的法定代表人是谁？", history)

    assert q2.resolved_question == "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？"
    assert q3.resolved_question == "武汉兴图新科电子股份有限公司的法定代表人是谁？"


def test_company_switch_follow_up_reuses_previous_question_shape():
    resolver = CoreferenceResolver()
    history = [
        _turn(resolver.resolve("武汉兴图新科电子股份有限公司法定代表人是谁？"))
    ]

    result = resolver.resolve("那武汉力源信息技术股份有限公司呢？", history)

    assert result.resolved_question == "武汉力源信息技术股份有限公司法定代表人是谁？"
    assert result.current_company == "武汉力源信息技术股份有限公司"


def test_alias_extraction_supports_short_names():
    resolver = CoreferenceResolver()

    assert resolver.extract_companies("兴图新科注册资本是多少？") == ["武汉兴图新科电子股份有限公司"]
    assert resolver.extract_companies("力源组织结构图中销售处最多的是哪个销售部？") == ["武汉力源信息技术股份有限公司"]


def test_rag_engine_uses_history_for_reference_not_answer_source():
    engine = RAGEngine()
    session_id = "coreference-unit"
    engine.session_manager.clear(session_id)
    from src.document import Document

    engine.vector_store.create_vectorstore(
        [
            Document(
                page_content="发行人的基本情况 公司名称：武汉兴图新科电子股份有限公司 法定代表人：程家明 注册资本：5,520万元",
                metadata={"source_file": "招股说明书1.pdf", "page": 1, "chunk_id": "xingtuxinke_basic"},
            ),
            Document(
                page_content="发行人的基本情况 公司名称：武汉力源信息技术股份有限公司 法定代表人：赵马克 注册资本：5,000万元",
                metadata={"source_file": "招股说明书2.pdf", "page": 1, "chunk_id": "liyuan_basic"},
            ),
        ]
    )

    first = engine.ask("武汉兴图新科电子股份有限公司法定代表人是谁？", session_id=session_id)
    second = engine.ask("这个公司的法定代表人是谁？", session_id=session_id)
    third = engine.ask("那武汉力源信息技术股份有限公司呢？", session_id=session_id)

    assert first.answer == "程家明"
    assert first.retrieval_mode == "hybrid"
    assert first.source_chunks
    assert second.answer == "程家明"
    assert second.query_analysis["resolved_question"] == "武汉兴图新科电子股份有限公司的法定代表人是谁？"
    assert third.answer == "赵马克"
    assert third.retrieval_mode == "hybrid"
    assert third.query_analysis["resolved_question"] == "武汉力源信息技术股份有限公司的法定代表人是谁？"
