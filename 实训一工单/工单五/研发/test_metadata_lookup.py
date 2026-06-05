# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

from src.rag_engine import NEGATIVE_QUERY_FALLBACK, RAGEngine, handle_negative_query, validate_answer_quality
from src.document import Document
from utils.llm_engine import extract_complete_entity


def test_legal_representative_uses_hybrid_retrieval():
    engine = RAGEngine()
    engine.vector_store.create_vectorstore(
        [
            Document(
                page_content="发行人的基本情况 公司名称：武汉兴图新科电子股份有限公司 法定代表人：程家明 注册资本：5,520万元",
                metadata={"source_file": "招股说明书1.pdf", "page": 1, "chunk_id": "basic_info"},
            )
        ]
    )

    response = engine.ask("武汉兴图新科电子股份有限公司法定代表人是谁？")

    assert response.answer == "程家明"
    assert response.retrieval_mode == "hybrid"
    assert response.source_chunks
    assert response.accuracy == 1.0
    assert response.query_analysis["intent"] == "entity_lookup"


def test_spaced_chinese_question_uses_hybrid_retrieval():
    engine = RAGEngine()
    engine.vector_store.create_vectorstore(
        [
            Document(
                page_content="发行人的基本情况 公司名称：武汉兴图新科电子股份有限公司 法定代表人：程家明 注册资本：5,520万元",
                metadata={"source_file": "招股说明书1.pdf", "page": 1, "chunk_id": "basic_info"},
            )
        ]
    )

    response = engine.ask("武汉兴图新科电子股份有限公司注 册资本是多少？")

    assert response.answer == "5,520万元"
    assert response.retrieval_mode == "hybrid"
    assert response.source_chunks
    assert response.query_analysis["intent"] == "numeric_lookup"


def test_numeric_answer_extracts_focused_amount_from_context():
    engine = RAGEngine()
    engine.vector_store.create_vectorstore(
        [
            Document(
                page_content=(
                    "（三）补充流动资金项目\n"
                    "为满足公司业务发展和新产品研发等对营运资金的需求，"
                    "拟使用本次发行募集资金15,000 万元用于补充流动资金。"
                    "报告期各期末，公司应收账款账面价值分别为9,107.26 万元、11,485.28 万元。"
                ),
                metadata={"source_file": "招股说明书1.pdf", "page": 491, "chunk_id": "working_capital"},
            )
        ]
    )

    response = engine.ask("武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？")

    assert response.answer == "15,000万元"
    assert "9,107.26" not in response.answer


def test_stream_legal_representative_uses_hybrid_retrieval():
    engine = RAGEngine()
    engine.vector_store.create_vectorstore(
        [
            Document(
                page_content="发行人的基本情况 公司名称：武汉兴图新科电子股份有限公司 法定代表人：程家明 注册资本：5,520万元",
                metadata={"source_file": "招股说明书1.pdf", "page": 1, "chunk_id": "basic_info"},
            )
        ]
    )

    assert "".join(engine.stream_answer("武汉兴图新科电子股份有限公司法定代表人是谁？")) == "程家明"


def test_extract_complete_entity_patterns():
    chunk = "公司名称：测试股份有限公司 注册地址：武汉市 " + ("经营情况良好。" * 20) + " 法定代表人：李四 注册资本：12,300.00万元"

    assert extract_complete_entity(chunk, "legal_representative") == "法定代表人：李四"
    assert extract_complete_entity(chunk, "registered_capital") == "注册资本：12,300万元"


def test_entity_extraction_from_context_is_not_hardcoded():
    engine = RAGEngine()
    engine.vector_store.create_vectorstore(
        [
            Document(
                page_content="发行人的基本情况 公司名称：测试股份有限公司 注册地址：武汉市东湖高新区 法定代表人：李四 注册资本：12,300万元",
                metadata={"source_file": "招股说明书1.pdf", "page": 1, "chunk_id": "basic_info"},
            )
        ]
    )

    representative = engine.ask("测试股份有限公司法定代表人是谁？")
    capital = engine.ask("测试股份有限公司注册资本是多少？")

    assert representative.answer == "李四"
    assert capital.answer == "12,300万元"


def test_national_award_query_uses_cached_search():
    engine = RAGEngine()
    engine.vector_store.create_vectorstore(
        [
            Document(
                page_content=(
                    "2014年12月，某大型研究所牵头承担的"
                    "“某情报、指挥、控制与通信网络一体化工程”荣获国家科技进步一等奖。"
                ),
                metadata={"source_file": "招股说明书1.pdf", "page": 1, "chunk_id": "award"},
            )
        ]
    )

    question = "哪个工程荣获了国家科技进步一等奖"
    first = engine.ask(question, question_id=795)
    second = engine.ask(question, question_id=795)

    assert first.answer == "某情报、指挥、控制与通信网络一体化工程"
    assert second.answer == "某情报、指挥、控制与通信网络一体化工程"
    assert first.accuracy > 0
    assert len(engine.vector_store._search_cache) == 1


def test_grounded_entity_answer_gets_high_accuracy():
    engine = RAGEngine()
    engine.vector_store.create_vectorstore(
        [
            Document(
                page_content=(
                    "截至报告期末，公司获得专利认证35项、软件著作权56项，"
                    "参与制定了全军第一个视频指挥系统技术标准（即《某视频技术规范1.0》）。"
                ),
                metadata={"source_file": "招股说明书1.pdf", "page": 100, "chunk_id": "standard"},
            )
        ]
    )

    response = engine.ask("武汉兴图新科电子股份有限公司参与制定了哪个技术标准？")

    assert "某视频技术规范1.0" in response.answer
    assert response.accuracy >= 0.95


def test_liyuan_metadata_and_table_questions():
    engine = RAGEngine()
    engine.vector_store.create_vectorstore(
        [
            Document(
                page_content=(
                    "武汉力源信息技术股份有限公司 本次发行股数为1,670万股，"
                    "占发行后总股本的比例为25.04%。"
                ),
                metadata={"source_file": "招股说明书2.pdf", "page": 10, "chunk_id": "issue_shares"},
            ),
            Document(
                page_content=(
                    "本次募集资金拟投资以下项目：仓储及物流中心、研发中心、电子商务平台、"
                    "扩充产品种类和数量、其他与主营业务相关的营运资金。"
                ),
                metadata={"source_file": "招股说明书2.pdf", "page": 22, "chunk_id": "table_22"},
            ),
            Document(
                page_content="一、存在控制关系的关联方",
                metadata={"source_file": "招股说明书2.pdf", "page": 157, "chunk_id": "control_title"},
            ),
            Document(
                page_content="| 关联方名称 | 持股比例 | 与本公司关系 |\n| --- | --- | --- |\n| 赵马克 | 42.35% | 公司控股股东 |",
                metadata={"source_file": "招股说明书2.pdf", "page": 157, "chunk_id": "control_table"},
            ),
            Document(
                page_content="二、不存在控制关系的关联方",
                metadata={"source_file": "招股说明书2.pdf", "page": 158, "chunk_id": "non_control_title"},
            ),
            Document(
                page_content=(
                    "| 企业名称 | 与本公司关系 |\n| --- | --- |\n| 融冰投资 | 持有公司股份5%以上的股东 |\n"
                    "| 武汉博润 | 持有公司股份5%以上的股东 |\n| 上海博润 | 持有公司股份5%以上的股东 |\n"
                    "| 听音投资 | 持有公司股份5%以上的股东 |\n| 联众聚源 | 持有公司股份5%以上的股东 |\n"
                    "| 力源贸易 | 公司实际控制人控制的企业 |\n| 普芯达 | 其他关联企业 |"
                ),
                metadata={"source_file": "招股说明书2.pdf", "page": 158, "chunk_id": "non_control_table"},
            )
        ]
    )

    a1 = engine.ask("武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？", question_id=1)
    a2 = engine.ask("武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？", question_id=2)
    a3 = engine.ask("与武汉力源信息技术股份有限公司存在控制关系的关联方是谁，持股比例和本公司关系是什么？", question_id=3)
    a4 = engine.ask("与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？", question_id=4)

    assert a1.answer == "1,670万股，占发行后总股本的比例为25.04%"
    assert a1.accuracy == 1.0
    assert a1.retrieval_mode == "hybrid"
    assert a2.answer.startswith("仓储及物流中心")
    assert a2.retrieval_mode == "hybrid"
    assert a2.source_chunks
    assert "融冰投资" not in a2.answer
    assert "赵马克" in a3.answer
    assert "融冰投资" in a4.answer
    assert "赵马克" not in a4.answer


def test_question_selects_target_pdf_for_retrieval():
    engine = RAGEngine()
    engine.vector_store.create_vectorstore(
        [
            Document(
                page_content="招股说明书1里的募集资金信息，不应回答力源问题。",
                metadata={"source_file": "招股说明书1.pdf", "page": 1, "chunk_id": "xingtuxinke"},
            ),
            Document(
                page_content="本次募集资金拟投资以下项目：仓储及物流中心、研发中心、电子商务平台。",
                metadata={"source_file": "招股说明书2.pdf", "page": 22, "chunk_id": "liyuan"},
            ),
        ]
    )

    liyuan = engine.ask("武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？", question_id=2)
    default = engine._build_context("募集资金信息", top_k=2)[4]

    assert liyuan.source_chunks
    assert all(chunk["metadata"]["source_file"] == "招股说明书2.pdf" for chunk in liyuan.source_chunks)
    assert default
    assert all(chunk["metadata"]["source_file"] == "招股说明书1.pdf" for chunk in default)


def test_quality_validator_rejects_negative_control_relation_conflict():
    result = validate_answer_quality(
        "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？",
        "赵马克，持股比例42.35%，公司控股股东，存在控制关系的关联方。",
        [{"content": "存在控制关系的关联方为赵马克。"}],
    )

    assert result["is_valid"] is False
    assert "negative_control_relation_conflict" in result["reason"]


def test_quality_validator_rejects_related_party_income_answer():
    result = validate_answer_quality(
        "未披露的关联方有哪些？",
        "2017年度、2018年度收入分别为1,000万元、2,000万元。",
        [{"content": "主营业务收入分别为1,000万元、2,000万元。"}],
    )

    assert result["is_valid"] is False
    assert "related_party_question_answered_with_income" in result["reason"]


def test_handle_negative_query_excludes_positive_control_related_party():
    engine = RAGEngine()
    engine.vector_store.create_vectorstore(
        [
            Document(
                page_content="二、关联方及关联交易 (一)关联方及关联关系 1、存在控制关系的关联方",
                metadata={"source_file": "招股说明书2.pdf", "page": 157, "chunk_id": "p157_004"},
            ),
            Document(
                page_content="| 关联方名称 | 持股比例 | 与本公司关系 |\n| --- | --- | --- |\n| 赵马克 | 42.35% | 公司控股股东 |",
                metadata={"source_file": "招股说明书2.pdf", "page": 157, "chunk_id": "p157_005"},
            ),
            Document(
                page_content="2、不存在控制关系的关联方",
                metadata={"source_file": "招股说明书2.pdf", "page": 157, "chunk_id": "p157_006"},
            ),
            Document(
                page_content=(
                    "| 企业名称 | 与本公司关系 |\n| --- | --- |\n| 融冰投资 | 持有公司股份5%以上的股东 |\n"
                    "| 武汉博润 | 持有公司股份5%以上的股东 |\n| 上海博润 | 持有公司股份5%以上的股东 |"
                ),
                metadata={"source_file": "招股说明书2.pdf", "page": 157, "chunk_id": "p157_007"},
            ),
        ]
    )

    results = handle_negative_query("与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？", engine.vector_store)

    assert "赵马克" not in results
    assert results[:3] == ["融冰投资", "武汉博润", "上海博润"]


def test_ask_uses_negative_query_handler():
    engine = RAGEngine()
    engine.vector_store.create_vectorstore(
        [
            Document(
                page_content="1、存在控制关系的关联方",
                metadata={"source_file": "招股说明书2.pdf", "page": 157, "chunk_id": "control_title"},
            ),
            Document(
                page_content="| 关联方名称 | 持股比例 | 与本公司关系 |\n| --- | --- | --- |\n| 赵马克 | 42.35% | 公司控股股东 |",
                metadata={"source_file": "招股说明书2.pdf", "page": 157, "chunk_id": "control_table"},
            ),
            Document(
                page_content="2、不存在控制关系的关联方",
                metadata={"source_file": "招股说明书2.pdf", "page": 157, "chunk_id": "non_control_title"},
            ),
            Document(
                page_content="| 企业名称 | 与本公司关系 |\n| --- | --- |\n| 融冰投资 | 持有公司股份5%以上的股东 |\n| 武汉博润 | 持有公司股份5%以上的股东 |",
                metadata={"source_file": "招股说明书2.pdf", "page": 157, "chunk_id": "non_control_table"},
            ),
        ]
    )

    response = engine.ask("与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？")

    assert response.retrieval_mode == "negative_query"
    assert "融冰投资" in response.answer
    assert "赵马克" not in response.answer
    assert response.query_analysis["intent"] == "negative_query"


def test_handle_negative_query_returns_disclosure_fallback_when_unknown():
    engine = RAGEngine()
    engine.vector_store.create_vectorstore(
        [
            Document(
                page_content="关联方及关联关系如下，但未列示未披露关联方清单。",
                metadata={"source_file": "招股说明书1.pdf", "page": 260, "chunk_id": "related"},
            )
        ]
    )

    results = handle_negative_query("兴图新科未披露的关联方有哪些？", engine.vector_store)

    assert results == [NEGATIVE_QUERY_FALLBACK]


def test_undisclosed_related_party_is_prechecked_without_retrieval():
    engine = RAGEngine()
    engine.vector_store.create_vectorstore(
        [
            Document(
                page_content="| 企业名称 | 与本公司关系 |\n| --- | --- |\n| 融冰投资 | 持有公司股份5%以上的股东 |",
                metadata={"source_file": "招股说明书2.pdf", "page": 157, "chunk_id": "related"},
            )
        ]
    )

    response = engine.ask("武汉力源信息技术股份有限公司未披露的关联方有哪些？")

    assert response.answer == NEGATIVE_QUERY_FALLBACK
    assert response.retrieval_mode == "negative_query"
    assert response.query_analysis["intent"] == "undisclosed_precheck"


def test_answer_cache_returns_same_response_within_30_seconds():
    engine = RAGEngine()
    first = engine.ask("武汉力源信息技术股份有限公司未披露的关联方有哪些？")
    second = engine.ask("武汉力源信息技术股份有限公司未披露的关联方有哪些？")

    assert first.answer == second.answer
    assert first is not second


def test_long_term_memory_does_not_override_pdf_answer_by_default():
    engine = RAGEngine()

    class WrongMemory:
        def search(self, question):
            return {"question": question, "answer": "错误答案", "score": 1.0}

    engine.long_term_memory = WrongMemory()
    engine.vector_store.create_vectorstore(
        [
            Document(
                page_content="发行人的基本情况 公司名称：武汉兴图新科电子股份有限公司 注册资本：5,520万元",
                metadata={"source_file": "招股说明书1.pdf", "page": 1, "chunk_id": "basic_info"},
            )
        ]
    )

    response = engine.ask("武汉兴图新科电子股份有限公司注册资本是多少？")

    assert response.answer == "5,520万元"
    assert response.memory_hit is False
    assert response.retrieval_mode == "hybrid"
    assert response.source_chunks


def test_select_pdf_uses_company_keywords():
    engine = RAGEngine()

    assert engine.select_pdf("武汉力源的本次发行情况") == "招股说明书2.pdf"
    assert engine.select_pdf("力源信息技术注册资本是多少") == "招股说明书2.pdf"
    assert engine.select_pdf("武汉兴图的法定代表人是谁") == "招股说明书1.pdf"
    assert engine.select_pdf("兴图新科注册资本是多少") == "招股说明书1.pdf"


def test_any_undisclosed_question_returns_fixed_answer_without_retrieval():
    engine = RAGEngine()

    response = engine.ask("同行业上市公司2019年半年度报告未披露员工数量信息是什么？")

    assert response.answer == NEGATIVE_QUERY_FALLBACK
    assert response.retrieval_mode == "negative_query"
    assert response.query_analysis["intent"] == "undisclosed_precheck"


def test_validator_rejects_concrete_answer_for_undisclosed_question():
    result = validate_answer_quality(
        "未披露的关联方有哪些？",
        "融冰投资、武汉博润、上海博润",
        [{"content": "关联方名称：融冰投资、武汉博润、上海博润"}],
    )

    assert result["is_valid"] is False
    assert "undisclosed_question_has_concrete_answer" in result["reason"]
