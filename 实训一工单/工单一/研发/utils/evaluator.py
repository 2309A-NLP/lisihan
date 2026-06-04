# -*- coding: utf-8 -*-
"""RAG evaluation module."""

from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import mean
from typing import Dict, List


REFERENCE_ANSWERS = {
    260: {
        "answer": "6,464.51万元、14,414.16万元、18,780.67万元、4,627.14万元",
        "aliases": [["6464.51万元", "14414.16万元", "18780.67万元", "4627.14万元"]],
    },
    95: {
        "answer": "全军第一个视频指挥系统技术标准（《某视频技术规范1.0》）",
        "aliases": [["视频指挥系统技术标准", "某视频技术规范1.0"]],
    },
    33: {
        "answer": "82.10%、97.31%、94.84%、94.34%",
        "aliases": [["82.10%", "97.31%", "94.84%", "94.34%"]],
    },
    34: {
        "answer": "电子元器件制造企业、机箱/机柜等金属壳体制造企业",
        "aliases": [["电子元器件制造", "金属壳体"], ["电子元器件", "机箱", "机柜"]],
    },
    957: {
        "answer": "国防军队视频指挥领域",
        "aliases": [["国防军队视频指挥领域"], ["军队视频指挥领域"]],
    },
    793: {
        "answer": "军队、政府机关、能源等行业企业",
        "aliases": [["军队", "政府机关", "能源"], ["军工行业"]],
    },
    795: {
        "answer": "某情报、指挥、控制与通信网络一体化工程",
        "aliases": [["某情报", "指挥", "控制", "通信网络一体化工程"], ["C4ISR"]],
    },
    543: {
        "answer": "5,520.00万元",
        "aliases": [["5520.00万元"], ["5520万元"]],
    },
    531: {
        "answer": "程家明",
        "aliases": [["程家明"]],
    },
    207: {
        "answer": "15,000.00万元",
        "aliases": [["15000.00万元"], ["15000万元"]],
    },
}


@dataclass
class RAGEvaluationResult:
    """RAG evaluation result."""

    context_relevance: float
    answer_faithfulness: float
    answer_relevance: float
    avg_response_time: float
    success_rate: float
    accuracy: float
    avg_retrieval_time: float
    avg_query_time: float
    avg_generation_time: float
    avg_total_time: float
    rag_vs_llm_improvement: float
    details: Dict


class RAGEvaluator:
    """RAG system evaluator."""

    def __init__(self):
        self.evaluation_criteria = {
            "context_relevance": "检索到的上下文与问题的相关程度",
            "answer_faithfulness": "答案是否基于检索到的上下文",
            "answer_completeness": "答案是否完整回答了问题",
        }

    def normalize_for_strict_match(self, text: str) -> str:
        text = text or ""
        text = text.replace("，", ",").replace("．", ".").replace("％", "%")
        text = text.replace("（", "(").replace("）", ")").replace("：", ":")
        text = re.sub(r"\s+", "", text.lower())
        text = re.sub(r"(?<=\d),(?=\d)", "", text)
        text = re.sub(r"(?<=\d)\.00(?=万?元|%|$)", "", text)
        text = re.sub(r"[。、；;，,：:\-_\[\]【】()（）《》\"'“”‘’/\\]", "", text)
        return text

    def strict_answer_match(self, predicted: str, reference: Dict) -> Dict:
        normalized_predicted = self.normalize_for_strict_match(predicted)
        groups = reference.get("aliases") or [[reference.get("answer", "")]]
        matched_group = None
        for group in groups:
            normalized_group = [self.normalize_for_strict_match(item) for item in group if item]
            if normalized_group and all(item in normalized_predicted for item in normalized_group):
                matched_group = group
                break
        return {
            "correct": matched_group is not None,
            "match_type": "normalized_exact_key_values" if matched_group else "no_strict_match",
            "reference_answer": reference.get("answer", ""),
            "matched_terms": matched_group or [],
        }

    def evaluate_accuracy(self, results: List[Dict]) -> Dict:
        matches = []
        for item in results:
            question_id = item.get("id")
            reference = REFERENCE_ANSWERS.get(question_id)
            answer = item.get("rag_answer", item.get("answer", ""))
            if reference:
                match = self.strict_answer_match(answer, reference)
            else:
                match = {
                    "correct": False,
                    "match_type": "missing_reference",
                    "reference_answer": "",
                    "matched_terms": [],
                }
            matches.append(
                {
                    "id": question_id,
                    "question": item.get("question", ""),
                    "answer": answer,
                    **match,
                }
            )
        correct_count = sum(1 for item in matches if item["correct"])
        total = len(matches)
        return {
            "accuracy": correct_count / total if total else 0.0,
            "correct_count": correct_count,
            "total": total,
            "match_standard": "严格匹配：标准答案关键值/实体归一化后必须全部命中，不采用包含即正确的宽松口径。",
            "matches": matches,
        }

    def evaluate_single(
        self,
        question: str,
        rag_answer: str,
        retrieved_contexts: List[str],
        llm_only_answer: str = None,
    ) -> Dict:
        scores = {}
        scores["context_relevance"] = self._compute_context_relevance(question, retrieved_contexts)
        scores["answer_faithfulness"] = self._compute_faithfulness(rag_answer, retrieved_contexts)
        scores["answer_completeness"] = self._compute_completeness(question, rag_answer)
        scores["overall_score"] = mean(
            [
                scores["context_relevance"],
                scores["answer_faithfulness"],
                scores["answer_completeness"],
            ]
        )
        if llm_only_answer:
            scores["rag_vs_llm"] = self._compare_with_llm_only(rag_answer, llm_only_answer, question)
        return scores

    def _compute_context_relevance(self, question: str, contexts: List[str]) -> float:
        if not contexts:
            return 0.0
        keywords = self._extract_keywords(question)
        if not keywords:
            return 0.5
        all_context = " ".join(contexts).lower()
        matched_count = sum(1 for kw in keywords if kw.lower() in all_context)
        return matched_count / len(keywords)

    def _compute_faithfulness(self, answer: str, contexts: List[str]) -> float:
        if not contexts or not answer:
            return 0.0
        import re

        answer_sentences = re.split(r"[。.!?]+", answer)
        all_context = " ".join(contexts).lower()
        supported_sentences = 0
        meaningful_sentences = 0
        for sentence in answer_sentences:
            if len(sentence.strip()) < 5:
                continue
            meaningful_sentences += 1
            words = self._extract_keywords(sentence)
            if words and any(w.lower() in all_context for w in words):
                supported_sentences += 1
        if meaningful_sentences == 0:
            return 0.5
        return supported_sentences / meaningful_sentences

    def _compute_completeness(self, question: str, answer: str) -> float:
        if not answer:
            return 0.0
        lowered = answer.lower()
        if (
            "未找到" in answer
            or "无法提供" in answer
            or "no sufficiently relevant information" in lowered
            or "not enough information" in lowered
        ):
            return 0.2
        min_length = 20
        length_score = min(1.0, len(answer) / min_length)
        has_numbers = any(c.isdigit() for c in answer)
        has_specific_info = has_numbers or len(answer.split()) > 10
        info_score = 1.0 if has_specific_info else 0.5
        return (length_score + info_score) / 2

    def _compare_with_llm_only(self, rag_answer: str, llm_only_answer: str, question: str) -> float:
        rag_has_data = any(c.isdigit() for c in rag_answer)
        llm_has_data = any(c.isdigit() for c in llm_only_answer)
        rag_lower = rag_answer.lower()
        llm_lower = llm_only_answer.lower()
        rag_admits_unknown = "未找到" in rag_answer or "no sufficiently relevant information" in rag_lower
        llm_admits_unknown = (
            "未找到" in llm_only_answer
            or "不确定" in llm_only_answer
            or "no sufficiently relevant information" in llm_lower
            or "not enough information" in llm_lower
            or "uncertain" in llm_lower
        )
        if rag_admits_unknown and not llm_admits_unknown:
            return 0.3
        if rag_has_data and not llm_has_data:
            return 0.8
        if rag_has_data and llm_has_data:
            return 0.5
        return 0.0

    def _extract_keywords(self, text: str) -> List[str]:
        import re

        raw_words = re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]+", text)
        stopwords = {
            "的",
            "了",
            "是",
            "在",
            "和",
            "中",
            "或",
            "等",
            "如何",
            "什么",
            "哪些",
            "the",
            "a",
            "an",
            "and",
            "or",
            "of",
            "to",
            "in",
            "for",
            "with",
            "what",
            "which",
            "who",
            "how",
            "much",
            "many",
            "is",
            "are",
            "was",
            "were",
        }
        keywords = []
        for word in raw_words:
            if word in stopwords or len(word) <= 1:
                continue
            if re.fullmatch(r"[\u4e00-\u9fff]+", word) and len(word) > 4:
                keywords.extend(word[i : i + 2] for i in range(len(word) - 1))
            else:
                keywords.append(word)

        seen = set()
        return [kw for kw in keywords if not (kw in seen or seen.add(kw))][:20]

    def evaluate_batch(self, results: List[Dict]) -> RAGEvaluationResult:
        context_relevances = []
        answer_faithfulnesses = []
        response_times = []
        retrieval_times = []
        query_times = []
        generation_times = []
        total_times = []
        successes = []
        rag_vs_llm_scores = []

        for r in results:
            cr = self._compute_context_relevance(r["question"], r.get("retrieved_contexts", []))
            af = self._compute_faithfulness(r["rag_answer"], r.get("retrieved_contexts", []))
            context_relevances.append(cr)
            answer_faithfulnesses.append(af)
            response_times.append(r.get("response_time", 0))
            retrieval_times.append(r.get("retrieval_time", 0))
            query_times.append(r.get("query_time", 0))
            generation_times.append(r.get("generation_time", 0))
            total_times.append(r.get("total_time", r.get("response_time", 0)))
            answer_lower = r["rag_answer"].lower()
            no_answer = "未找到" in r["rag_answer"] or "no sufficiently relevant information" in answer_lower
            success = 1 if (r.get("has_context", False) and not no_answer) else 0
            successes.append(success)
            if "llm_only_answer" in r:
                rag_vs_llm_scores.append(
                    self._compare_with_llm_only(r["rag_answer"], r["llm_only_answer"], r["question"])
                )

        answer_relevance = mean(
            [cr * af for cr, af in zip(context_relevances, answer_faithfulnesses)]
        ) if context_relevances else 0.0
        accuracy_result = self.evaluate_accuracy(results)

        return RAGEvaluationResult(
            context_relevance=float(mean(context_relevances)) if context_relevances else 0.0,
            answer_faithfulness=float(mean(answer_faithfulnesses)) if answer_faithfulnesses else 0.0,
            answer_relevance=float(answer_relevance),
            avg_response_time=float(mean(response_times)) if response_times else 0.0,
            success_rate=float(mean(successes)) if successes else 0.0,
            accuracy=float(accuracy_result["accuracy"]),
            avg_retrieval_time=float(mean(retrieval_times)) if retrieval_times else 0.0,
            avg_query_time=float(mean(query_times)) if query_times else 0.0,
            avg_generation_time=float(mean(generation_times)) if generation_times else 0.0,
            avg_total_time=float(mean(total_times)) if total_times else 0.0,
            rag_vs_llm_improvement=float(mean(rag_vs_llm_scores)) if rag_vs_llm_scores else 0.0,
            details={
                "individual_scores": results,
                "total_queries": len(results),
                "successful_queries": sum(successes),
                "accuracy": accuracy_result,
            },
        )

    def print_evaluation_report(self, result: RAGEvaluationResult):
        print("\n" + "=" * 60)
        print("RAG系统评估报告")
        print("=" * 60)
        print("检索质量:")
        print(f"   - 上下文相关性: {result.context_relevance:.2%}")
        print(f"   - 答案忠实度: {result.answer_faithfulness:.2%}")
        print(f"   - 答案相关性: {result.answer_relevance:.2%}")
        print("\n性能指标:")
        print(f"   - 平均响应时间: {result.avg_response_time:.3f}秒")
        print(f"   - 成功率: {result.success_rate:.2%}")
        print(f"   - 严格匹配准确率: {result.accuracy:.2%}")
        print(f"   - 平均检索时间: {result.avg_retrieval_time * 1000:.0f}ms")
        print(f"   - 平均查询时间: {result.avg_query_time * 1000:.0f}ms")
        print(f"   - 平均生成时间: {result.avg_generation_time * 1000:.0f}ms")
        print(f"   - 平均总输出时间: {result.avg_total_time * 1000:.0f}ms")
        print("\n对比分析 (RAG vs 纯LLM):")
        print(f"   - RAG改进分数: {result.rag_vs_llm_improvement:.2%}")
        if result.avg_response_time > 3:
            print("\n警告: 平均响应时间超过3秒限制。")
        else:
            print("\n响应时间符合要求 (<=3秒)。")
        print("\n统计:")
        print(f"   - 总查询数: {result.details['total_queries']}")
        print(f"   - 成功查询数: {result.details['successful_queries']}")
        print("=" * 60)
