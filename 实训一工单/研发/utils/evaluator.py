# -*- coding: utf-8 -*-
"""RAG evaluation module."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, List


@dataclass
class RAGEvaluationResult:
    """RAG evaluation result."""

    context_relevance: float
    answer_faithfulness: float
    answer_relevance: float
    avg_response_time: float
    success_rate: float
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
        answer_sentences = answer.split("。")
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
        if "未找到" in answer or "无法提供" in answer:
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
        rag_admits_unknown = "未找到" in rag_answer
        llm_admits_unknown = "未找到" in llm_only_answer or "不确定" in llm_only_answer
        if rag_admits_unknown and not llm_admits_unknown:
            return 0.3
        if rag_has_data and not llm_has_data:
            return 0.8
        if rag_has_data and llm_has_data:
            return 0.5
        return 0.0

    def _extract_keywords(self, text: str) -> List[str]:
        import re

        words = re.findall(r"[\w\u4e00-\u9fff]+", text)
        stopwords = {"的", "了", "是", "在", "和", "中", "或", "等", "如何", "什么", "哪些"}
        keywords = [w for w in words if w not in stopwords and len(w) > 1]
        return keywords[:10]

    def evaluate_batch(self, results: List[Dict]) -> RAGEvaluationResult:
        context_relevances = []
        answer_faithfulnesses = []
        response_times = []
        successes = []
        rag_vs_llm_scores = []

        for r in results:
            cr = self._compute_context_relevance(r["question"], r.get("retrieved_contexts", []))
            af = self._compute_faithfulness(r["rag_answer"], r.get("retrieved_contexts", []))
            context_relevances.append(cr)
            answer_faithfulnesses.append(af)
            response_times.append(r.get("response_time", 0))
            success = 1 if (r.get("has_context", False) and "未找到" not in r["rag_answer"]) else 0
            successes.append(success)
            if "llm_only_answer" in r:
                rag_vs_llm_scores.append(
                    self._compare_with_llm_only(r["rag_answer"], r["llm_only_answer"], r["question"])
                )

        answer_relevance = mean(
            [cr * af for cr, af in zip(context_relevances, answer_faithfulnesses)]
        ) if context_relevances else 0.0

        return RAGEvaluationResult(
            context_relevance=float(mean(context_relevances)) if context_relevances else 0.0,
            answer_faithfulness=float(mean(answer_faithfulnesses)) if answer_faithfulnesses else 0.0,
            answer_relevance=float(answer_relevance),
            avg_response_time=float(mean(response_times)) if response_times else 0.0,
            success_rate=float(mean(successes)) if successes else 0.0,
            rag_vs_llm_improvement=float(mean(rag_vs_llm_scores)) if rag_vs_llm_scores else 0.0,
            details={
                "individual_scores": results,
                "total_queries": len(results),
                "successful_queries": sum(successes),
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
