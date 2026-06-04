# -*- coding: utf-8 -*-
"""Prompt templates for the PDF QA system."""

from __future__ import annotations

from typing import Literal

PromptType = Literal["general", "numeric", "percentage", "entity"]
AnswerLanguage = Literal["zh", "en"]


BASE_RULES = """你是一个严谨的中文招股书问答助手。只能依据给定资料回答，不要加入资料外信息。
要求：
1. 回答要简洁直接，不复述无关上下文。
2. 如果问题问金额，只输出相关金额；如果问比例/比重/占比，只输出百分比。
3. 如果问题问“哪些/哪个”，只输出实体、行业、领域或事项名称。
4. 禁止输出银行授信、抵押、声明、签字页等与问题无关的信息。
5. 如果资料不足，回答“根据当前知识库，暂时没有找到足够相关的信息。”。
"""

GENERAL_PROMPT = BASE_RULES + """
资料：
{context}

问题：
{question}
"""

NUMERIC_PROMPT = BASE_RULES + """
这是数值类问题。请只抽取最相关的数字、金额、日期或数量，不要解释。

资料：
{context}

问题：
{question}
"""

PERCENTAGE_PROMPT = BASE_RULES + """
这是比例类问题。请只抽取百分比，不要输出金额。

资料：
{context}

问题：
{question}
"""

ENTITY_PROMPT = BASE_RULES + """
这是实体/列表类问题。请只抽取最相关的名称或列表，不要解释。

资料：
{context}

问题：
{question}
"""

LANGUAGE_INSTRUCTIONS = {
    "zh": "请用中文回答。",
    "en": "Answer in English only. Preserve numbers, dates, percentages and names accurately.",
}


def get_prompt_template(prompt_type: PromptType) -> str:
    if prompt_type == "percentage":
        return PERCENTAGE_PROMPT
    if prompt_type == "numeric":
        return NUMERIC_PROMPT
    if prompt_type == "entity":
        return ENTITY_PROMPT
    return GENERAL_PROMPT


def build_prompt(
    question: str,
    context: str,
    prompt_type: PromptType = "general",
    answer_language: AnswerLanguage = "zh",
) -> str:
    prompt = get_prompt_template(prompt_type).format(question=question, context=context)
    return f"{prompt}\n\n回答语言要求：{LANGUAGE_INSTRUCTIONS.get(answer_language, LANGUAGE_INSTRUCTIONS['zh'])}\n"
