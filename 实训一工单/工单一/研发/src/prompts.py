# -*- coding: utf-8 -*-
"""Prompt templates for the PDF QA system."""

from __future__ import annotations

from typing import Literal

PromptType = Literal["general", "numeric", "entity"]
Language = Literal["zh", "en"]


GENERAL_PROMPT = """你是一个专业的中文 PDF 问答助手。请只根据给定资料回答用户问题，答案要简洁、准确、直接。
要求：
1. 如果问题问收入金额，只回答最相关的金额，不要回答百分比。
2. 如果问题问收入占比，只回答百分比，不要回答金额。
3. “军用领域”可对应资料中的“国防领域”“军方客户”“军品业务”“军队用户”等表述。
4. 如果资料中有匹配事实，必须基于资料回答；如果资料不足，回答“根据当前知识库，暂时没有找到足够相关的信息。”
5. 不要输出 HTML、表格标签或调试内容。

资料：
{context}

问题：
{question}
"""


NUMERIC_PROMPT = """你是一个专业的中文 PDF 问答助手。请只根据给定资料回答数值类问题，答案只保留最相关的数字和单位。
要求：
1. 问收入金额时，只回答金额。
2. 问收入占比时，只回答百分比。
3. 问募集资金、注册资本等金额时，只回答对应金额。
4. 不要输出 HTML、表格标签或解释过程。
5. 如果资料不足，回答“根据当前知识库，暂时没有找到足够相关的信息。”

资料：
{context}

问题：
{question}
"""


ENTITY_PROMPT = """你是一个专业的中文 PDF 问答助手。请只根据给定资料回答实体或列表类问题。
要求：
1. 只回答最相关的实体名称、标准名称、行业名称或项目名称。
2. 保持简洁直接，不要复述大段原文。
3. 不要输出 HTML、表格标签或调试内容。
4. 如果资料不足，回答“根据当前知识库，暂时没有找到足够相关的信息。”

资料：
{context}

问题：
{question}
"""


GENERAL_PROMPT_EN = """You are a professional bilingual PDF question-answering assistant. Answer using only the provided material. Keep the answer concise and direct. Do not output HTML tags.

Material:
{context}

Question:
{question}
"""


NUMERIC_PROMPT_EN = """You are a professional bilingual PDF question-answering assistant. Answer the numeric question using only the provided material. Return only the relevant number and unit. Do not output HTML tags.

Material:
{context}

Question:
{question}
"""


ENTITY_PROMPT_EN = """You are a professional bilingual PDF question-answering assistant. Answer the entity/list question using only the provided material. Keep it concise. Do not output HTML tags.

Material:
{context}

Question:
{question}
"""


def get_prompt_template(prompt_type: PromptType, language: Language = "zh") -> str:
    if language == "en":
        if prompt_type == "numeric":
            return NUMERIC_PROMPT_EN
        if prompt_type == "entity":
            return ENTITY_PROMPT_EN
        return GENERAL_PROMPT_EN
    if prompt_type == "numeric":
        return NUMERIC_PROMPT
    if prompt_type == "entity":
        return ENTITY_PROMPT
    return GENERAL_PROMPT


def build_prompt(question: str, context: str, prompt_type: PromptType = "general", language: Language = "zh") -> str:
    return get_prompt_template(prompt_type, language=language).format(question=question, context=context)
