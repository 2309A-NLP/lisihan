# -*- coding: utf-8 -*-
"""Rule based entity and relation extraction for Graph RAG.

The extractor is intentionally local and deterministic. It is not meant to be a
complete Chinese information extraction system; it captures the high-value
entities and relations that appear in annual reports and prospectuses.

人工智能 NLP-RAG-基于 Graph RAG 实现金融问答
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from src.document import Document


@dataclass(frozen=True)
class ExtractedEntity:
    name: str
    type: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExtractedRelation:
    source: str
    relation: str
    target: str
    confidence: float
    evidence: str
    source_file: str
    page: int
    chunk_id: str


COMMON_METRICS = [
    "营业收入",
    "主营业务收入",
    "净利润",
    "归属于母公司股东的净利润",
    "信用减值损失",
    "注册资本",
    "募集资金",
    "发行股数",
    "总股本",
    "资本充足率",
    "不良贷款率",
    "资产总额",
    "负债总额",
]

COMMON_STRATEGIES = [
    "哑铃型",
    "资产配置策略",
    "资本结构优化",
    "绿色金融",
    "科技金融",
    "资产负债管理",
    "长期利率债",
    "权益类资产",
]

COMMON_ORG_TERMS = [
    "董事会",
    "监事会",
    "股东大会",
    "分公司",
    "证券分公司",
    "营业部",
    "委员会",
    "机构与交易业务委员会",
    "组织架构",
    "组织机构",
]

COMPANY_SUFFIXES = (
    "股份有限公司",
    "有限责任公司",
    "集团股份有限公司",
    "银行股份有限公司",
    "证券股份有限公司",
    "保险集团股份有限公司",
)

SOURCE_NAME_STOPWORDS = {
    "年度报告",
    "招股说明书",
    "年",
    "年度",
    "报告",
    "摘要",
}

PERSON_STOPWORDS = {
    "致辞",
    "目录",
    "报告",
    "保证",
    "负责人",
    "董事会",
    "监事会",
    "签字",
    "签字确认",
    "批准",
    "确认",
    "法定代",
}

PERSON_BAD_PARTS = (
    "法定代",
    "直接沟",
    "签字",
    "批准",
    "确认",
    "负责人",
    "党委书",
    "总经理及",
)


class GraphEntityExtractor:
    """Extracts graph entities and relations from parsed PDF chunks."""

    _company_patterns = [
        re.compile(r"([\u4e00-\u9fffA-Za-z0-9（）()]{2,40}(?:股份有限公司|有限责任公司|集团股份有限公司))"),
    ]
    _person_patterns = [
        re.compile(r"(?:法定代表人|董事长|总经理|负责人)[：:\s]*([\u4e00-\u9fff]{2,4})"),
        re.compile(r"([\u4e00-\u9fff]{2,4})(?:先生|女士)(?:担任|任|为)?(?:董事长|总经理|法定代表人)"),
    ]
    _amount_patterns = [
        re.compile(
            r"(营业收入|主营业务收入|归属于母公司股东的净利润|净利润|信用减值损失|注册资本|募集资金|发行股数)"
            r"[^。；;\n]{0,40}?([0-9,，.]+\s*(?:万|亿)?\s*元|[0-9,，.]+\s*(?:万|亿)?\s*股|[0-9.]+%)"
        ),
    ]

    def extract(self, documents: Sequence[Document]) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        entities: dict[tuple[str, str], ExtractedEntity] = {}
        relations: dict[tuple[str, str, str, str, int, str], ExtractedRelation] = {}

        for doc in documents:
            content = self._clean(doc.page_content)
            if not content:
                continue
            metadata = doc.metadata or {}
            source_file = str(metadata.get("source_file", ""))
            page = int(metadata.get("page", 0) or 0)
            chunk_id = str(metadata.get("chunk_id", ""))

            doc_entities = self._extract_entities(content, metadata)
            for entity in doc_entities:
                entities[(entity.name, entity.type)] = entity

            for relation in self._extract_relations(content, doc_entities, source_file, page, chunk_id):
                key = (
                    relation.source,
                    relation.relation,
                    relation.target,
                    relation.source_file,
                    relation.page,
                    relation.chunk_id,
                )
                relations[key] = relation

        return list(entities.values()), list(relations.values())

    def _clean(self, text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    def _extract_entities(self, content: str, metadata: dict) -> list[ExtractedEntity]:
        found: dict[tuple[str, str], ExtractedEntity] = {}

        for company in self._candidate_companies(content, metadata):
            found[(company, "company")] = ExtractedEntity(company, "company", self._company_aliases(company))

        for pattern in self._person_patterns:
            for match in pattern.finditer(content):
                name = self._normalize_name(match.group(1))
                if name:
                    found[(name, "person")] = ExtractedEntity(name, "person")

        for metric in COMMON_METRICS:
            if metric in content:
                found[(metric, "metric")] = ExtractedEntity(metric, "metric")

        for strategy in COMMON_STRATEGIES:
            if strategy in content:
                found[(strategy, "strategy")] = ExtractedEntity(strategy, "strategy")

        for org in COMMON_ORG_TERMS:
            if org in content:
                found[(org, "organization")] = ExtractedEntity(org, "organization")

        return list(found.values())

    def _candidate_companies(self, content: str, metadata: dict) -> list[str]:
        names = []
        for key in ("company_name", "short_name", "source_file"):
            raw = str(metadata.get(key, "") or "")
            if raw and raw.endswith(".pdf"):
                raw = raw[:-4]
            if raw:
                names.extend(self._split_source_name(raw))

        for pattern in self._company_patterns:
            for match in pattern.finditer(content):
                names.append(match.group(1))

        cleaned = []
        for name in names:
            item = self._normalize_company(name)
            if item and item not in cleaned:
                cleaned.append(item)
        return cleaned[:3]

    def _split_source_name(self, value: str) -> list[str]:
        if "__" in value:
            parts = [part for part in value.split("__") if part]
            candidates = []
            if len(parts) >= 2:
                candidates.append(parts[1])
            if len(parts) >= 4:
                candidates.append(parts[3])
            return candidates
        if value.startswith("招股说明书"):
            return [value]
        parts = [part for part in re.split(r"__|_|-|—|\s+", value) if part]
        return [
            part
            for part in parts
            if len(part) >= 2
            and not re.fullmatch(r"\d+", part)
            and not re.fullmatch(r"\d{4}.*", part)
            and part not in SOURCE_NAME_STOPWORDS
        ]

    def _normalize_company(self, name: str) -> str:
        name = re.sub(r"[\s,，。；;：:]+", "", name or "")
        name = re.sub(r"^(关于|公司|本行|本集团|发行人|报告期内|第[一二三四五六七八九十]+章)", "", name)
        for marker in ("获得", "筹建", "关于", "与", "及", "投资", "控股", "收购"):
            if marker in name and len(name) > 14:
                name = name[name.rfind(marker) + len(marker) :]
        name = re.sub(r"(年度报告|招股说明书|摘要)$", "", name)
        if re.fullmatch(r"\d+", name):
            return ""
        if re.fullmatch(r"\d{4}年?", name):
            return ""
        if any(mark in name for mark in ("第", "章", "节", "目录", "释义", "单位", "人民币")) and not any(
            suffix in name for suffix in COMPANY_SUFFIXES
        ):
            return ""
        if len(name) < 2 or len(name) > 45:
            return ""
        if not any(suffix in name for suffix in COMPANY_SUFFIXES) and len(name) > 8:
            return ""
        return name

    def _normalize_name(self, name: str) -> str:
        name = re.sub(r"[^\u4e00-\u9fff]", "", name or "")
        name = re.sub(r"(保证|担任|现任|曾任|负责|主管|分管).*$", "", name)
        if len(name) >= 4 and name.endswith("保"):
            name = name[:-1]
        if len(name) >= 4 and name.endswith("负"):
            name = name[:-1]
        if any(part in name for part in PERSON_BAD_PARTS):
            return ""
        if name[:1] in {"的", "及", "或", "将", "与", "为", "由", "对"}:
            return ""
        if name in PERSON_STOPWORDS:
            return ""
        if 2 <= len(name) <= 4:
            return name
        return ""

    def _company_aliases(self, company: str) -> tuple[str, ...]:
        aliases = []
        suffixes = ["股份有限公司", "有限责任公司", "集团股份有限公司"]
        for suffix in suffixes:
            if company.endswith(suffix):
                alias = company[: -len(suffix)]
                if len(alias) >= 2:
                    aliases.append(alias)
        for tail in ("银行", "证券", "保险", "人寿", "太保", "平安"):
            if tail in company and len(company) > len(tail):
                aliases.append(company[-min(len(company), 4) :])
        return tuple(dict.fromkeys(aliases))

    def _extract_relations(
        self,
        content: str,
        entities: Iterable[ExtractedEntity],
        source_file: str,
        page: int,
        chunk_id: str,
    ) -> list[ExtractedRelation]:
        entity_list = list(entities)
        companies = [item.name for item in entity_list if item.type == "company"]
        anchors = companies or [source_file[:-4] if source_file.endswith(".pdf") else source_file or "文档"]
        relations: list[ExtractedRelation] = []
        primary_anchor = anchors[0] if anchors else (source_file[:-4] if source_file.endswith(".pdf") else source_file or "文档")

        for entity in entity_list:
            if entity.type == "company":
                continue
            if entity.type == "person" and entity.name in content:
                relations.append(self._relation(primary_anchor, "关联人物", entity.name, content, source_file, page, chunk_id, 0.72))
            elif entity.type == "strategy" and entity.name in content and self._is_useful_strategy_context(content, entity.name):
                relations.append(self._relation(primary_anchor, "采用策略", entity.name, content, source_file, page, chunk_id, 0.8))

        for person in self._person_patterns:
            for match in person.finditer(content):
                name = self._normalize_name(match.group(1))
                prefix = match.group(0)
                relation = "法定代表人" if "法定代表人" in prefix else "管理人员"
                if name:
                    relations.append(self._relation(primary_anchor, relation, name, content, source_file, page, chunk_id, 0.9))

        for pattern in self._amount_patterns:
            for match in pattern.finditer(content):
                metric = self._clean(match.group(1))
                value = self._clean(match.group(2))
                if metric and value:
                    relations.append(self._relation(primary_anchor, metric, value, content, source_file, page, chunk_id, 0.86))

        if "同比增长" in content:
            for metric in COMMON_METRICS:
                if metric in content:
                    value_match = re.search(r"同比增长[^0-9]{0,10}([0-9.]+%)", content)
                    if value_match:
                        relations.append(
                            self._relation(
                                primary_anchor,
                                f"{metric}同比增长",
                                value_match.group(1),
                                content,
                                source_file,
                                page,
                                chunk_id,
                                0.86,
                            )
                        )

        for strategy in COMMON_STRATEGIES:
            if strategy in content and self._is_useful_strategy_context(content, strategy):
                relations.append(self._relation(primary_anchor, "采用策略", strategy, content, source_file, page, chunk_id, 0.8))

        for org in COMMON_ORG_TERMS:
            if org in content:
                rel = "调整组织架构" if "调整" in content and ("组织" in content or "架构" in content) else "涉及组织"
                if rel == "调整组织架构" or any(keyword in content for keyword in ("组织架构", "分公司", "证券分公司", "机构与交易业务委员会")):
                    relations.append(self._relation(primary_anchor, rel, org, content, source_file, page, chunk_id, 0.78))

        relations.extend(self._extract_business_relations(content, primary_anchor, source_file, page, chunk_id))
        return relations

    def _extract_business_relations(
        self,
        content: str,
        primary_anchor: str,
        source_file: str,
        page: int,
        chunk_id: str,
    ) -> list[ExtractedRelation]:
        """从金融研报/招股书原文中抽取可追溯的业务关系。"""
        relations: list[ExtractedRelation] = []

        def add(relation: str, target: str, confidence: float = 0.84, source: str | None = None) -> None:
            target = self._clean_target(target)
            if target:
                relations.append(
                    self._relation(source or primary_anchor, relation, target, content, source_file, page, chunk_id, confidence)
                )

        for match in re.finditer(r"(?:公司|兴图新科)[^。；]{0,25}?成为([^。；]{2,35}?领域)的重要供应商", content):
            add("成为重要供应商领域", match.group(1), 0.9)

        for match in re.finditer(r"参与制定了[^。；]{0,40}?(?:《([^》]+)》|即\s*([^。；）]+))", content):
            standard = match.group(1) or match.group(2)
            add("参与制定技术标准", standard, 0.9)

        for match in re.finditer(r"[“\"]([^”\"]{4,60}?工程)[”\"][^。；]{0,40}?荣获国家科技进步一等奖", content):
            add("荣获国家科技进步一等奖", match.group(1), 0.9)

        upstream_match = re.search(r"上游(?:涉及|行业即)([^。；]{4,90}?)(?:，|。|；)", content)
        if upstream_match:
            for item in self._split_targets(upstream_match.group(1)):
                if any(keyword in item for keyword in ("企业", "行业", "制造")):
                    add("上游涉及", item, 0.86)

        downstream_match = re.search(r"下游(?:行业)?[^。；]{0,20}?主要包括([^。；]{4,90}?)(?:。|；|，此外|等行业)", content)
        if downstream_match:
            for item in self._split_targets(downstream_match.group(1)):
                add("下游包括", item, 0.86)

        project_sentence_match = re.search(r"募集资金拟投资的项目分别为([^。；]{4,120})", content)
        if project_sentence_match:
            for item in self._split_targets(project_sentence_match.group(1)):
                add("募集资金拟投资项目", item, 0.9)

        issue_match = re.search(
            r"发行股数[^0-9]{0,20}([0-9,，.]+万股)[^。；]{0,40}?占发行后总股本(?:的比例)?(?:为)?\s*([0-9.]+%)",
            content,
        )
        if issue_match:
            add("发行股数", issue_match.group(1), 0.92)
            add("占发行后总股本比例", issue_match.group(2), 0.92)

        total_share_match = re.search(r"发行后总股本(?:为)?\s*([0-9,，.]+\s*万股)", content)
        if total_share_match:
            add("发行后总股本", total_share_match.group(1), 0.88)

        relations.extend(self._extract_table_relations(content, primary_anchor, source_file, page, chunk_id))

        return relations

    def _extract_table_relations(
        self,
        content: str,
        primary_anchor: str,
        source_file: str,
        page: int,
        chunk_id: str,
    ) -> list[ExtractedRelation]:
        """从已解析的 HTML 表格中动态抽取关系，不写死具体答案值。"""
        relations: list[ExtractedRelation] = []
        all_rows = self._html_table_rows(content)
        if not all_rows:
            return relations

        def add(source: str, relation: str, target: str, confidence: float = 0.9) -> None:
            target = self._clean_target(target)
            source = self._clean_target(source) or primary_anchor
            if target:
                relations.append(self._relation(source, relation, target, content, source_file, page, chunk_id, confidence))

        if "本次募集资金拟投资" in content:
            project_rows = self._html_table_rows(self._section_after(content, "本次募集资金拟投资"))
            if not project_rows:
                project_rows = all_rows
            has_project_header = any(any("项目名称" in cell for cell in cells) for cells in project_rows[:3])
            if not has_project_header:
                project_rows = []
            for cells in project_rows:
                if len(cells) < 2 or not re.fullmatch(r"\d+", cells[0]):
                    continue
                project = cells[1]
                if self._is_valid_table_value(project):
                    add(primary_anchor, "募集资金拟投资项目", project, 0.92)
                if len(cells) >= 3 and self._is_valid_table_value(cells[2]):
                    add(project, "计划总投资", cells[2], 0.86)

        if "发行股数" in content:
            for cells in all_rows:
                for idx, cell in enumerate(cells):
                    if "发行股数" in cell and idx + 1 < len(cells):
                        value_text = cells[idx + 1]
                        amount = re.search(r"([0-9,，.]+万股)", value_text)
                        ratio = re.search(r"([0-9.]+%)", value_text)
                        if amount:
                            add(primary_anchor, "发行股数", amount.group(1), 0.94)
                        if ratio:
                            add(primary_anchor, "占发行后总股本比例", ratio.group(1), 0.94)

        if "存在控制关系的关联方" in content:
            controlled_section = self._section_between(content, "存在控制关系的关联方", "不存在控制关系的关联方")
            for cells in self._html_table_rows(controlled_section):
                if len(cells) >= 3 and "持股比例" not in cells[1] and re.search(r"[0-9.]+%", cells[1]):
                    related_party, ratio, relation_to_company = cells[0], cells[1], cells[2]
                    if self._is_valid_table_value(related_party):
                        add(primary_anchor, "存在控制关系关联方", related_party, 0.94)
                        add(related_party, "持股比例", ratio, 0.94)
                        add(related_party, "本公司关系", relation_to_company, 0.94)

        if "不存在控制关系的关联方" in content:
            non_controlled_section = self._section_after(content, "不存在控制关系的关联方")
            for cells in self._html_table_rows(non_controlled_section):
                if len(cells) >= 2 and "企业名称" not in cells[0] and self._is_valid_table_value(cells[0]):
                    add(primary_anchor, "不存在控制关系关联方", cells[0], 0.92)
                    if self._is_valid_table_value(cells[1]):
                        add(cells[0], "本公司关系", cells[1], 0.88)

        return relations

    def _section_after(self, content: str, start_marker: str) -> str:
        if start_marker not in content:
            return content
        return content.split(start_marker, 1)[1]

    def _section_between(self, content: str, start_marker: str, end_marker: str) -> str:
        section = self._section_after(content, start_marker)
        if end_marker in section:
            section = section.split(end_marker, 1)[0]
        return section

    def _html_table_rows(self, content: str) -> list[list[str]]:
        rows: list[list[str]] = []
        for row_html in re.findall(r"<tr[^>]*>(.*?)</tr>", content or "", flags=re.I | re.S):
            cells = [
                self._clean_target(re.sub(r"<[^>]+>", " ", cell_html))
                for cell_html in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, flags=re.I | re.S)
            ]
            cells = [cell for cell in cells if cell]
            if cells:
                rows.append(cells)
        return rows

    def _is_valid_table_value(self, value: str) -> bool:
        value = self._clean_target(value)
        if not value:
            return False
        if value in {"序号", "项目名称", "计划总投资万元", "企业名称", "关联方名称", "持股比例", "与本公司关系"}:
            return False
        if re.fullmatch(r"[-—/无]+", value):
            return False
        return True

    def _split_targets(self, text: str) -> list[str]:
        text = re.sub(r"<[^>]+>", " ", text or "")
        text = re.sub(r"(覆盖范围广泛|竞争充分|采购便利|下游行业为|各类终端用户)", " ", text)
        parts = re.split(r"、|，|,|以及|；|;|等", text)
        return [self._clean_target(part) for part in parts if self._clean_target(part)]

    def _clean_target(self, value: str) -> str:
        value = re.sub(r"<[^>]+>", " ", value or "")
        value = re.sub(r"\s+", "", value)
        value = value.strip(" ：:，,。；;（）()[]【】《》\"'“”")
        value = value.rstrip("和")
        value = re.sub(r"^(即|为|包括|相关的|信息系统相关的)", "", value)
        value = re.sub(r"(企业|行业企业)$", lambda match: match.group(0), value)
        if not value or value in {"公司", "本公司", "发行人", "项目名称", "计划总投资"}:
            return ""
        if len(value) > 50:
            return ""
        return value

    def _relation(
        self,
        source: str,
        relation: str,
        target: str,
        evidence: str,
        source_file: str,
        page: int,
        chunk_id: str,
        confidence: float,
    ) -> ExtractedRelation:
        return ExtractedRelation(
            source=source,
            relation=relation,
            target=target,
            confidence=confidence,
            evidence=evidence[:600],
            source_file=source_file,
            page=page,
            chunk_id=chunk_id,
        )

    def _is_useful_strategy_context(self, content: str, strategy: str) -> bool:
        if strategy in {"哑铃型", "资产配置策略", "资产负债管理", "长期利率债", "权益类资产"}:
            return True
        return any(keyword in content for keyword in ("策略", "投资", "配置", "绿色金融", "科技金融", "资本结构优化"))
