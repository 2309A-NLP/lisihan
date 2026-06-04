"""
RAG系统核心模块 - 高性能版
目标：响应时间 < 3秒，召回率 > 95%，精确率 > 80%
"""

import os
import hashlib
import json
import logging
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

from app.core.cleaner import DataCleaner
from app.core.logging import log_exception

logger = logging.getLogger(__name__)

# ========== 依赖导入 ==========
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "cached_download") and hasattr(huggingface_hub, "hf_hub_download"):
        huggingface_hub.cached_download = huggingface_hub.hf_hub_download
except Exception:
    pass

SentenceTransformer = None
SENTENCE_TRANSFORMERS_IMPORT_ERROR = None

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:
    SENTENCE_TRANSFORMERS_IMPORT_ERROR = exc

from config.config import settings
from app.services.knowledge_service import KnowledgeService

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import jieba
except Exception:
    jieba = None

try:
    import requests
except Exception:
    requests = None

# ========== 默认知识库（确保100%召回率）==========
DEFAULT_KNOWLEDGE = [
    # 医生角色 (1)
    {"role_ids": [1], "content": "高血压注意事项：低盐饮食、规律运动、定期监测血压、戒烟限酒、保持情绪稳定。"},
    {"role_ids": [1], "content": "被狗咬伤处理：立即用肥皂水冲洗15分钟，尽快接种狂犬疫苗，必要时注射免疫球蛋白。"},
    {"role_ids": [1], "content": "糖尿病预防：控制饮食、适量运动、定期测血糖、保持健康体重、避免高糖食物。"},
    {"role_ids": [1], "content": "头疼/头痛处理建议：先休息、补充水分，测量体温和血压，避免熬夜、饮酒和过度用眼；如果头痛突然剧烈、持续加重，或伴随发热、呕吐、肢体麻木无力、意识异常、视物模糊、外伤后头痛，应立即就医。"},
    {"role_ids": [1], "content": "贫血注意事项：注意查明原因，常见原因包括缺铁、维生素B12或叶酸不足、慢性失血等；饮食可适当增加瘦肉、动物肝脏、蛋类、豆制品、深绿色蔬菜等，搭配富含维生素C的食物；如果乏力、心慌、气短、头晕明显，或月经过多、黑便、便血，应及时就医检查血常规和铁代谢等。"},

    # 律师角色 (2)
    {"role_ids": [2], "content": "民法典第1043条：夫妻应当互相忠实，互相尊重，互相关爱；家庭成员应当敬老爱幼，互相帮助。"},
    {"role_ids": [2], "content": "合同是民事主体之间设立、变更、终止民事法律关系的协议。合同各方应按约定履行义务。"},

    # 心理医生角色 (3)
    {"role_ids": [3], "content": "焦虑缓解方法：深呼吸放松、正念冥想、规律运动、保证睡眠、寻求社交支持。"},

    # 教师角色 (4)
    {"role_ids": [4], "content": "因材施教：根据学生个体差异，采用针对性教学方法，注重个性化指导，激发学习兴趣。"},

    # 科学家角色 (5)
    {"role_ids": [5], "content": "科学方法：观察现象→提出假设→设计实验→收集数据→分析验证→得出结论。"},

    # 股票分析师角色 (6)
    {"role_ids": [6], "content": "股票投资注意：分散投资、控制风险、长期持有、关注基本面、不追涨杀跌。"},

    # 英语助手角色 (7)
    {"role_ids": [7],"content": "现在进行时结构：主语 + be动词(am/is/are) + doing。be动词有三种形式：am、is、are。doing是动词的现在分词形式。表示正在进行的动作。例如：I am eating(我正在吃)、She is reading(她正在读)。"},
    # 虚拟朋友角色 (8)
    {"role_ids": [8], "content": "心情不好时：找人倾诉、听音乐、做喜欢的事、运动、保证充足睡眠。"},

    # 金融理财师角色 (9)
    {"role_ids": [9], "content": "理财原则：量入为出、分散投资、长期规划、控制风险、预留应急资金。"},
]

# ========== 角色文件匹配规则 ==========
ROLE_FILE_RULES = {
    1: ("高血压", "医疗", "医生", "健康", "糖尿病", "狂犬病", "贫血", "腰痛", "流感", "哮喘", "乳腺癌", "失明", "视力受损"),
    2: ("民法典", "律师", "法律", "宪法", "合同法", "国务院组织法", "妇女权益保护法", "工会法", "未成年人保护法", "童工"),
    3: ("心理", "心理健康", "心理治疗", "焦虑", "自闭症", "职场心理"),
    4: ("教育", "教师", "课程", "课程标准", "课程方案", "义务教育", "语文", "数学"),
    5: ("科学", "科技", "天体", "物理", "服务规范", "人力资源"),
    6: ("证券", "股票", "投资", "分析师", "投资顾问"),
    7: ("英语", "现在进行时"),
    8: ("虚拟",),
    9: ("理财", "金融", "财富", "财富管理", "商业银行"),
}

# ========== 常量定义 ==========
VAGUE_MEDICAL_TERMS = ("不舒服", "不适", "难受")
SPECIFIC_MEDICAL_TERMS = (
    "头疼", "头痛", "发烧", "咳嗽", "高血压", "糖尿病",
    "肚子", "腹痛", "腹泻", "腹胀", "胃痛", "胃胀", "恶心", "呕吐", "拉肚子",
    "贫血", "腰痛", "流感", "哮喘", "乳腺癌", "视力", "失明",
)
ROLE_VOCATIVES = ("医生", "律师", "老师", "教师", "心理医生", "科学家", "股票分析师", "英语学习助手", "虚拟朋友", "金融理财师")
RETRIEVAL_STOPWORDS = {"我", "你", "他", "她", "它", "我们", "你们", "他们", "现在", "今天", "应该", "怎么", "怎么办", "如何", "什么", "请问", "一下"}
IMPORTANT_SINGLE_CHAR_TERMS = {"狗", "猫", "犬", "咬", "疼", "痛", "胃", "腹"}
DOMAIN_RETRIEVAL_TERMS = tuple(sorted({
    *SPECIFIC_MEDICAL_TERMS,
    *VAGUE_MEDICAL_TERMS,
    "注意事项", "处理", "预防", "饮食", "清淡", "休息", "就医", "发热", "便血",
    "民法典", "合同", "夫妻", "权益", "保护", "劳动", "未成年人", "离婚", "婚姻", "抚养", "财产分割", "儿童", "童工", "未成年", "雇佣", "用工",
    "焦虑", "心理", "自闭症", "职场",
    "课程", "教学", "学习", "语文", "数学", "英语",
    "科学", "科技", "天体", "物理",
    "证券", "股票", "投资", "分析师", "理财", "金融", "财富",
}, key=len, reverse=True))
MIN_RELEVANCE_SCORE = 1.0

QUERY_EXPANSIONS = {
    "头疼": ["头痛", "头部疼痛", "头疼处理", "头痛处理"],
    "头痛": ["头疼", "头部疼痛", "头痛处理", "头疼处理"],
    "发烧": ["发热", "体温升高"],
    "肚子疼": ["腹痛", "胃痛"],
    "拉肚子": ["腹泻"],
}

ROLE_STYLE_GUIDES = {
    "医生": "像懂医学的朋友一样说话。先接住用户的不舒服，再把知识库里最重要的注意点讲成可执行的小建议，最后轻轻提醒危险信号。不要吓人，不要直接下诊断。",
    "律师": "像靠谱的法律朋友一样说话。先把大方向讲明白，再挑证据、材料、下一步里最关键的点说；提醒具体结果要看事实和当地规定。",
    "心理医生": "像温和陪伴的朋友一样说话。先回应情绪，再给一两个马上能试的小方法；少讲大道理，不评判，风险情况要建议线下专业支持。",
    "教师": "像耐心的老师朋友一样说话。先说这题卡在哪里，再用小例子讲清楚，最后给一个轻量练习或追问。",
    "科学家": "像爱科普的朋友一样回答。把原理讲准确，但用生活化比喻和短段落，不要像论文。",
    "股票分析师": "像冷静的投研朋友一样回答。先讲倾向，再挑基本面、技术面、情绪或风险里最关键的依据说，不承诺收益。",
    "英语学习助手": "像陪练朋友一样回答。先给自然表达或修改，再用一句话解释原因，最后给一个很小的练习。",
    "虚拟朋友": "像朋友聊天一样自然、轻松、真诚。多一点共情和接话，少一点说教，不要像客服。",
    "金融理财师": "像稳健的理财朋友一样回答。先理解目标和风险，再挑现金流、应急金、配置和风险里最重要的点说。",
}


def _enabled(value) -> bool:
    """把 bool 或字符串配置统一转换为开关。"""
    if isinstance(value, bool):
        return value
    return str(value or "").lower() in {"1", "true", "yes", "on"}


# Function: Coordinate knowledge retrieval, prompt construction, and answer generation.
class RAGSystem:
    """RAG系统"""


    def __init__(self, load_models: bool | None = None):
        self.cleaner = DataCleaner()
        self._added_hashes = set()
        self.embedding_model = None
        self.rerank_model = None
        self.knowledge_cache = []
        self.knowledge_items = []
        self.bm25_index = None
        self.tokenized_corpus = []
        self.query_cache = {}
        self.last_llm_error = ""

        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=2)

        if load_models is None:
            load_models = _enabled(settings.RAG_LOAD_MODELS)

        if load_models:
            self.init_models()
        else:
            logger.info("跳过本地 RAG 模型预加载，使用轻量启动模式")
        self.init_knowledge_base()

    def init_models(self):
        """初始化模型"""
        self.init_embedding_model()
        if _enabled(settings.RAG_LOAD_RERANK):
            self.init_rerank_model()

    def init_embedding_model(self):
        """初始化向量化模型"""
        try:
            if SentenceTransformer is None:
                logger.warning(f"sentence-transformers 不可用，将使用兜底向量: {SENTENCE_TRANSFORMERS_IMPORT_ERROR}")
                self.embedding_model = None
                return
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info("向量化模型初始化成功")
        except Exception as e:
            log_exception(logger, "向量化模型初始化失败", e, model=settings.EMBEDDING_MODEL)

    def init_rerank_model(self):
        """初始化重排序模型"""
        try:
            if SentenceTransformer is None:
                self.rerank_model = None
                return
            self.rerank_model = SentenceTransformer(settings.RERANK_MODEL)
            logger.info("重排序模型初始化成功")
        except Exception as e:
            log_exception(logger, "重排序模型初始化失败", e, model=settings.RERANK_MODEL)

    def init_knowledge_base(self):
        """初始化知识库"""
        try:
            # 这是要检索的知识库
            current_file = os.path.abspath(__file__)
            core_dir = os.path.dirname(current_file)
            app_dir = os.path.dirname(core_dir)
            knowledge_dir = os.path.join(app_dir, 'knowledge', 'data')

            self.knowledge_cache = []
            self.knowledge_items = []
            self._added_hashes = set()

            # # 先加载默认知识（确保100%召回率）
            # for item in DEFAULT_KNOWLEDGE:
            #     self.add_knowledge_segment(item["content"], role_ids=item["role_ids"], source_file="内置知识")

            # 再加载文件知识
            if os.path.exists(knowledge_dir):
                knowledge_service = KnowledgeService()
                knowledge_service.process_knowledge_files(knowledge_dir, self)

            self.init_bm25_index()
            logger.info(f"知识库加载完成，共 {len(self.knowledge_cache)} 条")
        except Exception as e:
            log_exception(logger, "知识库初始化失败", e)

    def add_knowledge_segment(self, content: str, role_ids: list = None, source_file: str = "") -> None:
        """添加知识片段"""
        content = (content or "").strip()
        if not content:
            return

        content_hash = hashlib.md5(content.encode()).hexdigest()
        if content_hash in self._added_hashes:
            return

        self._added_hashes.add(content_hash)
        role_ids = self._normalize_role_ids(role_ids)

        self.knowledge_items.append({
            "content": content,
            "role_ids": role_ids,
            "source_file": source_file,
        })
        self.knowledge_cache.append(content)

    def init_bm25_index(self):
        """初始化BM25索引"""
        try:
            if not self.knowledge_cache:
                return

            self.tokenized_corpus = []
            for segment in self.knowledge_cache:
                tokens = self._tokenize(segment)
                self.tokenized_corpus.append(tokens)

            if BM25Okapi:
                self.bm25_index = BM25Okapi(self.tokenized_corpus)
                logger.info(f"BM25索引构建完成: {len(self.tokenized_corpus)}条")
        except Exception as e:
            log_exception(logger, "BM25索引构建失败", e, segment_count=len(self.knowledge_cache))

    def get_embedding(self, text: str) -> list:
        """获取向量"""
        try:
            if not self.embedding_model:
                digest = hashlib.sha256(text.encode("utf-8")).digest()
                values = [byte / 255.0 for byte in digest]
                repeats = (768 // len(values)) + 1
                return (values * repeats)[:768]
            return self.embedding_model.encode(text).tolist()
        except Exception as exc:
            log_exception(logger, "文本向量化失败", exc, text_preview=str(text or "")[:80])
            return []

    def search_knowledge(self, query: str, top_k: int = 3, role_id: int = None) -> list:
        """检索知识 - 目标 < 500ms"""
        try:
            role_id = self._normalize_role_id(role_id)
            # 缓存检查
            cache_key = f"{query}:{role_id}"
            if cache_key in self.query_cache:
                return self.query_cache[cache_key]

            search_query = self._expand_query(query)

            # 多路召回
            future1 = self.executor.submit(self._bm25_search, search_query, role_id)  # BM25关键词检索
            future2 = self.executor.submit(self._vector_search, search_query, role_id) # 本地关键词/轻量候选检索
            future3 = self.executor.submit(self._milvus_search, search_query, role_id, top_k) # 向量语义检索
            future4 = self.executor.submit(self._milvus_keyword_search, search_query, role_id, top_k) # 文本关键词兜底检索
            # 并行执行多个见多任务，等待每个任务返回结果（有超时控制）
            bm25_results = future1.result(timeout=2)
            vector_results = future2.result(timeout=2)
            milvus_results = future3.result(timeout=5)
            milvus_keyword_results = future4.result(timeout=5)
            # 提取检索关键词
            terms = self._extract_retrieval_terms(search_query)
            # 准备存储综合得分的容器
            scored_results = []
            # 遍历合并后的所有结果
            for item in milvus_keyword_results + milvus_results + bm25_results + vector_results:
                # 复制并提取内容
                scored = dict(item)
                content = scored.get("content", "")
                # 计算关键词匹配得分
                keyword_score = self._keyword_score_from_terms(terms, content)
                # 获得原始检索得分
                source = scored.get("retrieval_source", "")
                base_score = float(scored.get("score") or 0)

                # Milvus 使用的是哈希兜底向量时，距离只能作为候选来源，不能直接当相关性。
                if source == "milvus":
                    base_score = max(base_score, 0.5 if keyword_score > 0 else 0)

                relevance_score = keyword_score + base_score
                scored["relevance_score"] = relevance_score
                scored.setdefault("distance", float(scored.get("distance", scored.get("score", relevance_score)) or 0))
                if relevance_score >= MIN_RELEVANCE_SCORE:
                    scored_results.append(scored)

            source_priority = {
                "milvus_keyword": 3,
                "milvus": 2,
                "bm25": 1,
                "keyword": 1,
            }
            scored_results.sort(
                key=lambda item: self._rerank_key(item, terms, query, source_priority),
                reverse=True,
            )

            # 合并去重，保留真正相关的知识片段。
            seen = set()
            merged = []
            for r in scored_results:
                content = r.get("content", "")
                if content and content not in seen:
                    seen.add(content)
                    merged.append(r)

            # 取前top_k个
            results = merged[:top_k]

            # 缓存结果
            self.query_cache[cache_key] = results
            if len(self.query_cache) > 50:
                self.query_cache.pop(next(iter(self.query_cache)))

            return results

        except Exception as e:
            log_exception(logger, "知识库检索失败", e, query=query, role_id=role_id, top_k=top_k)
            return []

    def _bm25_search(self, query: str, role_id: int = None) -> list:
        """BM25检索"""
        try:
            if not self.tokenized_corpus:
                return []

            candidate_indices = self._get_candidate_indices(role_id)
            if not candidate_indices:
                return []

            query_tokens = self._tokenize(query)

            if self.bm25_index:
                scores = self.bm25_index.get_scores(query_tokens)
                indexed_scores = [(idx, scores[idx]) for idx in candidate_indices if scores[idx] > 0]
            else:
                indexed_scores = []
                for idx in candidate_indices:
                    score = self._keyword_score(query, self.knowledge_cache[idx])
                    if score > 0:
                        indexed_scores.append((idx, score))

            indexed_scores.sort(key=lambda x: x[1], reverse=True)

            results = []
            for idx, score in indexed_scores[:5]:
                results.append({
                    "content": self.knowledge_cache[idx],
                    "score": score,
                    "retrieval_source": "bm25",
                    "source_file": self.knowledge_items[idx].get("source_file", ""),
                })
            return results
        except Exception as exc:
            log_exception(logger, "BM25检索失败", exc, query=query, role_id=role_id)
            return []

    def _vector_search(self, query: str, role_id: int = None) -> list:
        """向量检索（简化版，直接匹配关键词）"""
        try:
            candidate_indices = self._get_candidate_indices(role_id)
            if not candidate_indices:
                return []

            query_tokens = self._extract_retrieval_terms(query)
            results = []

            for idx in candidate_indices:
                content = self.knowledge_cache[idx]
                score = self._keyword_score_from_terms(query_tokens, content)

                if score > 0:
                    results.append({
                        "content": content,
                        "score": score,
                        "retrieval_source": "keyword",
                        "source_file": self.knowledge_items[idx].get("source_file", ""),
                    })

            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:5]
        except Exception as exc:
            log_exception(logger, "关键词检索失败", exc, query=query, role_id=role_id)
            return []

    def _milvus_search(self, query: str, role_id: int = None, top_k: int = 3) -> list:
        """从 Milvus 向量库检索知识片段。"""
        try:
            from app.core.vectorstore import vector_store

            if not vector_store.client:
                vector_store.connect()
            if not vector_store.client:
                return []

            results = vector_store.search(self.get_embedding(query), top_k=max(top_k * 3, top_k))
            if role_id is None:
                return results[:top_k]

            filtered = []
            for item in results:
                role_ids = self._normalize_role_ids(item.get("role_ids", []))
                if not role_ids or role_id in role_ids:
                    item["retrieval_source"] = "milvus"
                    filtered.append(item)
            return filtered[:top_k]
        except Exception as exc:
            log_exception(logger, "Milvus 向量检索失败", exc, query=query, role_id=role_id, top_k=top_k)
            return []

    def _milvus_keyword_search(self, query: str, role_id: int = None, top_k: int = 3) -> list:
        """从 Milvus 拉取知识文本并按关键词评分，作为无真实 embedding 时的可靠兜底。"""
        try:
            from app.core.vectorstore import vector_store

            if not vector_store.client:
                vector_store.connect()
            if not vector_store.client:
                return []

            rows = vector_store.client.query(
                collection_name=vector_store.collection_name,
                filter="id >= 0",
                output_fields=["content", "role_ids", "source_file"],
                limit=5000,
            )
            terms = self._extract_retrieval_terms(query)
            results = []
            for row in rows:
                role_ids = self._normalize_role_ids(row.get("role_ids", []))
                if role_id is not None and role_ids and role_id not in role_ids:
                    continue

                content = row.get("content", "")
                score = self._keyword_score_from_terms(terms, content)
                if score > 0:
                    results.append({
                        "content": content,
                        "role_ids": role_ids,
                        "source_file": row.get("source_file", ""),
                        "score": score,
                        "retrieval_source": "milvus_keyword",
                    })

            results.sort(key=lambda item: item["score"], reverse=True)
            return results[:max(top_k * 3, top_k)]
        except Exception as exc:
            log_exception(logger, "Milvus 关键词检索失败", exc, query=query, role_id=role_id, top_k=top_k)
            return []

    def _get_candidate_indices(self, role_id: int = None) -> list:
        """获取候选索引"""
        role_id = self._normalize_role_id(role_id)
        if role_id is None:
            return list(range(len(self.knowledge_cache)))
        return [i for i, item in enumerate(self.knowledge_items) if role_id in self._normalize_role_ids(item.get("role_ids", []))]

    def generate_response(self, query: str, context: list, role_template: str, **kwargs) -> str:
        """生成回复 - 目标 < 2.5秒"""
        try:
            role_name = self._extract_role_name(role_template)
            knowledge_context = [
                item for item in context
                if item.get("source_file") != "chat_context" and item.get("retrieval_source") != "chat_context"
            ]

            knowledge_context = self._rerank_context(knowledge_context, query)
            direct_answer = self._direct_knowledge_answer(query, knowledge_context, role_name)
            if direct_answer:
                return direct_answer

            # 使用LLM生成
            if context:
                prompt = self._build_prompt(query, context, role_name)
                response = self._quick_llm(prompt)
                if response:
                    return response

            # LLM不可用时，才从知识库片段生成一段自然兜底回复。
            for item in knowledge_context:
                content = item.get("content", "")
                if any(kw in content for kw in self._extract_retrieval_terms(query)):
                    return self._format_answer(content, role_name, query)

            # 兜底回答
            return self._get_fallback_answer(query, role_name)

        except Exception as e:
            log_exception(logger, "RAG生成失败", e, query=query, role_template_preview=str(role_template or "")[:80])
            return self._get_fallback_answer(query, "助手")

    def _quick_llm(self, prompt: str) -> str:
        """快速LLM调用"""
        if not settings.API_KEY or requests is None:
            return ""

        try:
            payload = {
                "model": settings.MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 200,
            }

            response = requests.post(
                settings.API_BASE_URL,
                headers={"Authorization": f"Bearer {settings.API_KEY}", "Content-Type": "application/json"},
                json=payload,
                timeout=(10, 60),
            )

            if response.status_code == 200:
                return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            logger.warning(
                "非流式 LLM 返回非 200 | model=%s | status_code=%s | body_preview=%s",
                settings.MODEL_NAME,
                response.status_code,
                response.text[:200],
            )
        except Exception as exc:
            log_exception(logger, "非流式 LLM 调用失败", exc, model=settings.MODEL_NAME, prompt_preview=prompt[:120])
        return ""

    def stream_llm(self, query: str, context: list, role_template: str):
        """流式调用 LLM；如果流式失败，回退为普通生成结果。"""
        role_name = self._extract_role_name(role_template)
        knowledge_context = [
            item for item in context
            if item.get("source_file") != "chat_context" and item.get("retrieval_source") != "chat_context"
        ]
        knowledge_context = self._rerank_context(knowledge_context, query)
        direct_answer = self._direct_knowledge_answer(query, knowledge_context, role_name)
        if direct_answer:
            yield from self._split_stream_text(direct_answer)
            return

        prompt = self._build_prompt(query, context, role_name)
        if not settings.API_KEY or requests is None:
            fallback = self.generate_response(query, context, role_template)
            yield from self._split_stream_text(fallback)
            return

        try:
            payload = {
                "model": settings.MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.45,
                "max_tokens": 500,
                "stream": True,
            }
            response = requests.post(
                settings.API_BASE_URL,
                headers={"Authorization": f"Bearer {settings.API_KEY}", "Content-Type": "application/json"},
                json=payload,
                timeout=60,
                stream=True,
            )
            if response.status_code != 200:
                fallback = self.generate_response(query, context, role_template)
                yield from self._split_stream_text(fallback)
                return

            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                try:
                    data = json.loads(line)
                    chunk = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if chunk:
                        yield chunk
                except Exception as exc:
                    log_exception(logger, "解析流式 LLM 响应失败", exc, raw_line=raw_line)
                    continue
        except Exception as exc:
            log_exception(logger, "流式 LLM 调用失败", exc, model=settings.MODEL_NAME, prompt_preview=prompt[:120])
            fallback = self.generate_response(query, context, role_template)
            yield from self._split_stream_text(fallback)

    def _split_stream_text(self, text: str, chunk_size: int = 12):
        """把非流式兜底文本拆成小片段返回，保证前端也能逐步显示。"""
        text = text or ""
        for start in range(0, len(text), chunk_size):
            yield text[start:start + chunk_size]

    def _direct_knowledge_answer(self, query: str, knowledge_context: list, role_name: str) -> str:
        """高置信命中知识库时直接回答，避免 LLM 把明确资料说偏。"""
        terms = self._extract_retrieval_terms(query)
        matched = []
        for item in knowledge_context:
            content = item.get("content", "")
            if any(term in content for term in terms):
                matched.append(item)
        if matched:
            return self._format_knowledge_answer(matched[:4], role_name, query)
        return ""

    def build_prompt(self, query: str, context: list, role_template: str) -> str:
        """公开的提示词构建入口，保留给测试和旧调用方使用。"""
        prompt = self._build_prompt(query, context, self._extract_role_name(role_template))
        return f"角色模板：\n{role_template}\n\n{prompt}"

    def _build_prompt(self, query: str, context: list, role_name: str) -> str:
        """构建简洁提示词"""
        knowledge_items = [
            item for item in context
            if item.get("source_file") != "chat_context" and item.get("retrieval_source") != "chat_context"
        ]
        memory_items = [
            item for item in context
            if item.get("source_file") == "chat_context" or item.get("retrieval_source") == "chat_context"
        ]
        relevant_knowledge = [item for item in knowledge_items if self._is_relevant_to_query(query, item.get("content", ""))]
        if self._has_query_terms(query):
            knowledge_items = relevant_knowledge

        if self._is_memory_lookup_query(query):
            memory_items = memory_items[:4]
        else:
            memory_items = [
                item for item in memory_items
                if self._is_relevant_to_query(query, item.get("content", ""))
            ][:2]

        knowledge = "\n".join(
            f"- 来源:{c.get('source_file') or c.get('retrieval_source') or 'unknown'}；内容:{c.get('content', '')[:500]}"
            for c in knowledge_items[:4]
        ) or "无直接相关资料。"
        memory = "\n".join(
            f"- 历史参考:{c.get('content', '')[:300]}"
            for c in memory_items
        ) or "无直接相关历史。"
        style = ROLE_STYLE_GUIDES.get(role_name, "像自然聊天一样回答，语气亲切、清楚，不要机械复述资料。")
        return f"""你是{role_name}。

表达风格：
{style}

当前问题是唯一主线：必须先回答“当前问题”，不要被历史对话或不相关资料带偏。
先阅读知识库资料，按“和当前问题最相关、最可执行、最能提醒风险”的顺序在心里重排，只挑重要内容回答。
历史参考只有在它和当前问题直接相关，或用户明确问“之前/刚才/上面说了什么”时才能使用；不相关历史要忽略。
没有资料支持时要坦诚说明，不要编造。回答要像聊天，不要生硬罗列；可以分成短段落，必要时用“如果……就……”表达判断。

当前问题：
{query}

知识库资料：
{knowledge}

历史参考：
{memory}

回答要求：
1. 用中文自然回答，像正在和用户对话。
2. 第一段必须直接回应当前问题，不要先回答历史里的旧问题。
3. 不要直接照搬知识库原文。
4. 医疗、法律、金融类问题要提醒风险边界，但语气不要吓人。
5. 不要输出“根据知识库”这类系统痕迹。
6. 不要自我介绍，不要说“你好，我是……”，直接回答用户的问题。
回答："""

    def _is_relevant_to_query(self, query: str, content: str) -> bool:
        """用当前问题关键词过滤明显无关的知识和历史。"""
        terms = [
            term for term in self._extract_retrieval_terms(query)
            if len(term) >= 2 and term not in RETRIEVAL_STOPWORDS
        ]
        if not terms:
            return False
        return self._keyword_score_from_terms(terms, content) > 0

    def _is_memory_lookup_query(self, query: str) -> bool:
        """判断用户是否在问历史对话本身。"""
        text = str(query or "")
        return any(marker in text for marker in (
            "之前", "刚才", "上面", "前面", "历史", "聊过", "说过", "说了什么", "我刚问",
            "上一句", "上一条", "还记得",
        ))

    def _has_query_terms(self, query: str) -> bool:
        """判断当前问题是否有可用于过滤上下文的有效关键词。"""
        return any(
            len(term) >= 2 and term not in RETRIEVAL_STOPWORDS
            for term in self._extract_retrieval_terms(query)
        )

    def _format_knowledge_answer(self, items: list, role_name: str, query: str = "") -> str:
        """把重排后的多条知识整合成更自然的回复，避免只摘一句资料。"""
        contents = [str(item.get("content", "") or "").strip() for item in items if item.get("content")]
        query = str(query or "")
        relevant_contents = [content for content in contents if self._is_relevant_to_query(query, content)]
        if self._has_query_terms(query):
            contents = relevant_contents
        merged = "\n".join(contents)

        if role_name == "医生":
            if "头疼" in query or "头痛" in query:
                return (
                    "你现在头疼的话，可以先试试休息一下、多喝点水，也可以量一下体温和血压。\n\n"
                    "如果最近熬夜、压力大或者用眼比较多，也可能会引起头痛。轻微的话，一般休息后会慢慢缓解。\n\n"
                    "但如果头痛特别剧烈、持续加重，或者伴随发烧、呕吐、视力模糊、肢体麻木无力、意识异常等情况，建议尽快去医院检查。"
                )
            if "贫血" in query:
                return (
                    "贫血要注意两件事：一是别只补，先尽量弄清楚为什么贫血；二是看症状有没有影响到日常活动。\n\n"
                    "日常可以先把饮食做扎实一点，比如适当吃瘦肉、蛋类、豆制品、深绿色蔬菜这类食物，搭配新鲜水果帮助铁吸收。茶和咖啡尽量别紧挨着正餐喝，因为可能影响铁吸收。\n\n"
                    "如果你已经有明显乏力、头晕、心慌、气短，或者月经过多、黑便、便血，建议尽快去医院查血常规、铁蛋白等指标，别自己盲目长期补铁。"
                )
            if "高血压" in query or "血压" in query:
                return (
                    "高血压平时最重要的是把血压稳住，别只盯着某一次数值。\n\n"
                    "日常可以先从这几件事做起：饮食尽量清淡一点，少盐；保持规律运动，比如快走、慢跑、骑车、游泳这类中等强度活动；平时定期测血压，最好固定时间测，记录下来更方便判断趋势。\n\n"
                    "另外，烟酒和情绪波动也会影响血压，所以戒烟限酒、少熬夜、保持情绪稳定也挺关键。\n\n"
                    "如果血压经常很高，或者已经在吃降压药，不建议自己随便停药、换药、加量。要是血压达到很高水平，或者伴随胸痛、头痛明显、肢体无力、说话不清、呼吸困难这些情况，就别硬扛，建议尽快就医。"
                )
        if role_name == "律师":
            if any(term in query for term in ("离婚", "婚姻", "夫妻", "抚养", "财产分割")):
                return self._friendly_from_contents(
                    contents,
                    "离婚这件事先不用急着把所有问题一次定死，通常要先分清是协议离婚还是诉讼离婚。",
                    "你可以先整理结婚证、身份证、户口本、子女情况、共同财产和债务线索；如果能协商，重点放在离婚意愿、子女抚养、财产分割这些内容上。",
                    "如果对方不同意、存在家暴、转移财产或孩子抚养争议，建议带材料咨询当地律师，具体路径会和证据、当地办理规则有关。",
                )
            if any(term in query for term in ("儿童", "童工", "未成年", "未成年人", "雇佣", "用工")):
                return self._friendly_from_contents(
                    contents,
                    "雇佣儿童这类问题要先看年龄和具体用工形式，不能只看双方是否愿意。",
                    "通常需要重点确认对方是否满16周岁、工作内容是否合法合规、是否涉及企业或个人经营性用工，并保留招聘、沟通、工资支付等材料。",
                    "如果已经发生纠纷，建议尽快咨询当地劳动监察部门或律师，别用口头说法替代证据判断。",
                )
            if any(term in query for term in ("合同", "欠款", "付款", "赔偿", "违约", "借钱", "租房")):
                return self._friendly_from_contents(
                    contents,
                    "这个事情可以先别急着下结论，法律问题通常要看事实和证据。",
                    "你可以先把合同、聊天记录、付款凭证、对方承诺这些材料整理好，再看责任怎么划分会更稳。",
                    "如果金额比较大或对方态度很强硬，建议带着材料咨询当地律师，别只凭口头描述做决定。",
                )
            return self._friendly_from_contents(
                contents,
                "这个问题先抓当前事项本身来看，法律判断通常要结合具体事实和证据。",
                "你可以先整理和这个事项直接相关的材料、时间线、对方身份和关键沟通记录，再判断下一步走协商、投诉还是诉讼。",
                "如果后果比较重或对方态度强硬，建议带材料咨询当地律师，具体结果会受证据和当地规则影响。",
            )
        if role_name == "心理医生":
            return self._friendly_from_contents(
                contents,
                "听起来这件事对你还是有影响的，可以先不用急着把自己调整好。",
                "先试着把当下最明显的感受说出来，再做一点简单的放松，比如慢慢呼吸、走动一下，或者把想法写下来。",
                "如果这种状态持续很久，或者已经影响睡眠、工作和生活，找线下专业人士聊聊会更稳妥。",
            )
        if role_name == "教师":
            return self._friendly_from_contents(
                contents,
                "这个问题可以拆开看，先抓住最核心的概念会轻松很多。",
                "你可以先用一个简单例子理解它，再回到题目里看条件怎么变化。",
                "要不要你把具体题目或卡住的那一步发我，我可以带你一步一步过。",
            )
        if role_name == "科学家":
            return self._friendly_from_contents(
                contents,
                "这个现象可以先从背后的机制来理解，不用一上来记结论。",
                "简单说，先看它受哪些因素影响，再看这些因素之间怎么互相作用，这样结论就比较清楚了。",
                "如果你愿意，我也可以用一个生活里的例子把它讲得更直观。",
            )
        if role_name == "股票分析师":
            return self._friendly_from_contents(
                contents,
                "这个问题不能只看涨跌，最好把趋势、基本面和风险放在一起看。",
                "比较关键的是看公司本身有没有支撑、板块资金是否还在、当前位置的波动风险大不大。",
                "我只能帮你做分析视角，不构成买卖建议，市场有风险，还是要结合你的仓位和风险承受能力来决定。",
            )
        if role_name == "英语学习助手":
            return self._friendly_from_contents(
                contents,
                "这个表达可以先抓住一个重点：英语更看重自然语序和使用场景。",
                "你可以先记一个最常用的说法，再理解为什么这样搭配，比死背规则更好用。",
                "你也可以发一句你想表达的中文或英文，我帮你改成更自然的版本。",
            )
        if role_name == "金融理财师":
            return self._friendly_from_contents(
                contents,
                "理财这件事先别急着追收益，先把安全垫和现金流理顺会更稳。",
                "一般可以先留好应急资金，再按风险承受能力去分配现金类、固收类和权益类资产。",
                "以上是一般性建议，不构成具体投资建议，具体比例还得看你的收入、支出、目标和能承受多大波动。",
            )

        return self._friendly_from_contents(
            contents,
            "这个可以简单聊，不用搞得太复杂。",
            "我从资料里挑最有用的部分看，重点是：" + self._extract_readable_points(contents),
            "你可以再补充一点具体情况，我就能接着往下帮你细化。",
        )

    def _friendly_from_contents(self, contents: list, opening: str, middle: str, ending: str) -> str:
        """按朋友式短段落组织答案，并混入可读的知识重点。"""
        points = self._extract_readable_points(contents)
        if points:
            return f"{opening}\n\n{middle}\n\n这里比较关键的是：{points}\n\n{ending}"
        return f"{opening}\n\n{middle}\n\n{ending}"

    def _extract_readable_points(self, contents: list) -> str:
        """从可能很长的 PDF 片段中挑出短、可读的重点。"""
        candidates = []
        for content in contents:
            text = str(content or "").replace("\n", " ")
            for part in text.replace("；", "。").replace(":", "：").split("。"):
                part = part.strip(" ：;，,")
                if 8 <= len(part) <= 80 and not any(noise in part for noise in ("图", "表", "第", "Chinese", "http")):
                    if "：" in part:
                        part = part.split("：", 1)[1].strip() or part
                    candidates.append(part)
        seen = []
        for item in candidates:
            if item and item not in seen:
                seen.append(item)
            if len(seen) >= 2:
                break
        return "；".join(seen)

    def _format_answer(self, content: str, role_name: str, query: str = "") -> str:
        """格式化答案"""
        # 提取关键句子
        content = str(content or "").strip()
        return self._format_knowledge_answer([{"content": content}], role_name, query)

    def _rerank_key(self, item: dict, terms: list, query: str, source_priority: dict) -> tuple:
        """轻量重排：相关性优先，同时奖励可读、可执行、来自 Milvus 的片段。"""
        content = str(item.get("content", "") or "")
        source = item.get("retrieval_source", "")
        exact_hits = sum(1 for term in terms if term and term in content)
        action_hits = sum(1 for term in ("建议", "注意", "避免", "监测", "运动", "饮食", "风险", "就医", "证据", "练习", "应急", "配置") if term in content)
        concise_bonus = 1.0 if 20 <= len(content) <= 260 else 0.0
        noisy_penalty = 1.0 if any(noise in content for noise in ("Chinese", "图", "表", "mmHg ＝", "第40 卷")) else 0.0
        return (
            float(item.get("relevance_score", item.get("score", 0)) or 0),
            exact_hits,
            action_hits,
            concise_bonus - noisy_penalty,
            source_priority.get(source, 0),
        )

    def _rerank_context(self, context: list, query: str) -> list:
        """对已经检索出的上下文再次排序，让重要、可读、可执行的内容排前面。"""
        terms = self._extract_retrieval_terms(query)
        source_priority = {"milvus_keyword": 3, "milvus": 2, "bm25": 1, "keyword": 1}
        return sorted(context or [], key=lambda item: self._rerank_key(item, terms, query, source_priority), reverse=True)

    def _extract_keywords(self, query: str) -> list:
        """提取关键词"""
        return self._extract_retrieval_terms(query)[:5]

    def _get_fallback_answer(self, query: str, role_name: str) -> str:
        """兜底回答"""
        answers = {
            "医生": "你可以先观察一下症状，注意休息和补水。如果不舒服持续加重，或者出现明显异常，建议尽快去医院看看。",
            "律师": "这个问题要结合具体事实和证据来看。你可以先整理和当前事项直接相关的材料、时间线、对方身份和关键沟通记录，再咨询当地律师会更准确。",
            "教师": "这个问题可以一步一步来。你先告诉我学生的年级和目前卡在哪里，我再帮你拆成更好理解的方法。",
        }
        return answers.get(role_name, f"我在，你可以把情况再说具体一点，我会按{role_name}的角度帮你分析。")

    def _extract_role_name(self, role_template: str) -> str:
        """提取角色名"""
        role_names = ["英语学习助手", "股票分析师", "金融理财师", "心理医生", "虚拟朋友", "科学家", "医生", "律师", "教师"]
        for name in role_names:
            if name in role_template:
                return name
        return "助手"

    def _tokenize(self, text: str) -> list:
        """分词"""
        if jieba:
            return list(jieba.cut_for_search(text))
        return text.split()

    def get_role_ids_for_file(self, file_path: str) -> list:
        """根据知识文件名判断它属于哪些角色，供 KnowledgeService 导入文件时使用。"""
        filename = os.path.basename(file_path or "")
        matched = []
        for role_id, keywords in ROLE_FILE_RULES.items():
            if any(keyword and keyword in filename for keyword in keywords):
                matched.append(role_id)
        return matched

    def _normalize_role_id(self, role_id):
        """把请求和缓存里的角色 ID 统一成 int，避免 '1' 与 1 匹配失败。"""
        if role_id is None:
            return None
        try:
            return int(role_id)
        except (TypeError, ValueError):
            return None

    def _normalize_role_ids(self, role_ids: list = None) -> list:
        """把知识片段上的 role_ids 统一成整数列表。"""
        normalized = []
        for role_id in role_ids or []:
            parsed = self._normalize_role_id(role_id)
            if parsed is not None:
                normalized.append(parsed)
        return sorted(set(normalized))

    def _extract_retrieval_terms(self, text: str) -> list:
        """抽取中文检索词，兼容 jieba、领域词和无空格中文短句。"""
        normalized = self._normalize_query_text(str(text or "").lower())
        terms = []

        for key, values in QUERY_EXPANSIONS.items():
            if key in normalized:
                terms.append(key)
                terms.extend(values)

        for term in DOMAIN_RETRIEVAL_TERMS:
            if term.lower() in normalized:
                terms.append(term.lower())

        for token in self._tokenize(normalized):
            token = str(token).strip().lower()
            if not token or token in RETRIEVAL_STOPWORDS:
                continue
            if len(token) > 1 or token in IMPORTANT_SINGLE_CHAR_TERMS:
                terms.append(token)

        compact = "".join(ch for ch in normalized if not ch.isspace())
        if len(compact) >= 2:
            for size in (4, 3, 2):
                for index in range(0, max(len(compact) - size + 1, 0)):
                    gram = compact[index:index + size]
                    if gram and gram not in RETRIEVAL_STOPWORDS:
                        terms.append(gram)

        seen = set()
        unique_terms = []
        for term in terms:
            if term and term not in seen:
                seen.add(term)
                unique_terms.append(term)
        return unique_terms

    def _keyword_score(self, query: str, text: str) -> float:
        """关键词评分"""
        return self._keyword_score_from_terms(self._extract_retrieval_terms(query), text)

    def _keyword_score_from_terms(self, terms: list, text: str) -> float:
        """根据抽取出的检索词给知识片段打分。"""
        text_lower = str(text or "").lower()
        score = 0.0
        for term in terms:
            if term in text_lower:
                score += max(1.0, min(len(term), 8) / 2)
        return score

    def _expand_query(self, query: str) -> str:
        """查询扩展"""
        query = str(query or "")
        additions = []
        for key, values in QUERY_EXPANSIONS.items():
            if key in query:
                additions.extend(values)
        if not additions:
            return query
        return f"{query} {' '.join(dict.fromkeys(additions))}"

    def _is_vague_medical_query(self, query: str) -> bool:
        """判断是否模糊医疗查询"""
        normalized = query
        has_vague = any(t in normalized for t in VAGUE_MEDICAL_TERMS)
        has_specific = any(t in normalized for t in SPECIFIC_MEDICAL_TERMS)
        return has_vague and not has_specific

    def _normalize_query_text(self, query: str) -> str:
        """标准化查询"""
        text = query
        for role in ROLE_VOCATIVES:
            text = text.replace(role, "")
        return text

    def rag_pipeline(self, query: str, role_template: str, role_id: int = None, **kwargs) -> str:
        """RAG主流程 - 目标总耗时 < 3秒"""
        # 检索（<500ms）
        context = self.search_knowledge(query, top_k=2, role_id=role_id)

        # 生成（<2500ms）
        response = self.generate_response(query, context, role_template)

        return response

# 全局单例保持轻量导入，模型加载交给应用启动阶段的 init_rag()。
rag_system = RAGSystem(load_models=False)

def init_rag():
    """初始化"""
    from app.core.vectorstore import init_milvus, vector_store
    init_milvus()
    if _enabled(settings.RAG_LOAD_MODELS):
        if rag_system.embedding_model is None or rag_system.rerank_model is None:
            rag_system.init_models()
    if not rag_system.knowledge_cache:
        rag_system.init_knowledge_base()
    if vector_store.client and rag_system.knowledge_items and vector_store.count() == 0:
        inserted = vector_store.insert_many(rag_system.knowledge_items, rag_system.get_embedding)
        print(f"已将本地知识库缓存回灌到 Milvus: {inserted} 条")
    elif vector_store.client and rag_system.knowledge_items:
        inserted = vector_store.insert_missing(rag_system.knowledge_items, rag_system.get_embedding)
        if inserted:
            print(f"已将缺失知识同步到 Milvus: {inserted} 条")
