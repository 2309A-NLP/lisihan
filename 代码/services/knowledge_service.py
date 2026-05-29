"""
知识库服务模块
================
这个模块负责知识库的管理和维护，包括：

核心功能：
1. 知识处理：解析文档、分块、向量化、存储
2. 缓存管理：避免重复处理，加速启动
3. 增量更新：检测文件变化，只处理变更
4. 角色绑定：将知识关联到特定角色

数据流向：
knowledge/data/*.docx → 解析 → 分块 → 向量化 → Milvus + BM25索引
                                    ↓
                              缓存到本地JSON
"""

import os
import time
import logging
from datetime import datetime
import json
import re
from app.models.role import KnowledgeBase, Role
from app.core.vectorstore import vector_store
from app.models import db
from app.utils import FileParser
import hashlib

# 缓存版本号（修改此值会强制重建所有缓存）
CACHE_VERSION = 5
logger = logging.getLogger(__name__)

def _knowledge_verbose() -> bool:
    return os.getenv("KNOWLEDGE_VERBOSE", "0").lower() in {"1", "true", "yes", "on"}

class KnowledgeService:
    """
    知识库服务类

    职责：
    - 管理知识文件的处理流程
    - 维护文件哈希值，检测变更
    - 处理知识的分块和向量化
    - 管理本地缓存（避免重复解析）

    使用场景：
    - 应用启动时加载知识库
    - 用户上传新知识文件
    - 手动刷新知识库
    - 增量更新检测
    """

    def __init__(self):
        """
        初始化知识库服务

        主要维护文件哈希字典，用于检测文件是否变更
        file_hashes 格式：
        {
            "/path/to/file1.docx": "md5哈希值",
            "/path/to/file2.pdf": "md5哈希值"
        }
        """
        # 存储文件哈希值，用于检测文件变化
        # 哈希值变化说明文件内容已修改
        self.file_hashes = {}

    def _store_segment(self, segment: str, role_ids: list, source_file: str, rag_system=None) -> bool:
        """把知识片段同时写入 Milvus 和本地 RAG 缓存。"""
        segment = (segment or "").strip()
        if not segment:
            return False

        if rag_system:
            embedding = rag_system.get_embedding(segment)
        else:
            from app.core.rag import rag_system as default_rag_system
            rag_system = default_rag_system
            embedding = rag_system.get_embedding(segment)

        if not embedding:
            raise RuntimeError(f"知识片段向量化失败: {source_file}")

        if not vector_store.insert(segment, embedding, role_ids=role_ids, source_file=source_file):
            raise RuntimeError(f"知识片段写入 Milvus 失败: {source_file}")

        if rag_system:
            if hasattr(rag_system, "add_knowledge_segment"):
                rag_system.add_knowledge_segment(
                    segment,
                    role_ids=role_ids,
                    source_file=source_file,
                )
            else:
                rag_system.knowledge_cache.append(segment)
        return True

    def add_knowledge(self, knowledge_base_id: int, content: str):
        """
        添加知识库内容（文本方式）

        功能：
        - 将纯文本知识添加到知识库
        - 自动向量化并存入Milvus
        - 更新本地RAG缓存
        - 重建BM25索引

        使用场景：
        - API调用添加知识
        - 小批量知识的快速添加

        Args:
            knowledge_base_id: 知识库ID
            content: 知识文本内容

        Returns:
            bool: 是否成功
        """
        from app.core.rag import rag_system

        # ========== 步骤1: 验证知识库存在 ==========
        try:
            knowledge_base = db.get(KnowledgeBase, knowledge_base_id)
        except Exception:
            knowledge_base = None

        # ========== 步骤2: 向量化内容 ==========
        # 调用RAG系统的向量化模型
        role_ids = self._role_ids_for_knowledge_base(knowledge_base_id)
        self._store_segment(content, role_ids, "接口新增知识", rag_system=rag_system)

        # ========== 步骤4: 更新知识库元数据 ==========
        if knowledge_base:
            knowledge_base.updated_at = datetime.utcnow()
            db.commit()

        rag_system.init_bm25_index()

        return True

    def add_file_knowledge(self, knowledge_base_id: int, role_id: int, file_path: str):
        """
        解析上传文件，并把内容绑定到指定角色的知识库

        这是处理上传文件的核心方法

        处理流程：
        1. 解析文件内容（支持PDF/DOCX/TXT）
        2. 将内容分块（每块约1000字符）
        3. 为每个分块生成向量
        4. 存入Milvus向量数据库
        5. 添加到本地RAG缓存
        6. 重建BM25索引

        Args:
            knowledge_base_id: 知识库ID
            role_id: 角色ID（用于绑定）
            file_path: 文件路径

        Returns:
            dict: 处理结果统计
        """
        from app.core.rag import rag_system

        # ========== 步骤1: 解析文件 ==========
        # FileParser 根据扩展名选择解析器
        content = FileParser.parse_file(file_path)

        # ========== 步骤2: 分块处理 ==========
        # 避免单个知识片段过长，影响检索效果
        segments = self._split_content(content, max_length=1000)

        # ========== 步骤3: 确定角色绑定 ==========
        # 如果有role_id，直接使用；否则从知识库推断
        role_ids = [role_id] if role_id else self._role_ids_for_knowledge_base(knowledge_base_id)

        # 获取文件名（用于追踪来源）
        source_file = os.path.basename(file_path)
        added = 0

        # ========== 步骤4: 处理每个分块 ==========
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            if self._store_segment(segment, role_ids, source_file, rag_system=rag_system):
                added += 1

        # ========== 步骤5: 重建BM25索引 ==========
        if hasattr(rag_system, "init_bm25_index"):
            rag_system.init_bm25_index()

        # ========== 步骤6: 更新知识库元数据 ==========
        try:
            knowledge_base = db.get(KnowledgeBase, knowledge_base_id)
        except Exception:
            knowledge_base = None

        if knowledge_base:
            knowledge_base.updated_at = datetime.utcnow()
            db.commit()

        return {"segments": added, "role_ids": role_ids, "source_file": source_file}

    def _role_ids_for_knowledge_base(self, knowledge_base_id: int) -> list:
        """
        根据知识库ID找到绑定的角色

        查询逻辑：
        1. 从数据库查询所有使用该知识库的角色
        2. 如果没有找到，使用knowledge_base_id作为角色ID（降级）

        Args:
            knowledge_base_id: 知识库ID

        Returns:
            list: 角色ID列表
        """
        try:
            # 查询所有关联该知识库的角色
            role_ids = [
                role.id
                for role in db.query(Role).filter(Role.knowledge_base_id == knowledge_base_id).all()
            ]
        except Exception:
            role_ids = []

        # 降级：如果没有找到角色，使用知识库ID作为角色ID
        return role_ids or [knowledge_base_id]

    def update_knowledge(self, knowledge_base_id: int, content: str):
        """
        更新知识库内容

        注意：当前实现是添加新知识，不是真正的更新
        实际应用中需要先删除旧内容再添加新内容

        Args:
            knowledge_base_id: 知识库ID
            content: 新的知识内容

        Returns:
            bool: 是否成功
        """
        from app.core.rag import rag_system

        # ========== 步骤1: 验证知识库存在 ==========
        try:
            knowledge_base = db.get(KnowledgeBase, knowledge_base_id)
        except Exception:
            knowledge_base = None

        # ========== 步骤2: 向量化 ==========
        role_ids = self._role_ids_for_knowledge_base(knowledge_base_id)
        self._store_segment(content, role_ids, "接口更新知识", rag_system=rag_system)

        # ========== 步骤4: 更新元数据 ==========
        if knowledge_base:
            knowledge_base.updated_at = datetime.utcnow()
            db.commit()

        rag_system.init_bm25_index()

        return True

    def process_knowledge_files(self, directory: str, rag_system=None, use_cache: bool = True):
        """
        处理知识库目录下的所有文件（核心方法）

        功能：
        - 扫描目录，处理所有支持的文件
        - 支持缓存（避免重复处理）
        - 自动识别文件所属角色

        处理流程：
        1. 检查缓存（如果启用）
        2. 遍历目录下的所有文件
        3. 对每个文件：
           - 识别角色ID
           - 解析文件内容
           - 分块
           - 添加到RAG缓存
        4. 保存新缓存

        Args:
            directory: 知识库文件目录
            rag_system: RAG系统实例
            use_cache: 是否使用缓存

        Returns:
            dict: 处理结果统计
        """
        try:
            # ========== 步骤1: 尝试从缓存加载 ==========
            if use_cache and self._load_cached_knowledge(directory, rag_system):
                # 从缓存加载成功，直接返回统计信息
                role_counts = self._role_counts(rag_system.knowledge_items if rag_system else [])
                skipped_files = getattr(rag_system, "_cached_skipped_files", [])
                result = {
                    "total": len(FileParser.list_files(directory)),
                    "processed": 0,  # 缓存加载，不需要重新处理
                    "failed": 0,
                    "from_cache": True,  # 标记来自缓存
                    "role_counts": role_counts,
                    "processed_files": [],
                    "failed_files": [],
                    "skipped": len(skipped_files),
                    "skipped_files": skipped_files,
                }
                print(
                    "已从知识库缓存加载，无需重复解析文件: "
                    f"total={result['total']}, role_counts={result['role_counts']}, skipped={result['skipped']}"
                )
                return result

            # ========== 步骤2: 获取所有文件 ==========
            files = FileParser.list_files(directory)
            processed_files = []
            failed_files = []
            skipped_files = []

            verbose = _knowledge_verbose()
            print(f"开始处理知识库文件，目录: {directory}")
            print(f"找到 {len(files)} 个文件")

            # ========== 步骤3: 处理每个文件 ==========
            for file_path in files:
                try:
                    if verbose:
                        print(f"正在处理文件: {os.path.basename(file_path)}")

                    # 3.1 识别角色ID
                    role_ids = []
                    if rag_system and hasattr(rag_system, "get_role_ids_for_file"):
                        role_ids = rag_system.get_role_ids_for_file(file_path)

                    # 如果没有匹配的角色，跳过
                    if not role_ids:
                        print(f"跳过未匹配角色的知识文件: {os.path.basename(file_path)}")
                        continue

                    # 3.2 解析文件内容
                    content = FileParser.parse_file(file_path)

                    # 3.3 分块处理
                    segments = self._split_content(content, max_length=1000)
                    if verbose:
                        print(f"文件分段数: {len(segments)}")

                    # 3.4 处理每个分块：必须写入 Milvus，再同步到本地 RAG 缓存
                    added_segments = 0
                    for i, segment in enumerate(segments):
                        if self._store_segment(
                            segment,
                            role_ids,
                            os.path.basename(file_path),
                            rag_system=rag_system,
                        ):
                            added_segments += 1

                    # 3.5 记录文件哈希
                    file_hash = self._get_file_hash(file_path)
                    self.file_hashes[file_path] = file_hash

                    processed_files.append(os.path.basename(file_path))
                    if verbose:
                        print(f"成功处理文件: {os.path.basename(file_path)}，角色: {role_ids}，写入分段数: {added_segments}")

                except Exception as e:
                    error_message = self._compact_error(str(e))
                    if self._is_skippable_ocr_error(error_message):
                        skipped_files.append({
                            "file": os.path.basename(file_path),
                            "reason": error_message,
                            "role_ids": role_ids,
                        })
                        if verbose:
                            print(f"跳过文件: {os.path.basename(file_path)} - {error_message}")
                        continue
                    failed_files.append({"file": os.path.basename(file_path), "error": error_message})
                    print(f"处理文件失败: {os.path.basename(file_path)} - {error_message}")

            # ========== 步骤4: 构建结果 ==========
            result = {
                "total": len(files),
                "processed": len(processed_files),
                "failed": len(failed_files),
                "processed_files": processed_files,
                "failed_files": failed_files,
                "skipped": len(skipped_files),
                "skipped_files": skipped_files,
            }

            if failed_files:
                print("以下文件未导入，可稍后修复后刷新知识库:")
                for item in failed_files:
                    print(f"  - {item['file']}: {item['error']}")
            if skipped_files:
                skipped_names = "、".join(item["file"] for item in skipped_files)
                print(
                    f"已跳过 {len(skipped_files)} 个扫描版PDF: {skipped_names}。"
                    "安装 Tesseract OCR + chi_sim，并设置 PDF_ENABLE_OCR=1 后可导入全文。"
                )

            print(f"知识库文件处理完成: {result}")

            # ========== 步骤5: 保存缓存 ==========
            if rag_system:
                print(f"本地缓存知识库段落数: {len(rag_system.knowledge_cache)}")
                self._save_knowledge_cache(directory, rag_system.knowledge_items, skipped_files=skipped_files)

            return result

        except Exception as e:
            print(f"处理知识库文件失败: {str(e)}")
            logger.exception("处理知识库文件失败: directory=%s, error=%s", directory, e)
            return {
                "total": 0,
                "processed": 0,
                "failed": 0,
                "processed_files": [],
                "failed_files": [],
                "skipped": 0,
                "skipped_files": []
            }

    def _cache_dir(self, directory: str) -> str:
        """
        获取缓存目录路径

        Args:
            directory: 知识库目录

        Returns:
            str: 缓存目录路径（knowledge/cache）
        """
        return os.path.join(os.path.dirname(directory), "cache")

    def _cache_path(self, directory: str) -> str:
        """
        获取缓存索引文件路径

        Args:
            directory: 知识库目录

        Returns:
            str: 缓存索引文件路径
        """
        return os.path.join(self._cache_dir(directory), "knowledge_index.json")

    def _role_cache_path(self, directory: str, role_id: int) -> str:
        """
        获取角色特定缓存文件路径

        Args:
            directory: 知识库目录
            role_id: 角色ID

        Returns:
            str: 角色缓存文件路径
        """
        return os.path.join(self._cache_dir(directory), "roles", f"role_{role_id}.json")

    def _build_file_state(self, directory: str) -> dict:
        """
        构建文件状态快照

        用于缓存验证，记录每个文件的状态

        Args:
            directory: 知识库目录

        Returns:
            dict: 文件状态字典
        """
        state = {}
        for file_path in FileParser.list_files(directory):
            state[os.path.basename(file_path)] = {
                "hash": self._get_file_hash(file_path),  # MD5哈希
                "size": os.path.getsize(file_path),  # 文件大小
                "mtime": os.path.getmtime(file_path),  # 修改时间
            }
        return state

    def _load_cached_knowledge(self, directory: str, rag_system=None) -> bool:
        """
        从缓存加载知识库

        缓存验证条件：
        1. 缓存文件存在
        2. 版本号匹配
        3. 文件列表一致
        4. 每个文件的哈希值一致

        Args:
            directory: 知识库目录
            rag_system: RAG系统实例

        Returns:
            bool: 是否成功从缓存加载
        """
        cache_path = self._cache_path(directory)

        # 检查是否可用
        if not rag_system or not os.path.exists(cache_path):
            return False

        try:
            # ========== 读取缓存 ==========
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            # ========== 验证版本 ==========
            if payload.get("version") != CACHE_VERSION:
                print("知识库缓存版本已过期，将重新构建")
                return False

            # ========== 验证文件列表 ==========
            current_files = self._build_file_state(directory)
            cached_files = payload.get("files", {})

            if current_files.keys() != cached_files.keys():
                print("知识库文件集合发生变化，将重新构建缓存")
                return False

            # ========== 验证每个文件的哈希 ==========
            for filename, current in current_files.items():
                cached = cached_files.get(filename, {})
                if current.get("hash") != cached.get("hash"):
                    print(f"知识库文件已更新，将重新构建缓存: {filename}")
                    return False

            # ========== 加载缓存的知识片段 ==========
            for item in payload.get("items", []):
                rag_system.add_knowledge_segment(
                    item.get("content", ""),
                    role_ids=item.get("role_ids", []),
                    source_file=item.get("source_file", "缓存知识")
                )
            rag_system._cached_skipped_files = payload.get("skipped_files", [])

            # ========== 恢复文件哈希记录 ==========
            self.file_hashes = {
                os.path.join(directory, filename): data.get("hash", "")
                for filename, data in cached_files.items()
            }

            return True

        except Exception as e:
            print(f"读取知识库缓存失败，将重新构建: {e}")
            return False

    def _save_knowledge_cache(self, directory: str, knowledge_items: list, skipped_files: list = None):
        """
        保存知识库到缓存

        功能：
        - 保存全局索引文件
        - 按角色分别保存缓存文件

        Args:
            directory: 知识库目录
            knowledge_items: 知识项列表
        """
        try:
            # ========== 创建缓存目录 ==========
            os.makedirs(self._cache_dir(directory), exist_ok=True)
            os.makedirs(os.path.join(self._cache_dir(directory), "roles"), exist_ok=True)

            # ========== 构建文件状态 ==========
            file_state = self._build_file_state(directory)

            # ========== 筛选来自文件的知识项 ==========
            file_items = [
                {
                    "content": item.get("content", ""),
                    "role_ids": item.get("role_ids", []),
                    "source_file": item.get("source_file", ""),
                }
                for item in knowledge_items
                if item.get("source_file") and item.get("source_file") in file_state
            ]

            # ========== 保存全局索引 ==========
            payload = {
                "version": CACHE_VERSION,
                "files": file_state,
                "items": file_items,
                "skipped_files": skipped_files or [],
                "role_counts": self._role_counts(file_items),
            }

            with open(self._cache_path(directory), "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            # ========== 按角色分别保存 ==========
            for role_id in range(1, 10):  # 角色ID 1-9
                role_items = [
                    item for item in file_items
                    if role_id in item.get("role_ids", [])
                ]
                role_payload = {
                    "version": CACHE_VERSION,
                    "role_id": role_id,
                    "count": len(role_items),
                    "items": role_items,
                }
                with open(self._role_cache_path(directory, role_id), "w", encoding="utf-8") as f:
                    json.dump(role_payload, f, ensure_ascii=False, indent=2)

            print(f"知识库缓存已保存: {self._cache_path(directory)}")
            print(f"按角色分类缓存已保存: {os.path.join(self._cache_dir(directory), 'roles')}")

        except Exception as e:
            print(f"保存知识库缓存失败: {e}")

    def _role_counts(self, knowledge_items: list) -> dict:
        """
        统计各角色的知识数量

        Args:
            knowledge_items: 知识项列表

        Returns:
            dict: 角色ID到数量的映射
        """
        # 初始化角色1-9的计数为0
        counts = {str(role_id): 0 for role_id in range(1, 10)}

        # 统计每个角色的知识数量
        for item in knowledge_items:
            for role_id in item.get("role_ids", []):
                counts[str(role_id)] = counts.get(str(role_id), 0) + 1

        return counts

    def _compact_error(self, message: str) -> str:
        """
        压缩长错误，避免启动日志被第三方库堆栈和OCR说明刷屏。
        """
        normalized = re.sub(r"\s+", " ", message or "").strip()
        if "扫描版PDF需要OCR" in normalized or "PDF未包含可提取的文本层" in normalized:
            return "扫描版PDF已跳过：安装 Tesseract OCR + chi_sim，并设置 PDF_ENABLE_OCR=1 后可导入全文"
        return normalized[:260]

    def _is_skippable_ocr_error(self, message: str) -> bool:
        return "扫描版PDF已跳过" in (message or "")

    def _split_content(self, content: str, max_length: int = 1000):
        """
        将内容分割成适当长度的段落

        分块策略：
        - 按段落分割（\n）
        - 保持段落完整性
        - 超过max_length时截断

        Args:
            content: 原始内容
            max_length: 最大段落长度

        Returns:
            list: 分段后的内容列表
        """
        segments = []
        current_segment = ""

        # 按段落分割
        for paragraph in content.split('\n'):
            # 如果当前段落加上新段落不超过限制
            if len(current_segment) + len(paragraph) + 1 <= max_length:
                current_segment += paragraph + '\n'
            else:
                # 保存当前段落，开始新段落
                if current_segment.strip():
                    segments.append(current_segment.strip())
                current_segment = paragraph + '\n'

        # 保存最后一段
        if current_segment.strip():
            segments.append(current_segment.strip())

        return segments

    def _get_file_hash(self, file_path: str) -> str:
        """
        计算文件的MD5哈希值

        用于检测文件是否被修改

        Args:
            file_path: 文件路径

        Returns:
            str: MD5哈希值（十六进制字符串）
        """
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except Exception as e:
            print(f"计算文件哈希值失败: {str(e)}")
            return ""

    def check_for_updates(self, directory: str, rag_system=None):
        """
        检查知识库文件是否有更新

        功能：
        - 扫描目录，识别新文件和修改过的文件
        - 只处理变化的文件（增量更新）

        Args:
            directory: 知识库文件目录
            rag_system: RAG系统实例

        Returns:
            dict: 更新结果
        """
        try:
            print(f"开始检查知识库文件更新，目录: {directory}")

            # ========== 步骤1: 扫描文件 ==========
            files = FileParser.list_files(directory)
            updated_files = []  # 修改过的文件
            new_files = []  # 新文件

            # ========== 步骤2: 识别变更 ==========
            for file_path in files:
                file_hash = self._get_file_hash(file_path)

                # 新文件
                if file_path not in self.file_hashes:
                    new_files.append(file_path)
                    self.file_hashes[file_path] = file_hash
                # 文件已修改（哈希变化）
                elif self.file_hashes[file_path] != file_hash:
                    updated_files.append(file_path)
                    self.file_hashes[file_path] = file_hash

            # ========== 步骤3: 处理变更 ==========
            if new_files or updated_files:
                print(f"发现 {len(new_files)} 个新文件，{len(updated_files)} 个更新文件")

                processed_files = []
                failed_files = []

                # 处理新文件和更新文件
                for file_path in new_files + updated_files:
                    try:
                        print(f"正在处理文件: {os.path.basename(file_path)}")

                        # 识别角色
                        role_ids = []
                        if rag_system and hasattr(rag_system, "get_role_ids_for_file"):
                            role_ids = rag_system.get_role_ids_for_file(file_path)

                        if not role_ids:
                            print(f"跳过未匹配角色的知识文件: {os.path.basename(file_path)}")
                            continue

                        # 解析和分块
                        content = FileParser.parse_file(file_path)
                        segments = self._split_content(content, max_length=1000)
                        print(f"文件分段数: {len(segments)}")

                        # 添加到 Milvus 和本地缓存
                        added_segments = 0
                        for i, segment in enumerate(segments):
                            if self._store_segment(
                                segment,
                                role_ids,
                                os.path.basename(file_path),
                                rag_system=rag_system,
                            ):
                                added_segments += 1

                        processed_files.append(os.path.basename(file_path))
                        print(f"成功处理文件: {os.path.basename(file_path)}，角色: {role_ids}，写入分段数: {added_segments}")

                    except Exception as e:
                        failed_files.append({"file": os.path.basename(file_path), "error": str(e)})
                        print(f"处理文件失败: {os.path.basename(file_path)} - {str(e)}")

                # ========== 步骤4: 重建BM25索引 ==========
                if rag_system:
                    rag_system.init_bm25_index()

                result = {
                    "new_files": [os.path.basename(f) for f in new_files],
                    "updated_files": [os.path.basename(f) for f in updated_files],
                    "processed": len(processed_files),
                    "failed": len(failed_files),
                    "processed_files": processed_files,
                    "failed_files": failed_files
                }
            else:
                print("没有发现文件更新")
                result = {
                    "new_files": [],
                    "updated_files": [],
                    "processed": 0,
                    "failed": 0,
                    "processed_files": [],
                    "failed_files": []
                }

            print(f"知识库文件更新检查完成: {result}")
            return result

        except Exception as e:
            print(f"检查知识库文件更新失败: {str(e)}")
            logger.exception("检查知识库文件更新失败: directory=%s, error=%s", directory, e)
            return {
                "new_files": [],
                "updated_files": [],
                "processed": 0,
                "failed": 0,
                "processed_files": [],
                "failed_files": []
            }

    def refresh_knowledge(self, directory: str, rag_system=None):
        """
        刷新知识库，重新处理所有文件（强制重建）

        功能：
        - 清空所有缓存
        - 重新处理所有文件
        - 重建BM25索引

        使用场景：
        - 手动触发知识库重建
        - 缓存损坏时的修复

        Args:
            directory: 知识库文件目录
            rag_system: RAG系统实例

        Returns:
            dict: 刷新结果
        """
        try:
            print(f"开始刷新知识库，目录: {directory}")

            # ========== 步骤1: 清空本地缓存 ==========
            if rag_system:
                rag_system.knowledge_cache = []
                if hasattr(rag_system, "knowledge_items"):
                    rag_system.knowledge_items = []
                print("已清空本地缓存")

            # ========== 步骤2: 清空文件哈希记录 ==========
            self.file_hashes = {}
            print("已清空文件哈希记录")

            # ========== 步骤3: 重新处理所有文件 ==========
            # use_cache=False 强制重新处理
            result = self.process_knowledge_files(directory, rag_system, use_cache=False)

            # ========== 步骤4: 重建BM25索引 ==========
            if rag_system:
                rag_system.init_bm25_index()
                print("已重新构建 BM25 索引")

            print(f"知识库刷新完成: {result}")
            return result

        except Exception as e:
            print(f"刷新知识库失败: {str(e)}")
            logger.exception("刷新知识库失败: directory=%s, error=%s", directory, e)
            return {
                "total": 0,
                "processed": 0,
                "failed": 0,
                "processed_files": [],
                "failed_files": []
            }
