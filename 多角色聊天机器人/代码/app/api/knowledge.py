"""
知识库管理API模块
==================
这个模块提供知识库的管理接口，包括：
- 文本知识的添加和更新
- 文件上传（PDF、DOCX、TXT）
- 知识库刷新和更新检查
- RAG系统性能评测
- 测试用例生成

知识库数据最终流向：
  上传文件 → 解析文档 → 文本分块 → 向量化 → 存入Milvus (chatbot_knowledge集合)
"""

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
# File: 用于接收上传的文件
# UploadFile: FastAPI的文件上传类型，提供异步读取功能

from app.services.knowledge_service import KnowledgeService  # 知识库业务逻辑服务
from app.services.role_service import RoleService  # 角色管理服务
from app.core.rag import rag_system  # RAG系统的核心实例
from app.core.vectorstore import vector_store  # Milvus 知识向量库
from app.core.memory import memory  # 记忆服务
from app.core.evaluation import RAGEvaluation  # RAG评测系统
from app.schemas.knowledge import (
    EvaluationRequest,
    EvaluationResponse,
    KnowledgeAddRequest,
    KnowledgeResponse,
    KnowledgeUpdateRequest,
    KnowledgeUpdateResponse,
)
from config.config import settings
import os  # 操作系统接口，用于路径操作
import shutil  # 高级文件操作（复制、移动）
import time  # 时间相关，用于生成唯一文件名

# 创建API路由器，所有路由都会以 /knowledge 为前缀
router = APIRouter()

# ========== 全局服务实例 ==========
# 保持文件哈希记录，用于检测文件是否变化
knowledge_service = KnowledgeService()
# 全局评测服务实例
rag_evaluation = RAGEvaluation()


@router.get("/milvus/status")
def milvus_status():
    """查看 Milvus 连接状态、知识库数量和长期记忆数量。"""
    try:
        vector_store.connect()
        connected = bool(vector_store.client)
        memory.long.connect()
        memory_client = memory.long.milvus_client
        memory_connected = bool(memory_client)
        return {
            "connected": connected,
            "uri": f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
            "collection": vector_store.get_collection() if connected else None,
            "count": vector_store.count() if connected else 0,
            "embedding": {
                "enabled": bool(settings.RAG_LOAD_MODELS),
                "loaded": rag_system.embedding_model is not None,
                "model_path": settings.EMBEDDING_MODEL,
            },
            "collections": {
                settings.MILVUS_COLLECTION: {
                    "exists": bool(vector_store.get_collection()) if connected else False,
                    "count": vector_store.count() if connected else 0,
                    "type": "knowledge",
                },
                settings.MILVUS_MEMORY_COLLECTION: {
                    "exists": (
                        memory_client.has_collection(settings.MILVUS_MEMORY_COLLECTION)
                        if memory_connected else False
                    ),
                    "count": memory.long.count() if memory_connected else 0,
                    "type": "long_term_memory",
                },
            },
        }
    except Exception as exc:
        return {
            "connected": False,
            "uri": f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
            "collection": None,
            "count": 0,
            "embedding": {
                "enabled": bool(settings.RAG_LOAD_MODELS),
                "loaded": rag_system.embedding_model is not None,
                "model_path": settings.EMBEDDING_MODEL,
            },
            "collections": {},
            "error": str(exc),
        }


@router.post("/milvus/rebuild")
def rebuild_milvus_knowledge():
    """使用当前 embedding 模型重新向量化并重建 Milvus 知识库集合。"""
    try:
        from app.core.rag import init_rag

        init_rag()
        if rag_system.embedding_model is None:
            raise HTTPException(
                status_code=500,
                detail="BGE embedding 模型未加载，请检查 RAG_LOAD_MODELS=true 和 EMBEDDING_MODEL 路径。",
            )
        inserted = vector_store.rebuild_from_items(rag_system.knowledge_items, rag_system.get_embedding)
        return {
            "success": inserted > 0,
            "collection": settings.MILVUS_COLLECTION,
            "embedding_model": settings.EMBEDDING_MODEL,
            "inserted": inserted,
            "count": vector_store.count(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"重建 Milvus 知识库失败: {exc}")


# ========== API 端点 1: 添加知识库内容 ==========
# HTTP POST /api/knowledge/add

@router.post("/add", response_model=KnowledgeResponse)
async def add_knowledge(request: KnowledgeAddRequest):
    """
    添加知识库内容（文本方式）

    功能说明：
    - 通过API直接添加文本知识到知识库
    - 知识会被分块、向量化后存入Milvus
    - 同时更新本地BM25索引

    使用场景：
    - 小批量知识的添加
    - 程序化地注入知识
    - 实时更新知识库

    参数：
        request: 包含 knowledge_base_id 和 content

    返回：
        KnowledgeResponse: 操作状态
    """
    try:
        # 调用知识服务添加知识
        # 这个方法会：
        # 1. 将文本分块（如果太长）
        # 2. 生成向量embedding
        # 3. 存储到Milvus
        # 4. 更新BM25索引
        knowledge_service.add_knowledge(
            knowledge_base_id=request.knowledge_base_id,
            content=request.content
        )
        return KnowledgeResponse(status="success")
    except Exception as e:
        # 捕获所有异常，返回400错误
        raise HTTPException(status_code=400, detail=f"添加知识库内容失败: {str(e)}")


# ========== API 端点 2: 更新知识库内容 ==========
# HTTP POST /api/knowledge/update
@router.post("/update", response_model=KnowledgeResponse)
async def update_knowledge(request: KnowledgeUpdateRequest):
    """
    更新知识库内容

    功能说明：
    - 更新已存在的知识条目
    - 会删除旧知识，添加新知识

    注意：
    - 需要知道 knowledge_base_id
    - 通常是先获取原内容，修改后调用此接口

    参数：
        request: 包含 knowledge_base_id 和新的 content

    返回：
        KnowledgeResponse: 操作状态
    """
    try:
        knowledge_service.update_knowledge(
            knowledge_base_id=request.knowledge_base_id,
            content=request.content
        )
        return KnowledgeResponse(status="success")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"更新知识库内容失败: {str(e)}")


# ========== API 端点 3: 上传知识库文件（核心接口） ==========
# HTTP POST /api/knowledge/upload
# 支持 form-data 格式：file=@文件路径&role_id=1&knowledge_base_id=2

@router.post("/upload", response_model=KnowledgeUpdateResponse)
async def upload_knowledge_file(
        file: UploadFile = File(...),  # 上传的文件，File(...) 表示必填
        role_id: int | None = Query(None),  # 角色ID（可选，如果指定则绑定到角色）
        knowledge_base_id: int | None = Query(None),  # 知识库ID（可选）
):
    """
    上传知识库文件（支持 PDF、DOCX、TXT）

    ⭐ 这是最常用的知识库管理接口 ⭐

    完整流程：
    1. 验证文件格式（只支持 .pdf, .docx, .txt）
    2. 保存文件到本地服务器（app/knowledge/data/）
    3. 如果指定了 role_id：
       - 验证角色是否存在
       - 获取或使用 knowledge_base_id
       - 调用 add_file_knowledge 解析并存入数据库
    4. 如果未指定 role_id：
       - 刷新整个知识库（处理所有文件）

    文件处理细节：
    - 文件名会加上 role_id 和时间戳，避免冲突
    - 支持的文件格式会被相应的解析器处理
    - 解析后的文本会被分块（chunk）
    - 每个块会被向量化并存入 Milvus

    参数：
        file: 上传的文件对象（PDF/DOCX/TXT）
        role_id: 关联的角色ID（可选，用于角色特定知识）
        knowledge_base_id: 知识库ID（可选，如果不提供会使用角色的默认知识库）

    返回：
        KnowledgeUpdateResponse: 包含文件信息和处理结果
    """
    try:
        # ========== 步骤1: 验证文件格式 ==========
        # 获取文件扩展名（如 .pdf, .docx, .txt）
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in {".pdf", ".docx", ".txt"}:
            raise HTTPException(status_code=400, detail="只支持 PDF、DOCX、TXT 文件")

        # ========== 步骤2: 构建保存路径 ==========
        # 获取当前文件的绝对路径
        current_file = os.path.abspath(__file__)
        # API目录：.../app/api/
        api_dir = os.path.dirname(current_file)
        # 应用根目录：.../app/
        app_dir = os.path.dirname(api_dir)
        # 知识库数据目录：.../app/knowledge/data/
        knowledge_dir = os.path.join(app_dir, 'knowledge', 'data')
        # 确保目录存在（如果不存在则创建）
        os.makedirs(knowledge_dir, exist_ok=True)

        # ========== 步骤3: 生成安全的文件名 ==========
        # 获取原始文件名（不含路径）
        safe_name = os.path.basename(file.filename)
        if role_id:
            # 如果指定了角色，添加角色ID和时间戳到文件名
            # 格式: role_{角色ID}_{时间戳}_{原文件名}
            # 例如: role_1_1734567890_民法典.pdf
            safe_name = f"role_{role_id}_{int(time.time())}_{safe_name}"
        # 完整的目标路径
        target_path = os.path.join(knowledge_dir, safe_name)

        # ========== 步骤4: 保存文件到本地 ==========
        # 将上传的文件内容复制到目标路径
        # shutil.copyfileobj 高效复制文件对象
        with open(target_path, "wb") as target:
            shutil.copyfileobj(file.file, target)

        # ========== 步骤5: 处理知识库更新 ==========
        if role_id:
            # 情况A: 指定了角色 - 添加到角色的知识库
            # 验证角色是否存在
            role = RoleService().get_role_by_id(role_id)
            if not role:
                raise HTTPException(status_code=404, detail="角色不存在")

            # 确定知识库ID：使用提供的或角色的默认知识库
            target_knowledge_base_id = knowledge_base_id or getattr(role, "knowledge_base_id", None)
            if not target_knowledge_base_id:
                raise HTTPException(status_code=400, detail="角色未绑定知识库")

            # 处理文件并添加到指定知识库
            # add_file_knowledge 会：
            # 1. 解析文件（PDF/DOCX/TXT）
            # 2. 分割成文本块
            # 3. 向量化
            # 4. 存储到Milvus
            # 5. 更新BM25索引
            result = knowledge_service.add_file_knowledge(
                knowledge_base_id=target_knowledge_base_id,
                role_id=role_id,
                file_path=target_path,
            )
        else:
            # 情况B: 未指定角色 - 刷新整个知识库
            # 处理 knowledge_dir 下的所有文件
            result = knowledge_service.refresh_knowledge(knowledge_dir, rag_system)

        # ========== 步骤6: 返回结果 ==========
        return KnowledgeUpdateResponse(
            success=True,
            data={"file": safe_name, "path": target_path, "refresh": result},
        )
    except HTTPException:
        raise  # 直接抛出HTTP异常，保持状态码
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"上传知识库文件失败: {str(e)}")


# ========== API 端点 4: 检查知识库更新 ==========
# HTTP POST /api/knowledge/check-updates
@router.post("/check-updates", response_model=KnowledgeUpdateResponse)
async def check_knowledge_updates():
    """
    检查知识库文件是否有更新

    功能说明：
    - 扫描知识库目录，检查文件是否有变化
    - 通过文件哈希值（MD5）检测文件是否被修改
    - 不实际加载文件，只检测变化

    工作原理：
    1. 遍历 knowledge/data 目录下的所有文件
    2. 计算每个文件的哈希值
    3. 与上次记录的哈希值比较
    4. 返回哪些文件有变化

    使用场景：
    - 定时检测知识库是否需要刷新
    - 在刷新前预先了解变更情况

    返回：
        KnowledgeUpdateResponse: 包含文件变更信息
    """
    try:
        # 构建知识库目录路径
        current_file = os.path.abspath(__file__)
        api_dir = os.path.dirname(current_file)
        app_dir = os.path.dirname(api_dir)
        knowledge_dir = os.path.join(app_dir, 'knowledge', 'data')

        # 检查是否有文件更新（不实际加载内容）
        result = knowledge_service.check_for_updates(knowledge_dir, rag_system)
        return KnowledgeUpdateResponse(success=True, data=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"检查知识库更新失败: {str(e)}")


# ========== API 端点 5: 刷新知识库 ==========
# HTTP POST /api/knowledge/refresh

# Function: Reparse all knowledge files and rebuild retrieval caches.
@router.post("/refresh", response_model=KnowledgeUpdateResponse)
async def refresh_knowledge():
    """
    刷新知识库（重新加载所有知识文件）

    功能说明：
    - 强制重新加载知识库目录下的所有文件
    - 清空现有知识，重新解析和向量化
    - 更新 Milvus 向量数据库和 BM25 索引

    执行流程：
    1. 扫描 knowledge/data 目录
    2. 对每个文件：
       - 根据扩展名选择解析器（PDF/DOCX/TXT）
       - 提取文本内容
       - 分割成文本块（chunk）
       - 生成向量embedding
       - 存入Milvus
    3. 构建 BM25 索引
    4. 更新文件哈希记录

    使用场景：
    - 手动更新知识库
    - 添加新文件后强制刷新
    - 修复知识库数据不一致

    注意：
    - 这是一个重量级操作，会消耗较多资源
    - 刷新期间可能影响RAG查询质量

    返回：
        KnowledgeUpdateResponse: 包含刷新统计信息
    """
    try:
        # 构建知识库目录路径
        current_file = os.path.abspath(__file__)
        api_dir = os.path.dirname(current_file)
        app_dir = os.path.dirname(api_dir)
        knowledge_dir = os.path.join(app_dir, 'knowledge', 'data')

        # 执行知识库刷新
        result = knowledge_service.refresh_knowledge(knowledge_dir, rag_system)
        return KnowledgeUpdateResponse(success=True, data=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"刷新知识库失败: {str(e)}")


# ========== API 端点 6: 评测RAG系统 ==========
# HTTP POST /api/knowledge/evaluate
@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_rag(request: EvaluationRequest):
    """
    评测 RAG 系统性能

    功能说明：
    - 使用测试用例评估RAG系统的准确性
    - 计算多个指标：准确率、召回率、F1分数、MRR等
    - 帮助优化知识库和检索参数

    评测流程：
    1. 对每个测试用例的问题进行RAG检索
    2. 获取检索到的相关文档
    3. 与期望答案/上下文比较
    4. 计算各种性能指标

    测试用例格式示例：
    [
        {
            "question": "民法典第1043条是什么？",
            "expected_context": "夫妻应当互相忠实",
            "expected_answer": "关于家庭关系的规定"
        },
        ...
    ]

    参数：
        request: 包含 test_cases 列表

    返回：
        EvaluationResponse: 包含评测指标和详细结果
    """
    try:
        test_cases = request.test_cases
        if not test_cases:
            raise HTTPException(status_code=400, detail="测试用例不能为空")

        # 执行评测
        # evaluate 方法会：
        # 1. 对每个测试用例执行RAG查询
        # 2. 计算检索准确率
        # 3. 计算生成答案的质量
        # 4. 汇总统计指标
        result = rag_evaluation.evaluate(test_cases)
        return EvaluationResponse(success=True, data=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"评测失败: {str(e)}")


# ========== API 端点 7: 生成测试用例 ==========
# HTTP POST /api/knowledge/generate-test-cases?count=10
@router.post("/generate-test-cases", response_model=KnowledgeUpdateResponse)
async def generate_test_cases(count: int = 10):
    """
    从知识库自动生成测试用例

    功能说明：
    - 基于现有知识库内容自动生成测试问题
    - 用于自动化评测RAG系统

    生成方法：
    1. 从知识库中随机抽取文本块
    2. 使用LLM根据文本块生成相关问题
    3. 将文本块作为期望答案
    4. 生成问题-答案对作为测试用例

    参数：
        count: 要生成的测试用例数量（默认10个）

    返回：
        KnowledgeUpdateResponse: 包含生成的测试用例列表

    使用场景：
    - 快速创建评测数据集
    - 持续集成中的自动化测试
    - 知识库质量验证
    """
    try:
        # 使用本地缓存的知识库内容生成测试用例
        # rag_system.knowledge_cache 包含已加载的所有知识文本
        test_cases = rag_evaluation.generate_test_cases(rag_system.knowledge_cache, count)
        return KnowledgeUpdateResponse(
            success=True,
            data={"test_cases": test_cases, "count": len(test_cases)}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"生成测试用例失败: {str(e)}")
