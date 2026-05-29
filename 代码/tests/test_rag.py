import pytest
# pytest: Python测试框架
# 提供断言、测试发现、fixture等功能

from app.core.rag import rag_system
# 导入RAG系统实例
# rag_system是整个聊天机器人的核心，负责检索和生成

from app.core.vectorstore import vector_store


# 导入向量存储实例
# vector_store负责向量数据的存储和检索


# 测试向量化功能
# Function: Test the embedding behavior.
def test_embedding():
    """
    测试文本向量化功能

    验证点：
    - 向量化函数能正常工作
    - 返回的是列表类型
    - 向量不为空

    向量化是将文本转换为数字向量的过程
    这是RAG系统的基础
    """
    text = "测试文本"

    # 调用RAG系统的向量化方法
    # get_embedding 会将文本转换为768维向量
    embedding = rag_system.get_embedding(text)

    # 断言：返回的是列表
    assert isinstance(embedding, list)

    # 断言：列表不为空（至少有一个元素）
    assert len(embedding) > 0


# 测试知识库搜索
# Function: Test the search knowledge behavior.
def test_search_knowledge():
    """
    测试知识库搜索功能

    流程：
    1. 先添加一条测试数据到向量数据库
    2. 搜索相关内容
    3. 验证搜索结果

    验证点：
    - 搜索返回列表
    - 搜索结果不为空
    - 每个结果包含content和distance字段
    """

    # ========== 步骤1: 准备测试数据 ==========
    test_content = "测试知识库内容"

    # 将测试文本向量化
    embedding = rag_system.get_embedding(test_content)

    # 将向量和文本存入Milvus
    # insert方法会存储到chatbot_knowledge集合
    vector_store.insert(test_content, embedding)

    # ========== 步骤2: 执行搜索 ==========
    # 搜索与"测试"相关的知识
    # search_knowledge会返回最相关的前5条结果
    results = rag_system.search_knowledge("测试")

    # ========== 步骤3: 验证结果 ==========
    # 断言：返回的是列表
    assert isinstance(results, list)

    # 断言：至少有一条结果
    assert len(results) > 0

    # 断言：第一条结果包含content字段（文本内容）
    assert "content" in results[0]

    # 断言：第一条结果包含distance字段（相似度距离）
    # distance越小表示越相似
    assert "distance" in results[0]


# 测试提示词构建
# Function: Test the build prompt behavior.
def test_build_prompt():
    """
    测试提示词构建功能

    验证RAG系统能否正确构建发送给大模型的提示词

    提示词组成：
    - 角色模板（定义角色行为）
    - 用户问题
    - 检索到的上下文
    - 回答策略
    - 历史对话等
    """

    # ========== 准备测试数据 ==========
    query = "测试问题"  # 用户问题
    context = [{"content": "测试上下文"}]  # 检索到的相关知识
    role_template = "测试角色模板"  # 角色提示词模板

    # ========== 构建提示词 ==========
    # build_prompt 会将各个部分组合成完整的提示词
    prompt = rag_system.build_prompt(query, context, role_template)

    # ========== 验证结果 ==========
    # 断言：返回的是字符串
    assert isinstance(prompt, str)

    # 断言：提示词包含用户问题
    assert query in prompt

    # 断言：提示词包含检索到的上下文
    assert "测试上下文" in prompt

    # 断言：提示词包含角色模板
    assert role_template in prompt


# 测试RAG完整流程
# Function: Test the rag pipeline behavior.
def test_rag_pipeline():
    """
    测试RAG完整流程（最重要的测试）

    RAG流程：
    1. 接收用户问题
    2. 检索相关知识库
    3. 构建提示词
    4. 调用大模型生成回复
    5. 返回最终答案

    这是集成测试，验证整个系统能否协同工作
    """

    # ========== 准备测试数据 ==========
    query = "测试问题"  # 模拟用户问题
    role_template = "你是一个测试角色，请回答用户的问题。"  # 角色模板

    # ========== 执行完整RAG流程 ==========
    # rag_pipeline 是RAG系统的主入口方法
    # 它会自动完成检索、生成等所有步骤
    response = rag_system.rag_pipeline(query, role_template)

    # ========== 验证结果 ==========
    # 断言：返回的是字符串
    assert isinstance(response, str)

    # 断言：返回的字符串不为空
    assert len(response) > 0


# Function: Test unrelated chat memory is not allowed to steer the current answer.
def test_build_prompt_filters_unrelated_memory():
    query = "贫血应该注意什么"
    context = [
        {
            "content": "贫血注意事项：注意查明原因，饮食可增加瘦肉、蛋类、深绿色蔬菜。",
            "source_file": "medical.txt",
        },
        {
            "content": "历史上下文(user): 我头疼怎么办",
            "source_file": "chat_context",
        },
    ]

    prompt = rag_system._build_prompt(query, context, "医生")

    assert "贫血应该注意什么" in prompt
    assert "贫血注意事项" in prompt
    assert "我头疼怎么办" not in prompt
    assert "当前问题是唯一主线" in prompt


# Function: Test doctor direct answer follows the current anemia query, not old headache context.
def test_doctor_anemia_answer_does_not_fall_back_to_headache():
    answer = rag_system._format_knowledge_answer(
        [{"content": "头疼/头痛处理建议：先休息、补充水分。"}],
        "医生",
        "贫血应该注意什么",
    )

    assert "贫血" in answer
    assert "头疼" not in answer
    assert "头痛" not in answer


# Function: Test lawyer fallback is specific to divorce and does not reuse contract wording.
def test_lawyer_divorce_answer_does_not_reuse_contract_template():
    answer = rag_system._format_knowledge_answer(
        [{"content": "合同纠纷：整理合同、聊天记录、付款凭证。"}],
        "律师",
        "我要离婚",
    )

    assert "离婚" in answer
    assert "付款凭证" not in answer
