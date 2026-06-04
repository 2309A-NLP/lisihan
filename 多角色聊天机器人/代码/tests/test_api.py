import pytest
# pytest: Python 最流行的测试框架
# 提供断言、fixture、参数化等功能

from uuid import uuid4
# uuid4: 生成随机唯一标识符
# 用于生成不重复的测试用户名，避免冲突

from fastapi.testclient import TestClient
# TestClient: FastAPI 提供的测试客户端
# 可以模拟HTTP请求，不需要启动真实服务器

from main import app

# 导入主应用实例

# 创建测试客户端
client = TestClient(app)


# 这个客户端会直接调用FastAPI路由，不经过网络


# 测试用户注册
# Function: Test the register behavior.
def test_register():
    """
    测试用户注册接口

    场景：新用户注册账号
    验证点：
    - 返回200状态码
    - 响应包含user_id和username
    - username与请求一致
    """
    # 生成随机用户名（避免重复）
    # uuid4().hex[:8] 取UUID的前8个字符
    username = f"testuser_{uuid4().hex[:8]}"

    # 发送POST请求到注册接口
    response = client.post("/api/user/register", json={
        "username": username,
        "password": "testpassword123",  # 包含字母和数字，符合规则
        "email": f"{username}@example.com"
    })

    # 断言：期望返回200 OK
    assert response.status_code == 200

    # 解析JSON响应
    data = response.json()

    # 断言：响应中包含user_id字段
    assert "user_id" in data

    # 断言：响应中包含username字段
    assert "username" in data

    # 断言：返回的用户名与请求一致
    assert data["username"] == username


# 测试用户登录
# Function: Test the login behavior.
def test_login():
    """
    测试用户登录接口

    场景：已注册用户登录
    验证点：
    - 返回200状态码
    - 响应包含access_token和token_type
    - token_type为bearer
    """
    # 生成随机用户名
    username = f"testlogin_{uuid4().hex[:8]}"

    # 步骤1: 先注册用户（登录前需要先有用户）
    client.post("/api/user/register", json={
        "username": username,
        "password": "testpassword123",
        "email": f"{username}@example.com"
    })

    # 步骤2: 登录测试
    response = client.post("/api/user/login", json={
        "username": username,
        "password": "testpassword123"
    })

    # 断言：登录成功
    assert response.status_code == 200

    data = response.json()

    # 断言：返回JWT令牌
    assert "access_token" in data

    # 断言：返回令牌类型
    assert "token_type" in data

    # 断言：令牌类型是bearer
    assert data["token_type"] == "bearer"


# 测试获取角色列表
# Function: Test the get role list behavior.
def test_get_role_list():
    """
    测试获取角色列表接口

    场景：前端加载角色选择页面
    验证点：
    - 返回200状态码
    - 返回的是列表
    - 列表不为空
    - 每个角色包含id、name、description
    """
    # GET请求获取角色列表
    response = client.get("/api/role/list")

    # 断言：请求成功
    assert response.status_code == 200

    data = response.json()

    # 断言：返回的是列表
    assert isinstance(data, list)

    # 断言：列表不为空（至少有内置角色）
    assert len(data) > 0

    # 断言：第一个角色包含必要字段
    assert "id" in data[0]
    assert "name" in data[0]
    assert "description" in data[0]


# 测试聊天接口
# Function: Test the chat behavior.
def test_chat():
    """
    测试聊天接口

    场景：用户发送消息给AI
    验证点：
    - 返回200状态码
    - 响应包含response（AI回复）
    - 响应包含conversation_id（对话ID）
    """
    # 发送聊天请求
    # user_id=1: admin用户
    # role_id=1: 医生角色
    # message="你好": 用户消息
    response = client.post("/api/chat", json={
        "user_id": 1,
        "role_id": 1,
        "message": "你好"
    })

    # 断言：请求成功
    assert response.status_code == 200

    data = response.json()

    # 断言：返回AI回复内容
    assert "response" in data

    # 断言：返回对话ID
    assert "conversation_id" in data


# 测试添加知识库内容
# Function: Test the add knowledge behavior.
def test_add_knowledge():
    """
    测试添加知识库接口

    场景：通过API向知识库添加内容
    验证点：
    - 返回200状态码
    - 响应包含status字段
    - status值为success
    """
    response = client.post("/api/knowledge/add", json={
        "knowledge_base_id": 1,  # 知识库ID=1（医疗知识库）
        "content": "测试知识库内容"
    })

    assert response.status_code == 200

    data = response.json()

    assert "status" in data
    assert data["status"] == "success"


# 测试更新知识库内容
# Function: Test the update knowledge behavior.
def test_update_knowledge():
    """
    测试更新知识库接口

    场景：更新已有的知识库内容
    验证点：
    - 返回200状态码
    - 响应包含status字段
    - status值为success
    """
    response = client.post("/api/knowledge/update", json={
        "knowledge_base_id": 1,
        "content": "更新后的测试知识库内容"
    })

    assert response.status_code == 200

    data = response.json()

    assert "status" in data
    assert data["status"] == "success"


# 测试流式聊天接口
# Function: Test the streaming chat behavior.
def test_chat_stream(monkeypatch):
    """
    测试 SSE 流式聊天接口

    场景：前端需要边接收边显示回答
    验证点：
    - 返回200状态码
    - Content-Type 是 text/event-stream
    - 响应包含 token 和 done 事件
    """
    from app.api import chat as chat_api

    monkeypatch.setattr(
        chat_api,
        "_run_chat_sync",
        lambda request: chat_api.ChatResult(response="流式输出测试内容", conversation_id=123),
    )

    response = client.post("/api/chat/stream", json={
        "user_id": 1,
        "role_id": 1,
        "message": "测试流式输出"
    })

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: token" in response.text
    assert "event: done" in response.text
    assert '"conversation_id": 123' in response.text


# 测试聊天控制参数
# Function: Test chat context and embedding control parameters.
def test_chat_uses_context_and_embedding_controls(monkeypatch):
    """
    测试聊天接口的上下文和 embedding 控制参数

    场景：调用方需要控制历史长度、上下文预算、知识检索数量和长记忆检索策略
    验证点：
    - 参数能被 Pydantic 接收并传入聊天流程
    - 普通 JSON 接口仍保持原有 response/conversation_id 返回结构
    """
    from app.api import chat as chat_api

    captured = {}

    def fake_run_chat_sync(request):
        captured["history_limit"] = request.history_limit
        captured["context_budget"] = request.context_budget
        captured["knowledge_top_k"] = request.knowledge_top_k
        captured["memory_top_k"] = request.memory_top_k
        captured["use_embedding_memory"] = request.use_embedding_memory
        return chat_api.ChatResult(response="控制参数已生效", conversation_id=456)

    monkeypatch.setattr(chat_api, "_run_chat_sync", fake_run_chat_sync)

    response = client.post("/api/chat", json={
        "user_id": 1,
        "role_id": 1,
        "message": "测试控制参数",
        "history_limit": 3,
        "context_budget": 1200,
        "knowledge_top_k": 0,
        "memory_top_k": 0,
        "use_embedding_memory": False
    })

    assert response.status_code == 200
    assert response.json() == {"response": "控制参数已生效", "conversation_id": 456}
    assert captured == {
        "history_limit": 3,
        "context_budget": 1200,
        "knowledge_top_k": 0,
        "memory_top_k": 0,
        "use_embedding_memory": False,
    }


# 测试上下文参与 RAG 生成
# Function: Test context is passed into RAG generation.
def test_generate_response_uses_context_without_knowledge(monkeypatch):
    """
    测试关闭知识库检索时，上下文仍会进入 RAG 生成流程

    场景：调用方把 knowledge_top_k 设为0，只希望使用历史/记忆上下文
    验证点：
    - 不调用知识库检索
    - 裁剪后的上下文被转换为 generate_response 的 context
    """
    from app.api import chat as chat_api

    captured = {}

    def fail_search(*args, **kwargs):
        raise AssertionError("knowledge search should be skipped")

    def fake_generate_response(message, context, role_template, **kwargs):
        captured["message"] = message
        captured["context"] = context
        captured["role_template"] = role_template
        return "基于上下文生成"

    monkeypatch.setattr(chat_api.rag_system, "search_knowledge", fail_search)
    monkeypatch.setattr(chat_api.rag_system, "generate_response", fake_generate_response)

    request = chat_api.ChatRequest(
        user_id=1,
        role_id=1,
        message="他之前说了什么",
        knowledge_top_k=0,
    )
    context_bundle = {
        "history": [],
        "short_memory": [],
        "long_memory": [],
        "combined": [{"sender": "user", "content": "我喜欢清淡饮食"}],
    }

    result = chat_api._generate_response(request, "医生角色模板", context_bundle)

    assert result == "基于上下文生成"
    assert captured["message"] == "他之前说了什么"
    assert captured["role_template"] == "医生角色模板"
    assert captured["context"] == [{
        "content": "历史上下文(user): 我喜欢清淡饮食",
        "source_file": "chat_context",
        "score": 0,
    }]


# Function: Test memory rows are scoped to the active conversation.
def test_memory_context_is_filtered_by_conversation():
    from app.api import chat as chat_api

    rows = [
        {"sender": "user", "content": "old chat question", "timestamp": 30, "conversation_id": 11},
        {"sender": "role", "content": "current chat answer", "timestamp": 20, "conversation_id": 22},
        {"sender": "user", "content": "current chat question", "timestamp": 10, "conversation_id": 22},
        {"sender": "role", "content": "missing conversation id", "timestamp": 40},
    ]

    assert chat_api._memory_for_conversation(rows, 22) == [
        {"sender": "user", "content": "current chat question", "timestamp": 10, "conversation_id": 22},
        {"sender": "role", "content": "current chat answer", "timestamp": 20, "conversation_id": 22},
    ]
