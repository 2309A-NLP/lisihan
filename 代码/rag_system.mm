<map version="1.0.1">
  <node TEXT="多角色RAG智能聊天系统">
    <node TEXT="项目定位">
      <node TEXT="基于RAG的智能问答系统"/>
      <node TEXT="多角色聊天"/>
      <node TEXT="多轮对话"/>
      <node TEXT="知识库增强"/>
      <node TEXT="SSE流式响应"/>
    </node>

    <node TEXT="用户端功能">
      <node TEXT="登录注册">
        <node TEXT="用户注册"/>
        <node TEXT="用户登录"/>
        <node TEXT="JWT鉴权"/>
      </node>

      <node TEXT="角色系统">
        <node TEXT="医生"/>
        <node TEXT="律师"/>
        <node TEXT="心理医生"/>
        <node TEXT="教师"/>
        <node TEXT="科学家"/>
        <node TEXT="股票分析师"/>
        <node TEXT="英语学习助手"/>
        <node TEXT="虚拟朋友"/>
        <node TEXT="金融理财师"/>
        <node TEXT="自定义角色"/>
      </node>

      <node TEXT="聊天交互">
        <node TEXT="普通聊天"/>
        <node TEXT="SSE流式聊天"/>
        <node TEXT="多轮上下文"/>
        <node TEXT="会话列表"/>
        <node TEXT="历史消息"/>
      </node>

      <node TEXT="知识库管理">
        <node TEXT="添加文本"/>
        <node TEXT="上传文件"/>
        <node TEXT="PDF解析"/>
        <node TEXT="DOCX解析"/>
        <node TEXT="TXT解析"/>
      </node>
    </node>

    <node TEXT="前端 Frontend">
      <node TEXT="index.html"/>
      <node TEXT="login.html"/>
      <node TEXT="admin.html"/>
      <node TEXT="download.html"/>
      <node TEXT="assets">
        <node TEXT="style.css"/>
        <node TEXT="script.js"/>
      </node>
    </node>

    <node TEXT="后端 FastAPI">
      <node TEXT="main.py">
        <node TEXT="应用入口"/>
        <node TEXT="路由注册"/>
        <node TEXT="CORS配置"/>
      </node>

      <node TEXT="app/api">
        <node TEXT="chat.py"/>
        <node TEXT="user.py"/>
        <node TEXT="role.py"/>
        <node TEXT="knowledge.py"/>
        <node TEXT="admin.py"/>
      </node>

      <node TEXT="app/services">
        <node TEXT="user_service.py"/>
        <node TEXT="role_service.py"/>
        <node TEXT="knowledge_service.py"/>
      </node>
    </node>

    <node TEXT="RAG核心能力">
      <node TEXT="BM25检索"/>
      <node TEXT="向量检索"/>
      <node TEXT="Milvus检索"/>
      <node TEXT="Prompt构建"/>
      <node TEXT="大模型调用"/>
      <node TEXT="长期记忆"/>
    </node>

    <node TEXT="数据与存储">
      <node TEXT="MySQL"/>
      <node TEXT="Redis"/>
      <node TEXT="Milvus"/>
      <node TEXT="本地文件"/>
    </node>

    <node TEXT="系统运行流程">
      <node TEXT="启动流程"/>
      <node TEXT="聊天流程"/>
      <node TEXT="知识入库流程"/>
    </node>

    <node TEXT="测试与压测">
      <node TEXT="test_api.py"/>
      <node TEXT="test_rag.py"/>
      <node TEXT="stress_test.py"/>
      <node TEXT="JMeter压测"/>
    </node>

    <node TEXT="重点设计">
      <node TEXT="角色隔离"/>
      <node TEXT="多轮对话"/>
      <node TEXT="流式体验"/>
      <node TEXT="高可靠性"/>
      <node TEXT="可扩展性"/>
    </node>

  </node>
</map>
