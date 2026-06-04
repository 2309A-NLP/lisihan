"""
角色、模板、知识库数据模型模块
================================
这个模块定义了三个核心配置表的数据库结构：

1. roles（角色表）：定义AI角色的基本信息
2. templates（模板表）：存储角色的提示词模板
3. knowledge_bases（知识库表）：管理知识库的元信息

表关系：
templates (模板) 1 ── N roles (角色) N ── 1 knowledge_bases (知识库)

设计思路：
- 角色 = 模板 + 知识库
- 模板定义角色的"性格"和"说话方式"
- 知识库提供角色的"专业知识"
"""

# ========== SQLAlchemy 导入 ==========
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
# Column: 定义数据库列
# Integer: 整数类型
# String: 字符串类型（有长度限制）
# Text: 文本类型（无长度限制）
# DateTime: 日期时间类型
# ForeignKey: 外键约束，用于表关联

from sqlalchemy.sql import func
# func.now(): 获取当前时间的SQL函数

# ========== 导入基础模型类 ==========
from app.models.database import Base


# Base: SQLAlchemy 的声明性基类
class Role(Base):
    """
    角色表模型

    功能：
    - 定义AI角色的基本属性
    - 关联到模板（决定角色行为）
    - 关联到知识库（决定角色专业知识）

    表名：roles

    使用场景：
    - 创建新角色时插入记录
    - 聊天时根据role_id获取角色信息
    - 前端角色列表展示

    角色示例：
    - id=1, name="医生", template_id=1, knowledge_base_id=1
    - id=2, name="律师", template_id=2, knowledge_base_id=2
    - id=3, name="心理医生", template_id=3, knowledge_base_id=3

    与模板的关系：
    - 一个角色必须关联一个模板（模板定义角色性格）
    - 一个模板可以被多个角色使用

    与知识库的关系：
    - 一个角色必须关联一个知识库（提供专业知识）
    - 一个知识库可以被多个角色使用
    """

    # 表名定义
    __tablename__ = "roles"
    # 在MySQL中创建名为 'roles' 的表

    # ========== 字段定义 ==========

    # 主键ID
    # primary_key=True: 主键
    # index=True: 创建索引
    id = Column(Integer, primary_key=True, index=True)

    # 角色名称
    # String(50): 最大50个字符
    # unique=True: 名称必须唯一
    # index=True: 创建索引，加速按名称查询
    # nullable=False: 不能为空
    name = Column(String(50), unique=True, index=True, nullable=False)

    # 角色描述
    # Text: 文本类型，无长度限制
    # nullable=True: 可以为空（可选）
    description = Column(Text, nullable=True)

    # 模板ID（外键）
    # ForeignKey("templates.id"): 关联到templates表的id字段
    # nullable=False: 不能为空（每个角色必须有模板）
    template_id = Column(Integer, ForeignKey("templates.id"), nullable=False)

    # 知识库ID（外键）
    # ForeignKey("knowledge_bases.id"): 关联到knowledge_bases表的id字段
    # nullable=False: 不能为空（每个角色必须有知识库）
    knowledge_base_id = Column(Integer, ForeignKey("knowledge_bases.id"), nullable=False)

    # 创建时间
    # DateTime(timezone=True): 带时区的日期时间
    # server_default=func.now(): 数据库默认值为当前时间
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 更新时间
    # server_default=func.now(): 创建时默认为当前时间
    # onupdate=func.now(): 更新时自动更新为当前时间
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# Function: Map the prompt template database table.
class Template(Base):
    """
    模板表模型（提示词模板）

    功能：
    - 存储角色的系统提示词
    - 定义角色的性格、说话风格、行为规范
    - 控制AI如何理解和回应用户

    表名：templates

    什么是模板？
    模板是告诉AI"你是什么角色、该怎么说话"的系统指令。
    例如医生角色的模板：
    "你是一位专业的医生，回答要严谨、专业，提供健康建议时要加上免责声明..."

    使用场景：
    - 聊天时加载角色的模板
    - 构建提示词时注入模板内容
    - 管理不同角色的行为规范

    模板示例：
    - 医生模板：强调专业、谨慎、免责声明
    - 律师模板：强调法律依据、证据分析
    - 心理医生模板：强调共情、倾听、专业支持

    与角色的关系：
    - 一个模板可以被多个角色使用
    - 一个角色只能使用一个模板
    """

    # 表名定义
    __tablename__ = "templates"
    # 在MySQL中创建名为 'templates' 的表

    # ========== 字段定义 ==========

    # 主键ID
    id = Column(Integer, primary_key=True, index=True)

    # 模板名称
    # String(50): 最大50个字符
    # unique=True: 名称必须唯一
    # index=True: 创建索引
    # nullable=False: 不能为空
    name = Column(String(50), unique=True, index=True, nullable=False)

    # 模板内容（系统提示词）
    # Text: 文本类型，无长度限制
    # nullable=False: 不能为空
    # 内容示例：
    # "你是一位经验丰富的律师，毕业于知名法学院...
    #  回答法律问题时，要引用相关法律条文...
    #  保持专业、客观的态度..."
    content = Column(Text, nullable=False)

    # 创建时间
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 更新时间
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# Function: Map the knowledge_bases database table.
class KnowledgeBase(Base):
    """
    知识库表模型

    功能：
    - 管理知识库的元信息
    - 记录哪个角色使用哪个知识库
    - 知识库的实际内容存储在Milvus中

    表名：knowledge_bases

    注意：
    - 这个表只存储知识库的元数据（名称、描述）
    - 实际的知识内容存储在Milvus的chatbot_knowledge集合中
    - 通过knowledge_base_id关联Milvus中的数据

    使用场景：
    - 管理不同角色的知识源
    - 为角色分配知识库
    - 知识库的增删改查

    知识库示例：
    - id=1, name="医疗知识库", description="包含疾病诊断、治疗方案等"
    - id=2, name="法律知识库", description="包含民法典、宪法等法律条文"
    - id=3, name="教育知识库", description="包含课程标准、教学方法等"

    与角色的关系：
    - 一个知识库可以被多个角色使用
    - 一个角色只能使用一个知识库

    与Milvus的关系：
    - 知识库中的文件被处理后存入Milvus
    - Milvus中的每条数据都标记了knowledge_base_id
    """

    # 表名定义
    __tablename__ = "knowledge_bases"
    # 在MySQL中创建名为 'knowledge_bases' 的表

    # ========== 字段定义 ==========

    # 主键ID
    id = Column(Integer, primary_key=True, index=True)

    # 知识库名称
    # String(100): 最大100个字符
    # unique=True: 名称必须唯一
    # index=True: 创建索引
    # nullable=False: 不能为空
    name = Column(String(100), unique=True, index=True, nullable=False)

    # 知识库描述
    # Text: 文本类型，无长度限制
    # nullable=True: 可以为空（可选）
    description = Column(Text, nullable=True)

    # 创建时间
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 更新时间
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
