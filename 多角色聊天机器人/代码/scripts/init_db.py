from app.models import Base, engine, db
# Base: SQLAlchemy 的声明性基类，用于创建表
# engine: 数据库引擎，负责连接数据库
# db: 数据库会话对象，用于执行查询

from app.models.user import User
# User: 用户模型

from app.models.role import Role, Template, KnowledgeBase
# Role: 角色模型
# Template: 模板模型
# KnowledgeBase: 知识库模型

from app.services.user_service import UserService


# UserService: 用户服务，用于创建用户


# Function: Create all database tables.
def init_database():
    """
    初始化数据库（主入口函数）

    功能：
    1. 创建所有数据库表
    2. 初始化基础数据（模板、知识库、角色、用户）
    """
    try:
        # ========== 步骤1: 创建所有表 ==========
        # Base.metadata.create_all 会根据模型定义创建表
        # 如果表已存在，不会重复创建
        Base.metadata.create_all(bind=engine)
        print("成功创建数据库表")

        # ========== 步骤2: 初始化基础数据 ==========
        init_base_data()
        print("成功初始化基础数据")

    except Exception as e:
        print(f"初始化数据库失败: {e}")


# Function: Initialize templates, knowledge bases, roles, and default users.
def init_base_data():
    """
    初始化所有基础数据

    包括：
    - 提示词模板
    - 知识库
    - 角色
    - 默认用户
    """
    init_templates()  # 初始化模板
    init_knowledge_bases()  # 初始化知识库
    init_roles()  # 初始化角色
    init_default_user()  # 创建默认管理员


# Function: Insert default prompt templates.
def init_templates():
    """
    初始化提示词模板

    模板定义了各个角色的性格、说话方式、专业领域
    """
    # 定义所有模板
    templates = [
        {"name": "医生", "content": "你是一名专业的医生，擅长解答医疗健康相关问题。"},
        {"name": "心理医生", "content": "你是一名专业的心理医生，擅长心理咨询和心理疏导。"},
        {"name": "律师", "content": "你是一名专业的律师，擅长解答法律相关问题。"},
        {"name": "股票分析师", "content": "你是一名专业的股票分析师，擅长分析股票市场和投资策略。"},
        {"name": "金融理财师", "content": "你是一名专业的金融理财师，擅长个人理财规划。"},
        {"name": "科学家", "content": "你是一名专业的科学家，擅长解答科学相关问题。"},
        {"name": "教师", "content": "你是一名专业的教师，擅长教育教学。"},
        {"name": "英语学习助手", "content": "你是一名专业的英语学习助手，擅长英语教学和辅导。"},
        {"name": "虚拟朋友", "content": "你是一个友好、善解人意的虚拟朋友，擅长倾听和交流。"}
    ]

    # 遍历并插入模板（如果不存在）
    for template_data in templates:
        # 检查模板是否已存在
        existing_template = db.query(Template).filter_by(name=template_data["name"]).first()

        if not existing_template:
            # 不存在则创建
            template = Template(
                name=template_data["name"],
                content=template_data["content"]
            )
            db.add(template)

    # 提交所有更改
    db.commit()


# Function: Insert default knowledge base records.
def init_knowledge_bases():
    """
    初始化知识库

    知识库用于存储角色的专业知识
    """
    knowledge_bases = [
        {"name": "医疗知识库", "description": "包含医疗健康相关知识"},
        {"name": "心理学知识库", "description": "包含心理学相关知识"},
        {"name": "法律知识库", "description": "包含法律法规相关知识"},
        {"name": "金融知识库", "description": "包含金融投资相关知识"},
        {"name": "科学知识库", "description": "包含科学技术相关知识"},
        {"name": "教育知识库", "description": "包含教育教学相关知识"},
        {"name": "英语知识库", "description": "包含英语学习相关知识"},
        {"name": "社交知识库", "description": "包含社交交流相关知识"}
    ]

    # 遍历并插入知识库（如果不存在）
    for kb_data in knowledge_bases:
        existing_kb = db.query(KnowledgeBase).filter_by(name=kb_data["name"]).first()
        if not existing_kb:
            knowledge_base = KnowledgeBase(
                name=kb_data["name"],
                description=kb_data["description"]
            )
            db.add(knowledge_base)

    db.commit()


# Function: Insert default role records.
def init_roles():
    """
    初始化角色

    角色 = 模板 + 知识库
    每个角色关联特定的模板和知识库
    """
    roles = [
        # 角色名, 描述, 使用的模板, 使用的知识库
        {"name": "医生", "description": "医疗健康顾问", "template_name": "医生", "kb_name": "医疗知识库"},
        {"name": "心理医生", "description": "心理咨询师", "template_name": "心理医生", "kb_name": "心理学知识库"},
        {"name": "律师", "description": "法律顾问", "template_name": "律师", "kb_name": "法律知识库"},
        {"name": "股票分析师", "description": "股票投资顾问", "template_name": "股票分析师", "kb_name": "金融知识库"},
        {"name": "金融理财师", "description": "理财规划顾问", "template_name": "金融理财师", "kb_name": "金融知识库"},
        {"name": "科学家", "description": "科学顾问", "template_name": "科学家", "kb_name": "科学知识库"},
        {"name": "教师", "description": "教育顾问", "template_name": "教师", "kb_name": "教育知识库"},
        {"name": "英语学习助手", "description": "英语学习顾问", "template_name": "英语学习助手",
         "kb_name": "英语知识库"},
        {"name": "虚拟朋友", "description": "虚拟社交伙伴", "template_name": "虚拟朋友", "kb_name": "社交知识库"}
    ]

    for role_data in roles:
        # 检查角色是否已存在
        existing_role = db.query(Role).filter_by(name=role_data["name"]).first()

        if not existing_role:
            # 获取关联的模板
            template = db.query(Template).filter_by(name=role_data["template_name"]).first()

            # 获取关联的知识库
            knowledge_base = db.query(KnowledgeBase).filter_by(name=role_data["kb_name"]).first()

            # 如果模板和知识库都存在，创建角色
            if template and knowledge_base:
                role = Role(
                    name=role_data["name"],
                    description=role_data["description"],
                    template_id=template.id,  # 关联模板ID
                    knowledge_base_id=knowledge_base.id  # 关联知识库ID
                )
                db.add(role)

    db.commit()


# Function: Insert a default user account.
def init_default_user():
    """
    创建默认管理员账号

    用户名: admin
    密码: admin123
    邮箱: admin@example.com
    """
    # 检查 admin 用户是否已存在
    existing_user = db.query(User).filter_by(username="admin").first()

    if not existing_user:
        # 使用用户服务创建新用户
        user_service = UserService()
        user_service.register(
            username="admin",
            password="admin123",
            email="admin@example.com"
        )
        print("默认用户创建成功: admin/admin123")


# 如果直接运行此脚本，执行初始化
if __name__ == "__main__":
    init_database()