"""
角色服务模块
================
这个模块提供角色的业务逻辑处理，包括：

核心功能：
1. 角色管理：获取所有角色、根据ID获取角色
2. 角色创建：创建新角色并关联模板和知识库
3. 模板获取：获取角色的提示词模板

角色数据来源：
- 内置角色：预定义的9个角色（医生、律师等）
- 自定义角色：用户通过API创建的角色（存储在数据库）

设计特点：
- 内置角色使用内存存储（SimpleNamespace）
- 自定义角色使用数据库存储
- 优先返回内置角色（保持ID一致性）
"""

from app.models.role import KnowledgeBase, Role, Template
# KnowledgeBase: 知识库模型
# Role: 角色模型
# Template: 模板模型

from app.models import Base, db, engine
# Base: SQLAlchemy的声明性基类
# db: 数据库会话对象（用于查询）
# engine: 数据库引擎（用于创建表）

import os
# os: 操作系统接口，用于读取模板文件

from types import SimpleNamespace
# SimpleNamespace: 创建简单的对象，类似字典但用点访问属性
# 用于将内置角色转换为类似数据库对象的格式

from sqlalchemy import func

# func: SQL函数，这里用于获取最大ID


# ========== 默认角色数据 ==========
# 预定义的9个内置角色，覆盖不同的专业领域
# ID固定为1-9，确保前后端一致
DEFAULT_ROLES = [
    # 角色1: 医生
    {"id": 1, "name": "医生", "description": "医疗健康顾问",
     "template": "你是医生，但说话要像一位懂医学、耐心的朋友。先检索知识库并重排重点，再用自然短段落给建议和风险提醒，不要像系统说明书。"},

    # 角色2: 律师
    {"id": 2, "name": "律师", "description": "法律顾问",
     "template": "你是律师，但说话要像一位靠谱、清楚的法律朋友。先检索知识库并重排重点，再讲大方向、证据材料和下一步，不要照搬法条。"},

    # 角色3: 心理医生
    {"id": 3, "name": "心理医生", "description": "心理咨询师",
     "template": "你是心理医生，但说话要像一个温和、有边界感的朋友。先检索知识库并重排重点，再共情、给轻量方法和必要安全提醒。"},

    # 角色4: 教师
    {"id": 4, "name": "教师", "description": "教育顾问",
     "template": "你是教师，但说话要像一位耐心的老师朋友。先检索知识库并重排重点，再用简单例子讲清楚，最后给轻量练习或追问。"},

    # 角色5: 科学家
    {"id": 5, "name": "科学家", "description": "科学顾问",
     "template": "你是科学家，但回答要像爱科普的朋友。先检索知识库并重排重点，再用直观结论、原理和生活化例子解释。"},

    # 角色6: 股票分析师
    {"id": 6, "name": "股票分析师", "description": "股票投资顾问",
     "template": "你是股票分析师，但说话要像冷静的投研朋友。先检索知识库并重排重点，再讲观察角度、依据和风险，不给买卖指令。"},

    # 角色7: 英语学习助手
    {"id": 7, "name": "英语学习助手", "description": "英语学习顾问",
     "template": "你是英语学习助手，但说话要像陪练朋友。先检索知识库并重排重点，再给自然表达、简短解释和小练习。"},

    # 角色8: 虚拟朋友
    {"id": 8, "name": "虚拟朋友", "description": "虚拟社交伙伴",
     "template": "你是虚拟朋友，不是客服。可以检索知识库和聊天记忆，但只选贴近当前对话的重点，用朋友式语气自然接话。"},

    # 角色9: 金融理财师
    {"id": 9, "name": "金融理财师", "description": "理财规划顾问",
     "template": "你是金融理财师，但说话要像稳健的理财朋友。先检索知识库并重排重点，再讲目标、现金流、配置和风险。"},
]


def _to_role(data):
    """
    将内置角色字典转换为 SimpleNamespace 对象

    为什么要转换？
    - 内置角色不存储在数据库，但需要和数据库角色统一接口
    - SimpleNamespace 提供和 SQLAlchemy 对象类似的属性访问方式
    - 这样 get_role_by_id 可以返回统一格式的对象

    Args:
        data: 内置角色字典

    Returns:
        SimpleNamespace: 角色对象，可通过 .id, .name 等访问
    """
    return SimpleNamespace(
        id=data["id"],  # 角色ID
        name=data["name"],  # 角色名称
        description=data["description"],  # 角色描述
        template_id=data["id"],  # 模板ID（使用相同ID）
        knowledge_base_id=data["id"],  # 知识库ID（使用相同ID）
    )


class RoleService:
    """
    角色服务类

    职责：
    - 统一管理内置角色和数据库角色
    - 提供角色的CRUD操作
    - 获取角色的提示词模板

    设计思路：
    - 内置角色ID固定为1-9，自定义角色ID从10开始
    - 内置角色优先级高于数据库角色（同名覆盖）
    - 模板可以从文件系统、数据库或内置数据获取
    """

    def get_all_roles(self):
        """
        获取所有角色（内置 + 自定义）

        功能：
        - 返回预定义的9个内置角色
        - 同时返回数据库中用户创建的自定义角色
        - 自动去重（如果数据库有相同ID的自定义角色，忽略）

        返回顺序：
        - 内置角色在前（ID 1-9）
        - 自定义角色在后（ID > 9）

        Returns:
            list: 角色对象列表
        """
        # 确保数据库表存在
        Base.metadata.create_all(bind=engine)

        # 转换内置角色为SimpleNamespace格式
        roles = [_to_role(role) for role in DEFAULT_ROLES]

        # 记录内置角色的ID集合，用于去重
        default_ids = {role.id for role in roles}

        # 从数据库查询所有自定义角色（按ID排序）
        db_roles = db.query(Role).order_by(Role.id.asc()).all()

        # 添加数据库角色（排除与内置角色ID重复的）
        # 这样如果用户创建了ID=1的角色，会被忽略（保护内置角色）
        roles.extend(role for role in db_roles if role.id not in default_ids)

        return roles

    def get_role_by_id(self, role_id: int):
        """
        根据ID获取角色

        查询优先级：
        1. 先查内置角色（ID 1-9）
        2. 再查数据库角色（ID > 9 或自定义角色）

        Args:
            role_id: 角色ID

        Returns:
            Role对象 或 SimpleNamespace对象 或 None
        """
        # ========== 步骤1: 检查是否为内置角色 ==========
        role = next((item for item in DEFAULT_ROLES if item["id"] == role_id), None)
        if role:
            return _to_role(role)  # 转换为SimpleNamespace

        # ========== 步骤2: 查数据库角色 ==========
        # 确保表存在
        Base.metadata.create_all(bind=engine)

        # 从数据库获取
        return db.get(Role, role_id)

    def create_role(
            self,
            name: str,
            description: str,
            template_id: int = None,
            knowledge_base_id: int = None,
            template_content: str = None,
            domain: str = None,
            personality: str = None,
    ):
        """
        创建新角色（支持自定义模板和知识库）

        创建流程：
        1. 检查角色名是否重复
        2. 处理模板（如果提供了template_content，创建新模板）
        3. 处理知识库（如果未提供，创建默认知识库）
        4. 生成新角色ID（大于所有现有角色的最大ID）
        5. 保存到数据库

        Args:
            name: 角色名称（必填）
            description: 角色描述
            template_id: 模板ID（可选，与template_content二选一）
            knowledge_base_id: 知识库ID（可选，不提供则自动创建）
            template_content: 模板内容（可选，自动创建模板）
            domain: 专业领域（可选，附加到描述）
            personality: 性格特征（可选，附加到描述）

        Returns:
            Role: 新创建的角色对象

        Raises:
            Exception: 角色名已存在或模板不存在
        """
        # ========== 步骤1: 确保表存在 ==========
        Base.metadata.create_all(bind=engine)

        # ========== 步骤2: 检查角色名是否重复 ==========
        if db.query(Role).filter_by(name=name).first():
            raise Exception("角色名称已存在")

        # ========== 步骤3: 处理模板 ==========
        # 如果提供了模板内容，创建新模板
        if template_content:
            template = Template(
                name=f"{name}角色模板",  # 模板名称
                content=template_content.strip(),  # 模板内容
            )
            db.add(template)
            db.commit()
            db.refresh(template)  # 刷新获取生成的ID
            template_id = template.id

        # 验证模板是否存在
        template = db.get(Template, template_id) if template_id else None
        if not template:
            raise Exception("提示词模板不存在，请传 template_id 或 template_content")

        # ========== 步骤4: 处理知识库 ==========
        # 如果提供了知识库ID，验证是否存在
        knowledge_base = db.get(KnowledgeBase, knowledge_base_id) if knowledge_base_id else None

        # 如果没有知识库，创建默认知识库
        if not knowledge_base:
            knowledge_base = KnowledgeBase(
                name=f"{name}知识库",
                description=f"{name}的专属知识库",
            )
            db.add(knowledge_base)
            db.commit()
            db.refresh(knowledge_base)  # 刷新获取生成的ID
            knowledge_base_id = knowledge_base.id

        # ========== 步骤5: 构建完整描述 ==========
        detail = description or ""
        extras = []

        # 添加专业领域信息
        if domain:
            extras.append(f"领域：{domain}")

        # 添加性格特征信息
        if personality:
            extras.append(f"人设/性格：{personality}")

        # 合并额外信息到描述
        if extras:
            detail = (detail + "\n" if detail else "") + "\n".join(extras)

        # ========== 步骤6: 生成新角色ID ==========
        # 获取数据库中的最大ID
        max_db_id = db.query(func.max(Role.id)).scalar() or 0

        # 获取内置角色的最大ID
        default_max_id = max(item["id"] for item in DEFAULT_ROLES)

        # 新ID = max(数据库最大ID, 内置最大ID) + 1
        # 确保自定义角色ID不会与内置角色冲突
        new_id = max(max_db_id, default_max_id) + 1

        # ========== 步骤7: 创建角色对象 ==========
        new_role = Role(
            id=new_id,  # 自动生成的新ID
            name=name,  # 角色名称
            description=detail,  # 完整描述
            template_id=template_id,  # 关联的模板ID
            knowledge_base_id=knowledge_base_id  # 关联的知识库ID
        )

        # ========== 步骤8: 保存到数据库 ==========
        db.add(new_role)
        db.commit()

        return new_role

    def get_role_template(self, role_id: int) -> str:
        """
        获取角色的提示词模板

        模板获取优先级：
        1. 数据库角色关联的模板
        2. 文件系统中的模板文件（app/templates/roles/{角色名}.txt）
        3. 内置角色的默认模板
        4. 通用兜底模板

        这样设计的好处：
        - 灵活：可以通过文件系统自定义模板
        - 可维护：模板与代码分离
        - 容错：有多个降级方案

        Args:
            role_id: 角色ID

        Returns:
            str: 提示词模板内容
        """
        # ========== 步骤1: 获取角色对象 ==========
        role = self.get_role_by_id(role_id)
        if not role:
            return None

        # ========== 步骤2: 检查是否为数据库角色 ==========
        # 如果是数据库角色（Role实例），从数据库获取模板
        if isinstance(role, Role):
            template = db.get(Template, role.template_id)
            if template and template.content:
                return template.content

        # ========== 步骤3: 尝试从文件系统读取模板 ==========
        # 模板文件路径：app/templates/roles/{角色名}.txt
        # 例如：app/templates/roles/医生.txt
        template_path = os.path.join(
            "app", "templates", "roles", f"{role.name.lower()}.txt"
        )

        try:
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            # 文件不存在或读取失败，继续降级
            pass

        # ========== 步骤4: 使用内置角色的默认模板 ==========
        default_role = next((item for item in DEFAULT_ROLES if item["id"] == role_id), None)
        if default_role:
            return default_role["template"]

        # ========== 步骤5: 兜底模板 ==========
        return f"你是{role.name}，请根据你的专业知识回答用户的问题。"
