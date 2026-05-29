"""
角色管理API模块
================
这个模块提供角色的管理接口，包括：
- 获取角色列表
- 创建新角色

角色系统说明：
- 每个角色代表一个特定的AI人格（如律师、医生、老师等）
- 角色有独立的：名称、描述、性格、知识库、提示词模板
- 聊天时选择不同角色，会获得不同风格的回复
- 角色可以绑定专属知识库（如律师角色绑定法律知识库）
"""

from fastapi import APIRouter, HTTPException
# APIRouter: 用于创建API路由分组
# HTTPException: 用于返回标准HTTP错误响应

from pydantic import BaseModel
# BaseModel: Pydantic的数据模型基类，用于请求/响应数据验证

from app.services.role_service import RoleService
# RoleService: 角色业务逻辑服务，处理角色的数据库操作

from typing import List

# List: 类型注解，表示返回列表类型

# ========== 创建API路由器 ==========
# 所有角色相关的API都会以 /role 为前缀
# 完整路径示例: http://localhost:8000/api/role/list
router = APIRouter()


# ========== Pydantic 数据模型 ==========

class RoleCreateRequest(BaseModel):
    """
    创建角色的请求数据模型

    当创建新角色时，前端需要提供这些字段

    字段说明：
        name: 角色名称（必填），如"法律顾问"、"心理咨询师"
        description: 角色描述（可选），简要说明角色的功能
        template_id: 模板ID（可选），使用已有的提示词模板
        knowledge_base_id: 知识库ID（可选），绑定专用的知识库
        template_content: 模板内容（可选），直接提供提示词模板文本
        domain: 专业领域（可选），如"法律"、"医疗"、"教育"
        personality: 性格特征（可选），如"严谨"、"温和"、"幽默"

    注意：template_id 和 template_content 二选一即可
    """
    name: str  # 角色名称（必填）
    description: str = ""  # 角色描述，默认为空字符串
    template_id: int | None = None  # 使用预设模板的ID（可选）
    knowledge_base_id: int | None = None  # 关联的知识库ID（可选）
    template_content: str | None = None  # 自定义模板内容（可选）
    domain: str | None = None  # 角色专业领域（可选）
    personality: str | None = None  # 角色性格设定（可选）


class RoleResponse(BaseModel):
    """

    返回给前端的角色数据格式

    字段说明：
        id: 角色的唯一标识符（数据库自增ID）
        name: 角色名称
        description: 角色描述
        knowledge_base_id: 关联的知识库ID（可能为None）
    """
    id: int  # 角色ID
    name: str  # 角色名称
    description: str  # 角色描述
    knowledge_base_id: int | None = None  # 知识库ID（可选）


# ========== API 端点 1: 获取角色列表 ==========
# HTTP GET /api/role/list
@router.get("/list", response_model=List[RoleResponse])
# response_model=List[RoleResponse] 表示返回角色响应对象的列表
async def get_role_list():
    """
    获取所有角色的列表

    功能说明：
    - 查询数据库中所有已创建的角色
    - 返回角色的基本信息（ID、名称、描述、知识库ID）
    - 用于前端展示角色选择界面

    使用场景：
    - 用户打开聊天页面时，显示所有可用角色
    - 角色切换下拉菜单的数据源
    - 角色管理后台展示角色列表

    返回示例：
    [
        {
            "id": 1,
            "name": "法律顾问",
            "description": "专业法律咨询助手",
            "knowledge_base_id": 1
        },
        {
            "id": 2,
            "name": "心理咨询师",
            "description": "温暖的心理支持伙伴",
            "knowledge_base_id": null
        }
    ]

    可能的错误：
        - 500: 数据库查询失败或其他服务器错误
    """
    try:
        # 实例化角色服务
        # RoleService 封装了角色的所有数据库操作
        role_service = RoleService()

        # 获取所有角色
        # get_all_roles() 返回所有角色的ORM对象列表
        # 这些对象来自数据库的 roles 表
        roles = role_service.get_all_roles()

        # 将ORM对象转换为API响应格式
        # 列表推导式：对每个角色创建 RoleResponse 对象
        return [
            RoleResponse(
                id=role.id,  # 角色ID
                name=role.name,  # 角色名称
                description=role.description,  # 角色描述
                knowledge_base_id=getattr(role, "knowledge_base_id", None),  # 安全获取知识库ID
            )
            for role in roles  # 遍历所有角色
        ]
    except Exception as e:
        # 发生任何异常都返回500错误
        # 使用 raise HTTPException 直接返回错误响应
        raise HTTPException(status_code=500, detail=f"获取角色列表失败: {str(e)}")


# ========== API 端点 2: 创建新角色 ==========
# HTTP POST /api/role/create
@router.post("/create", response_model=RoleResponse)
# response_model=RoleResponse 表示返回新创建的角色信息
async def create_role(request: RoleCreateRequest):
    """
    创建新角色

    功能说明：
    - 创建一个新的AI角色
    - 可以配置角色的各种属性
    - 创建成功后返回角色信息

    创建角色的完整流程：
    1. 验证输入数据（Pydantic自动完成）
    2. 调用 RoleService 创建角色记录
    3. 在数据库中插入新角色
    4. 如果指定了模板内容，保存提示词模板
    5. 如果指定了知识库ID，建立关联
    6. 返回新角色的信息

    使用场景：
    - 管理员在后台创建新角色
    - 用户自定义角色（如果开放此功能）
    - 系统初始化时创建默认角色

    请求示例：
    POST /api/role/create
    Content-Type: application/json

    {
        "name": "法律顾问",
        "description": "专业的法律咨询助手，提供法律条文解读和建议",
        "domain": "法律",
        "personality": "严谨、专业、耐心",
        "template_content": "你是一位专业的法律顾问，擅长解答...",
        "knowledge_base_id": 1
    }

    参数：
        request: RoleCreateRequest 对象，包含角色配置

    返回：
        RoleResponse: 新创建的角色信息（包含自动生成的ID）

    可能的错误：
        - 400: 请求参数无效（如角色名称已存在）
        - 500: 数据库操作失败
    """
    try:
        # 实例化角色服务
        role_service = RoleService()

        # 调用服务层创建角色
        # create_role 方法会：
        # 1. 检查角色名是否重复
        # 2. 创建角色记录到数据库
        # 3. 如果提供了 template_content，保存到 role_templates 表
        # 4. 关联知识库（如果提供了 knowledge_base_id）
        # 5. 返回创建的角色对象
        role = role_service.create_role(
            name=request.name,  # 角色名称
            description=request.description,  # 角色描述
            template_id=request.template_id,  # 可选的模板ID
            knowledge_base_id=request.knowledge_base_id,  # 可选的知识库ID
            template_content=request.template_content,  # 可选的模板内容
            domain=request.domain,  # 可选的领域设定
            personality=request.personality,  # 可选的性格设定
        )

        # 将ORM对象转换为API响应格式
        # 注意：这里只返回部分字段，不返回敏感信息
        return RoleResponse(
            id=role.id,  # 新生成的角色ID
            name=role.name,  # 角色名称
            description=role.description,  # 角色描述
            knowledge_base_id=getattr(role, "knowledge_base_id", None),  # 知识库ID
        )
    except Exception as e:
        # 创建失败时返回400错误（客户端请求问题）
        # 可能的失败原因：
        # - 角色名称已存在
        # - 模板ID不存在
        # - 知识库ID不存在
        # - 数据验证失败
        raise HTTPException(status_code=400, detail=f"创建角色失败: {str(e)}")