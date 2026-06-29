"""
Dify Workflow 配置脚本（修正版）
修正内容：
1. 记账本端口 8080 → 8081
2. 所有后端参数名统一为 query（兼容 message/question）
3. 文生图新增 /generate_json base64 接口 + Code 节点
"""
import requests, json, uuid, sys

BASE = "http://localhost:5001"
LOGIN = {"email": "admin@example.com", "password": "password123"}

# ---------- 登录 ----------
r = requests.post(f"{BASE}/console/api/login", json=LOGIN)
if r.status_code != 200:
    print(f"❌ 登录失败: {r.text[:200]}")
    print("请检查 Dify 管理员账号密码，然后手动替换 setup_workflow.py 的 LOGIN 信息")
    sys.exit(1)

token = r.json()["data"]["access_token"]
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# ---------- 获取工作区 ----------
r = requests.get(f"{BASE}/console/api/workspaces/current", headers=headers)
ws = r.json()
tenant_id = ws.get("id") or ws.get("tenant_id", "")

# ---------- 工具函数 ----------
def nid(prefix="n"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

start_id = "1782098306880"  # 保留原开始节点
llm_id = nid("llm")
code_img_id = nid("code")    # 文生图 Code 节点
http_ids = [nid(f"http{i}") for i in range(5)]
end_id = nid("end")

# ---------- 构建工作流图 ----------
graph = {
    "nodes": [
        # ===== 开始节点 =====
        {
            "id": start_id,
            "type": "custom",
            "data": {
                "type": "start",
                "title": "开始",
                "variables": [
                    {"variable": "query", "label": "用户输入", "type": "text-input", "required": True},
                    {"variable": "files", "label": "上传文件", "type": "file-list", "required": False}
                ]
            },
            "position": {"x": 80, "y": 200},
            "sourcePosition": "right",
            "targetPosition": "left"
        },
        # ===== LLM 意图识别 =====
        {
            "id": llm_id,
            "type": "custom",
            "data": {
                "type": "llm",
                "title": "意图识别与路由",
                "model": {
                    "provider": "langgenius/siliconflow/siliconflow",
                    "name": "deepseek-ai/DeepSeek-V3",
                    "mode": "chat",
                    "completion_params": {
                        "temperature": 0.1,
                        "max_tokens": 512
                    }
                },
                "prompt_template": [
                    {
                        "role": "system",
                        "text": """你是一个智能路由助手。分析用户输入，只返回分类编号，不要其他内容。

分类：
1 - 记账/收支相关（记账、查账、账单、收支、花了、收入）
2 - 日程/提醒相关（日程、提醒、会议、安排、预约）
3 - 文生图相关（生成图片、人像、照片、角度、扩图）
4 - 基金数据查询（基金、股票、债券、行情、收益）
5 - 招股书查询（招股书、说明书、PDF内容）

用户输入: {{#1782098306880.query#}}

注意：
- 如果用户提到了"帮我生成"、"照片"、"人像"等图片相关词汇，返回3
- 只返回数字1-5中的一个。"""
                    }
                ]
            },
            "position": {"x": 400, "y": 200},
            "sourcePosition": "right",
            "targetPosition": "left"
        },
        # ===== 记账本 HTTP（端口 8081 ✅） =====
        {
            "id": http_ids[0],
            "type": "custom",
            "data": {
                "type": "http-request",
                "title": "记账本服务",
                "method": "post",
                "url": "http://host.docker.internal:8081/api/chat",
                "authorization": {"type": "no-auth"},
                "body": {
                    "type": "json",
                    "data": json.dumps({"query": "{{#1782098306880.query#}}"})
                },
                "timeout": {
                    "connect": 10, "read": 60, "write": 20,
                    "max_connect_timeout": 10, "max_read_timeout": 60, "max_write_timeout": 20
                },
                "ssl_verify": True
            },
            "position": {"x": 750, "y": 50},
            "sourcePosition": "right",
            "targetPosition": "left"
        },
        # ===== 日程提醒 HTTP =====
        {
            "id": http_ids[1],
            "type": "custom",
            "data": {
                "type": "http-request",
                "title": "日程提醒服务",
                "method": "post",
                "url": "http://host.docker.internal:5000/chat",
                "authorization": {"type": "no-auth"},
                "body": {
                    "type": "json",
                    "data": json.dumps({"query": "{{#1782098306880.query#}}"})
                },
                "timeout": {
                    "connect": 10, "read": 60, "write": 20,
                    "max_connect_timeout": 10, "max_read_timeout": 60, "max_write_timeout": 20
                },
                "ssl_verify": True
            },
            "position": {"x": 750, "y": 170},
            "sourcePosition": "right",
            "targetPosition": "left"
        },
        # ===== 文生图 Code 节点 =====
        {
            "id": code_img_id,
            "type": "custom",
            "data": {
                "type": "code",
                "title": "文生图-上传并生成",
                "variables": [
                    {"variable": "query", "value_selector": [start_id, "query"]},
                    {"variable": "files", "value_selector": [start_id, "files"]}
                ],
                "code_language": "python",
                "code": """import requests, json, base64, io

def main(query: str, files: list):
    # 获取上传的图片
    image_b64 = ""
    if files and len(files) > 0:
        # 读取上传的第一个文件
        file_info = files[0]
        if isinstance(file_info, dict) and 'url' in file_info:
            resp = requests.get(file_info['url'], timeout=30)
            image_b64 = base64.b64encode(resp.content).decode()
    
    if not image_b64:
        return {
            "result": "请先在聊天中上传一张面部照片，然后再发送'生成'指令。",
            "images": ""
        }
    
    # 调用文生图后端
    resp = requests.post(
        "http://host.docker.internal:7860/generate_json",
        json={"image_b64": image_b64, "do_outpaint": True},
        timeout=300
    )
    data = resp.json()
    
    if not data.get("success"):
        return {"result": f"生成失败: {data.get('error', '未知错误')}", "images": ""}
    
    # 构造 Markdown 显示图片
    imgs = data["images"]
    md = "## 生成结果\\n\\n"
    md += "| 左转 | 端正 | 右转 |\\n"
    md += "|------|------|------|\\n"
    md += f"| ![左转]({imgs.get('left', '')}) | ![端正]({imgs.get('front', '')}) | ![右转]({imgs.get('right', '')}) |\\n"
    
    if data.get("outpainted") and data["outpainted"].get("front"):
        md += "\\n### 扩图结果\\n\\n"
        md += "| 左转 | 端正 | 右转 |\\n"
        md += "|------|------|------|\\n"
        o = data["outpainted"]
        md += f"| ![左转扩图]({o.get('left', '')}) | ![端正扩图]({o.get('front', '')}) | ![右转扩图]({o.get('right', '')}) |\\n"
    
    return {"result": md, "images": json.dumps(imgs)}

""",
                "outputs": {
                    "result": {"type": "string", "children": []},
                    "images": {"type": "string", "children": []}
                }
            },
            "position": {"x": 750, "y": 290},
            "sourcePosition": "right",
            "targetPosition": "left"
        },
        # ===== 基金问答 HTTP =====
        {
            "id": http_ids[3],
            "type": "custom",
            "data": {
                "type": "http-request",
                "title": "基金问答服务",
                "method": "post",
                "url": "http://host.docker.internal:5002/ask",
                "authorization": {"type": "no-auth"},
                "body": {
                    "type": "json",
                    "data": json.dumps({"query": "{{#1782098306880.query#}}"})
                },
                "timeout": {
                    "connect": 10, "read": 60, "write": 20,
                    "max_connect_timeout": 10, "max_read_timeout": 60, "max_write_timeout": 20
                },
                "ssl_verify": True
            },
            "position": {"x": 750, "y": 410},
            "sourcePosition": "right",
            "targetPosition": "left"
        },
        # ===== 招股书问答 HTTP =====
        {
            "id": http_ids[4],
            "type": "custom",
            "data": {
                "type": "http-request",
                "title": "招股书问答服务",
                "method": "post",
                "url": "http://host.docker.internal:5003/ask",
                "authorization": {"type": "no-auth"},
                "body": {
                    "type": "json",
                    "data": json.dumps({"query": "{{#1782098306880.query#}}"})
                },
                "timeout": {
                    "connect": 10, "read": 120, "write": 20,
                    "max_connect_timeout": 10, "max_read_timeout": 120, "max_write_timeout": 20
                },
                "ssl_verify": True
            },
            "position": {"x": 750, "y": 530},
            "sourcePosition": "right",
            "targetPosition": "left"
        },
        # ===== 结束节点 =====
        {
            "id": end_id,
            "type": "custom",
            "data": {
                "type": "end",
                "title": "结束"
            },
            "position": {"x": 1050, "y": 280},
            "sourcePosition": "left",
            "targetPosition": "left"
        }
    ],
    "edges": [
        # 开始 → LLM
        {"id": f"e_{start_id}_{llm_id}", "source": start_id, "target": llm_id, "type": "custom"},
        # LLM → 各节点
        {"id": f"e_{llm_id}_{http_ids[0]}", "source": llm_id, "target": http_ids[0], "type": "custom", "sourceHandle": "1"},
        {"id": f"e_{llm_id}_{http_ids[1]}", "source": llm_id, "target": http_ids[1], "type": "custom", "sourceHandle": "2"},
        {"id": f"e_{llm_id}_{code_img_id}", "source": llm_id, "target": code_img_id, "type": "custom", "sourceHandle": "3"},
        {"id": f"e_{llm_id}_{http_ids[3]}", "source": llm_id, "target": http_ids[3], "type": "custom", "sourceHandle": "4"},
        {"id": f"e_{llm_id}_{http_ids[4]}", "source": llm_id, "target": http_ids[4], "type": "custom", "sourceHandle": "5"},
        # 各节点 → 结束
        {"id": f"e_{http_ids[0]}_{end_id}", "source": http_ids[0], "target": end_id, "type": "custom"},
        {"id": f"e_{http_ids[1]}_{end_id}", "source": http_ids[1], "target": end_id, "type": "custom"},
        {"id": f"e_{code_img_id}_{end_id}", "source": code_img_id, "target": end_id, "type": "custom"},
        {"id": f"e_{http_ids[3]}_{end_id}", "source": http_ids[3], "target": end_id, "type": "custom"},
        {"id": f"e_{http_ids[4]}_{end_id}", "source": http_ids[4], "target": end_id, "type": "custom"}
    ],
    "viewport": {"x": 0, "y": 0, "zoom": 1}
}

# ---------- 查询已有应用 ----------
print("=== 查询已有应用 ===")
r = requests.get(f"{BASE}/console/api/apps?page=1&limit=50", headers=headers)
apps = r.json().get("data", [])
app_id = None
for a in apps:
    if "智能体编排" in a["name"] or "07" in a["name"]:
        app_id = a["id"]
        print(f"  找到应用: {a['name']} (ID: {app_id})")
        break

if not app_id:
    print("❌ 未找到智能体编排应用，请先在 Dify UI 中创建")
    sys.exit(1)

# ---------- 同步工作流草稿 ----------
features = {
    "opening_statement": "你好！我是你的智能数字人助手，可以帮你完成以下任务：\n\n📋 **记账本** - 记录和查询家庭收支\n📅 **日程提醒** - 管理日程和设置提醒\n🎨 **文生图** - 基于照片生成多角度人像图（请先上传照片）\n💰 **基金问答** - 查询基金股票数据\n📄 **招股书问答** - 查询招股说明书内容\n\n请告诉我你需要什么帮助？",
    "suggested_questions": [
        "帮我记一笔账，今天买菜花了150元",
        "明天上午10点帮我设置一个会议提醒",
        "帮我查询一下最近的收支情况"
    ],
    "suggested_questions_after_answer": {"enabled": True},
    "file_upload": {"image": {"enabled": True, "number_limits": 5, "detail": "high"}},
    "speech_to_text": {"enabled": False},
    "text_to_speech": {"enabled": False},
    "retriever_resource": {"enabled": False},
    "sensitive_word_avoidance": {"enabled": False}
}

payload = {
    "graph": graph,
    "features": features,
    "environment_variables": [],
    "conversation_variables": []
}

print("\n=== 同步工作流草稿 ===")
r = requests.post(f"{BASE}/console/api/apps/{app_id}/workflows/draft", headers=headers, json=payload)
print(f"Status: {r.status_code}")
if r.status_code == 200:
    print("✅ 工作流草稿同步成功！")
else:
    print(f"❌ 失败: {r.text[:500]}")

# ---------- 发布 ----------
print("\n=== 发布工作流 ===")
r = requests.post(f"{BASE}/console/api/apps/{app_id}/workflows/publish", headers=headers)
print(f"Status: {r.status_code}")
if r.status_code == 200:
    print("✅ 工作流发布成功！")
else:
    print(f"❌ 失败: {r.text[:300]}")

print(f"\n📌 访问地址: http://localhost:3000/workflow/{app_id}")
print(f"\n📌 如果自动同步失败，请在 Dify UI 中手动修改：")
print(f"   1. 打开 http://localhost:3000/workflow/{app_id}")
print(f"   2. 记账本 HTTP 节点: URL 改为 http://host.docker.internal:8081/api/chat")
print(f"   3. 文生图: 新增 Code 节点（见上方代码）")
print(f"   4. 启动后发布")
