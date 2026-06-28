"""
招股书问答智能体 - ReAct 模式
智能体自主决策：检索招股书、多步推理、给出答案
"""
import json, os, time, re, requests, numpy as np

API_URL = "https://api.siliconflow.cn/v1/chat/completions"
EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"
API_KEY = "sk-jozgtgkyvzxikozrtkzgyfuptcamffjnpofushlitmktwyst"
LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"
EMBED_MODEL = "BAAI/bge-large-zh-v1.5"

_JUNCTION_BASE = None

def _get_base():
    global _JUNCTION_BASE
    if _JUNCTION_BASE is not None:
        return _JUNCTION_BASE
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _JUNCTION_BASE = base
    return base

def _get_index_path():
    return os.path.join(_get_base(), "output", "prospectus_index.faiss")

def _get_chunks_path():
    return os.path.join(_get_base(), "output", "prospectus_chunks.json")

def _get_txt_dir():
    return os.path.join(_get_base(), "bs_challenge_financial_14b_dataset", "pdf_txt_file")

OUTPUT_DIR = None  # computed on demand
INDEX_PATH = None
CHUNKS_PATH = None
TXT_DIR = None

TOP_K = 5

# 全局缓存索引
_index = None
_chunks = None

# ─── 工具函数 ──────────────────────────────────────────────────

def call_llm(messages, temperature=0.01, max_tokens=2000):
    h = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    try:
        r = requests.post(API_URL, headers=h, json={
            "model": LLM_MODEL, "messages": messages,
            "temperature": 0.01, "max_tokens": 2000
        }, timeout=60)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
    except:
        pass
    return "【LLM调用失败】"


def embed(texts, retries=3):
    h = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    for a in range(retries):
        try:
            r = requests.post(EMBED_URL, headers=h, json={"model": EMBED_MODEL, "input": texts}, timeout=30)
            if r.status_code == 200:
                return np.array([d["embedding"] for d in r.json()["data"]], dtype=np.float32)
            time.sleep(2**a)
        except:
            time.sleep(2**a)
    return None


def extract_block(text, tag):
    pattern = rf'{tag}:\s*(.*?)(?=\n(?:Thought|Action|Final Answer)|\Z)'
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else None


# ─── 索引 ──────────────────────────────────────────────────

def load_index():
    global _index, _chunks
    if _index is not None:
        return _index, _chunks
    import faiss
    import numpy as np
    # 用 Python 的 open() 读文件（支持中文路径），转 numpy 数组再反序列化
    with open(_get_index_path(), 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    _index = faiss.deserialize_index(data)
    with open(_get_chunks_path(), 'r', encoding='utf-8') as f:
        _chunks = json.load(f)
    return _index, _chunks


# ─── 工具：检索招股书 ──────────────────────────────────

def search_prospectus(query):
    """搜索招股书，返回相关文本片段"""
    try:
        idx, chunks = load_index()
    except Exception as e:
        return f"【索引错误】{e}"
    
    emb = embed([query])
    if emb is None:
        return "【向量化失败】"
    import faiss
    faiss.normalize_L2(emb)
    scores, idxs = idx.search(emb, TOP_K)
    
    results = []
    for i, ix in enumerate(idxs[0]):
        if 0 <= ix < len(chunks):
            results.append({
                "score": float(scores[0][i]),
                "filename": chunks[ix]["filename"],
                "text": chunks[ix]["text"]
            })
    
    # 清理表格标记
    import re
    for r in results:
        r["text"] = re.sub(r'<\|[^>]+\|>', '', r["text"])
    
    if not results:
        return "检索无结果"
    
    text = f"找到 {len(results)} 个相关片段：\n\n"
    for i, r in enumerate(results):
        text += f"[片段{i+1}] (文件: {r['filename']}, 相关度: {r['score']:.3f})\n{r['text']}\n\n"
    return text


# ─── ReAct 智能体 ──────────────────────────────────

SYSTEM_PROMPT = """你是一个专业的招股说明书问答智能体。你有以下工具可用：

### 工具：search_prospectus
搜索招股说明书的内容。输入是搜索关键词，输出是相关文本片段。
可以多次搜索不同关键词来获取完整信息。

### 使用格式：

需要搜索时：
Thought: 分析当前需要什么信息，以及搜索关键词
Action: search_prospectus
Action Input: 搜索关键词

已有足够信息回答时：
Final Answer: 你的最终答案

### 重要规则：
1. **你应该先搜索再回答。但如果搜索失败（返回错误信息）或检索不到相关内容，可以凭你的知识回答。**
2. 一次只能调用一个工具
3. 如果搜索结果不完整或没找到关键信息，换不同的关键词再搜1-2次
4. 注意搜索文本中的表格标记如 <|TABLE_xxx|>，跳过它们看真实文本内容
5. 答案必须基于招股书内容或你的专业知识，不能编造
6. 如果经过2-3次不同关键词搜索仍找不到答案，可以凭你的知识回答
7. 答案要简洁准确，用中文"""


def init():
    """预加载索引"""
    try:
        load_index()
        return True
    except Exception:
        return False


def react_agent(question, max_steps=10):
    """ReAct 智能体：自主检索+推理"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"问题：{question}"}
    ]
    
    steps = []
    has_searched = False
    for step in range(max_steps):
        response = call_llm(messages, max_tokens=1500)
        messages.append({"role": "assistant", "content": response})
        
        # 检查是否有最终答案
        final = extract_block(response, "Final Answer")
        if final and has_searched:
            return final, steps
        elif final and not has_searched:
            # 搜索失败时允许直接回答（API key可能无效导致搜索不可用）
            if step >= 2:
                return final, steps
            # 先尝试搜索一次
            messages.append({"role": "user", "content": "请先尝试搜索招股书，如果搜索不到相关信息，可以直接凭你的知识回答。"})
            continue
        
        # 提取工具调用
        action = extract_block(response, "Action")
        action_input = extract_block(response, "Action Input")
        thought = extract_block(response, "Thought")
        
        if not action or not action_input:
            return response, steps  # 无法解析，直接返回
        
        steps.append({"thought": thought, "action": action, "input": action_input})
        
        # 执行工具
        if action == "search_prospectus":
            has_searched = True
            result = search_prospectus(action_input)
        else:
            result = f"未知工具: {action}"
        
        messages.append({"role": "user", "content": f"Observation: {result}"})
    
    return "【已达最大推理步数】", steps
