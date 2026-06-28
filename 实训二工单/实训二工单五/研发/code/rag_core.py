"""
招股书问答智能体 - RAG 核心引擎
读取 TXT → 建索引 → 向量检索 → LLM回答
"""
import os, json, time, requests, numpy as np
from config import *


def load_txt_files():
    """加载所有 TXT 文件并分块"""
    txt_dir = os.path.join(BASE, "..", "bs_challenge_financial_14b_dataset", "pdf_txt_file")
    all_chunks = []
    if not os.path.isdir(txt_dir):
        print(f"[ERROR] TXT 目录不存在: {txt_dir}")
        return all_chunks
    files = sorted([f for f in os.listdir(txt_dir) if f.endswith('.txt')])
    print(f"找到 {len(files)} 个 TXT 文件")
    for fn in files:
        path = os.path.join(txt_dir, fn)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            for i in range(0, len(text), CHUNK_SIZE):
                end = min(i + CHUNK_SIZE, len(text))
                all_chunks.append({
                    "filename": fn,
                    "chunk_id": f"{fn}#{i//(CHUNK_SIZE-CHUNK_OVERLAP)}",
                    "text": text[i:end]
                })
        except Exception as e:
            print(f"  [WARN] {fn}: {e}")
    print(f"共 {len(all_chunks)} 个文本块")
    return all_chunks


def embed_texts(texts, retries=3):
    """批量获取 embedding"""
    url = f"{API_BASE}/embeddings"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers,
                json={"model": EMBEDDING_MODEL, "input": texts},
                timeout=EMBEDDING_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                return np.array([item["embedding"] for item in data["data"]], dtype=np.float32)
            time.sleep(2 ** attempt)
        except:
            time.sleep(2 ** attempt)
    return None


def build_index(chunks):
    """构建 FAISS 索引"""
    import faiss
    texts = [c["text"] for c in chunks]
    print(f"向量化 {len(texts)} 个块...")
    batch_size = 32
    all_emb = []
    for i in range(0, len(texts), batch_size):
        emb = embed_texts(texts[i:i+batch_size])
        if emb is None:
            raise Exception(f"第 {i//batch_size+1} 批向量化失败")
        all_emb.append(emb)
        print(f"  {min(i+batch_size, len(texts))}/{len(texts)}", flush=True)
        time.sleep(0.1)
    embeddings = np.vstack(all_emb)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index, chunks


def load_index():
    """加载已有索引"""
    import faiss
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    return index, chunks


def retrieve(query, index, chunks, top_k=TOP_K):
    """检索相关片段"""
    emb = embed_texts([query])
    if emb is None:
        return []
    import faiss
    faiss.normalize_L2(emb)
    scores, indices = index.search(emb, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx >= 0 and idx < len(chunks):
            results.append({"score": float(scores[0][i]), "filename": chunks[idx]["filename"], "text": chunks[idx]["text"]})
    return results


def ask_llm(question, context):
    """LLM 回答"""
    url = f"{API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    system = "你是一个专业的招股说明书问答助手。严格基于上下文回答，不编造。如果找不到答案，说'无法从提供的资料中找到'。保持简洁准确。"
    user = f"## 上下文（招股说明书片段）\n\n{context}\n\n## 问题\n{question}\n\n## 回答"
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json={
                "model": LLM_MODEL, "temperature": LLM_TEMPERATURE, "max_tokens": 1024,
                "messages": [{"role":"system","content":system},{"role":"user","content":user}]
            }, timeout=LLM_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            time.sleep(3)
        except:
            time.sleep(3)
    return None


def answer_question(question, index, chunks):
    """单题完整流程"""
    results = retrieve(question, index, chunks)
    if not results:
        return "[检索失败]"
    context = "\n\n".join(f"[文档{i+1}]({r['filename']}, score={r['score']:.3f})\n{r['text']}" for i, r in enumerate(results))
    answer = ask_llm(question, context)
    return answer or "[LLM请求失败]"
