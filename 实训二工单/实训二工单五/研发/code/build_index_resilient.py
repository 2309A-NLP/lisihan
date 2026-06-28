"""建索引脚本 - 带重试和断点续跑"""
import sys, os, json, time, requests, numpy as np

sys.path.insert(0, r'C:\Users\freedom\Desktop\招股书问答智能体\code')

API_KEY = "sk-jozgtgkyvzxikozrtkzgyfuptcamffjnpofushlitmktwyst"
EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBED_MODEL = "BAAI/bge-large-zh-v1.5"
TXT_DIR = r'C:\Users\freedom\Desktop\招股书问答智能体\bs_challenge_financial_14b_dataset\pdf_txt_file'
OUT_DIR = r'C:\Users\freedom\Desktop\招股书问答智能体\output'
INDEX_PATH = os.path.join(OUT_DIR, "prospectus_index.faiss")
CHUNKS_PATH = os.path.join(OUT_DIR, "prospectus_chunks.json")
PROGRESS_PATH = os.path.join(OUT_DIR, "index_progress.json")

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
BATCH_SIZE = 16  # 更小的批次减少限流
MAX_RETRIES = 10  # 更多重试


def chunk_docs():
    """加载并分块"""
    all_chunks = []
    files = sorted([f for f in os.listdir(TXT_DIR) if f.endswith('.txt')])
    print(f"加载 {len(files)} 个 TXT 文件...")
    for fn in files:
        path = os.path.join(TXT_DIR, fn)
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            all_chunks.append({"filename": fn, "text": text[start:end]})
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return all_chunks


def embed_batch(texts):
    """向量化一批文本，带重试"""
    h = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    for a in range(MAX_RETRIES):
        try:
            r = requests.post(EMBED_URL, headers=h,
                json={"model": EMBED_MODEL, "input": texts}, timeout=60)
            if r.status_code == 200:
                return np.array([d["embedding"] for d in r.json()["data"]], dtype=np.float32)
            print(f"  API返回{r.status_code}，重试({a+1}/{MAX_RETRIES})", flush=True)
            time.sleep(5 * (a + 1))
        except Exception as e:
            print(f"  错误:{e}，重试({a+1}/{MAX_RETRIES})", flush=True)
            time.sleep(5 * (a + 1))
    return None


def build():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 分块
    chunks = chunk_docs()
    print(f"共 {len(chunks)} 个文本块")

    # 保存chunks（先保存，后面只管建索引）
    with open(CHUNKS_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False)

    texts = [c["text"] for c in chunks]
    n = len(texts)

    # 断点续跑：检查已有进度
    start_idx = 0
    all_emb = []
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, 'r') as f:
            prog = json.load(f)
        start_idx = prog.get("done", 0)
        # 加载已有的向量
        emb_files = sorted([f for f in os.listdir(OUT_DIR) if f.startswith("emb_batch_")])
        for ef in emb_files:
            all_emb.append(np.load(os.path.join(OUT_DIR, ef)))
        print(f"断点续跑: 已有 {start_idx}/{n}")

    # 逐批向量化
    for i in range(start_idx, n, BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        emb = embed_batch(batch)
        if emb is None:
            print(f"\n第{i//BATCH_SIZE+1}批失败，已保存进度到{start_idx}，下次可续跑")
            # 保存进度
            with open(PROGRESS_PATH, 'w') as f:
                json.dump({"done": i}, f)
            return False

        all_emb.append(emb)
        # 每200批保存一次向量到磁盘
        if len(all_emb) % 200 == 0:
            np.save(os.path.join(OUT_DIR, f"emb_batch_{i}.npy"), np.vstack(all_emb[-200:]))
        print(f"  {min(i+BATCH_SIZE, n)}/{n}", flush=True)
        time.sleep(0.2)

    # 合并所有向量
    print("合并向量...")
    embeddings = np.vstack(all_emb)

    # 建FAISS索引
    import faiss
    idx = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    idx.add(embeddings)
    faiss.write_index(idx, INDEX_PATH)

    # 清理临时文件
    for f in os.listdir(OUT_DIR):
        if f.startswith("emb_batch_"):
            os.remove(os.path.join(OUT_DIR, f))
    if os.path.exists(PROGRESS_PATH):
        os.remove(PROGRESS_PATH)

    print(f"索引完成! {idx.ntotal} 个向量 -> {INDEX_PATH}")
    return True


if __name__ == "__main__":
    print("开始构建招股书索引...")
    while True:
        ok = build()
        if ok:
            break
        print("重试构建（从上次进度续跑）...")
        time.sleep(3)
