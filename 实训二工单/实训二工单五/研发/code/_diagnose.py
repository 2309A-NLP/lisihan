"""诊断：手动搜索几个失败的问题"""
import sys, json
sys.path.insert(0, r'C:\Users\freedom\Desktop\招股书问答智能体\code')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from react_agent import load_index
import faiss
import numpy as np
import requests
import os

API_KEY = "sk-jozgtgkyvzxikozrtkzgyfuptcamffjnpofushlitmktwyst"
EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBED_MODEL = "BAAI/bge-large-zh-v1.5"

def embed(texts):
    h = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    r = requests.post(EMBED_URL, headers=h, json={"model": EMBED_MODEL, "input": texts}, timeout=30)
    if r.status_code == 200:
        return np.array([d["embedding"] for d in r.json()["data"]], dtype=np.float32)
    return None

idx, chunks = load_index()

# 测试1: 华瑞电器 专利
print("测试1: 华瑞电器 获得多少项专利")
emb = embed(["华瑞电器 专利"])
if emb is not None:
    faiss.normalize_L2(emb)
    scores, idxs = idx.search(emb, 5)
    for i, ix in enumerate(idxs[0]):
        if ix >= 0:
            fn = chunks[ix]["filename"]
            txt = chunks[ix]["text"][:200]
            print(f"  [{scores[0][i]:.3f}] {fn}: {txt}")
            print()

# 测试2: 大连派思燃气 控股股东
print("测试2: 大连派思燃气 控股股东 持股比例")
emb = embed(["大连派思燃气 控股股东 持股"])
if emb is not None:
    faiss.normalize_L2(emb)
    scores, idxs = idx.search(emb, 5)
    for i, ix in enumerate(idxs[0]):
        if ix >= 0:
            fn = chunks[ix]["filename"]
            txt = chunks[ix]["text"][:200]
            print(f"  [{scores[0][i]:.3f}] {fn}: {txt}")
            print()
