"""Diagnose why search returns nothing"""
import sys
sys.path.insert(0, r'C:\Users\freedom\Desktop\招股书问答智能体\code')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from react_agent import load_index, embed
import faiss
import numpy as np

# Check index
print("Loading index...")
try:
    idx, chunks = load_index()
    print(f"Index: {idx.ntotal} vectors, {len(chunks)} chunks")
except Exception as e:
    print(f"Load failed: {e}")
    raise

# Check embedding
print("\nTesting embed...")
emb = embed(["长远锂科"])
if emb is None:
    print("Embed FAILED - returned None")
else:
    print(f"Embed OK: shape={emb.shape}")
    faiss.normalize_L2(emb)
    scores, idxs = idx.search(emb, 5)
    print(f"Search results: {len(idxs[0])} ids")
    for i, ix in enumerate(idxs[0]):
        if ix >= 0:
            print(f"  [{scores[0][i]:.3f}] chunk #{ix}: {chunks[ix]['filename']}")
            print(f"    {chunks[ix]['text'][:150]}")
        else:
            print(f"  [{scores[0][i]:.3f}] id={ix} (invalid)")
