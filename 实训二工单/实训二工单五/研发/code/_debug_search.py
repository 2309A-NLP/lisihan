"""Debug search for 长远锂科"""
import sys
sys.path.insert(0, r'C:\Users\freedom\Desktop\招股书问答智能体\code')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from react_agent import search_prospectus, load_index, embed
import faiss, numpy as np

# Test different search terms
for term in ["长远锂科 发起人", "长远锂科 变更设立", "长远锂科", "法人 发起人"]:
    result = search_prospectus(term)
    lines = result.split('\n')
    relevant = [l for l in lines if '长远锂科' in l or '发起人' in l or '变更设立' in l]
    print(f"Search: '{term}'")
    print(f"  Total chars: {len(result)}")
    print(f"  Relevant lines: {len(relevant)}")
    for l in relevant[:5]:
        print(f"    {l[:150]}")
    print()
