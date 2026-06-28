"""Trace ReAct loop - why does it hit max steps?"""
import sys
sys.path.insert(0, r'C:\Users\freedom\Desktop\招股书问答智能体\code')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from react_agent import call_llm, search_prospectus, extract_block, SYSTEM_PROMPT

question = "湖南长远锂科股份有限公司变更设立时作为发起人的法人有哪些？"

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": f"问题：{question}"}
]

for step in range(12):
    resp = call_llm(messages, max_tokens=1500)
    messages.append({"role": "assistant", "content": resp})
    
    final = extract_block(resp, "Final Answer")
    if final:
        print(f"Step {step+1}: FINAL ANSWER -> {final[:200]}")
        break
    
    thought = extract_block(resp, "Thought")
    action = extract_block(resp, "Action")
    action_input = extract_block(resp, "Action Input")
    
    print(f"Step {step+1}: action={action} input={action_input}")
    
    if action == "search_prospectus" and action_input:
        obs = search_prospectus(action_input)
        # Check if it has relevant info
        has_key = "长远锂科" in obs or "发起人" in obs or "法人" in obs
        print(f"  obs len={len(obs)}, has_key={has_key}")
        messages.append({"role": "user", "content": f"Observation: {obs[:2000]}"})
    else:
        print(f"  Unknown action or no input")
        break
