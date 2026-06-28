"""Trace the ReAct loop for a failed case"""
import sys
sys.path.insert(0, r'C:\Users\freedom\Desktop\招股书问答智能体\code')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from react_agent import search_prospectus, call_llm, extract_block, SYSTEM_PROMPT

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "华瑞电器股份有限公司获得多少项国内专利？其中有多少项发明专利？"}
]

for step in range(4):
    resp = call_llm(messages, max_tokens=1500)
    messages.append({"role": "assistant", "content": resp})
    
    final = extract_block(resp, "Final Answer")
    if final:
        print(f"Step {step+1}: FINAL ANSWER -> {final[:200]}")
        break
    
    thought = extract_block(resp, "Thought")
    action = extract_block(resp, "Action")
    action_input = extract_block(resp, "Action Input")
    
    print(f"Step {step+1}: Thought={thought}")
    print(f"  Action={action}")
    print(f"  Input={action_input}")
    
    if action == "search_prospectus" and action_input:
        obs = search_prospectus(action_input)
        has_67 = "67" in obs
        has_patent = "专利" in obs
        has_83 = "83" in obs
        print(f"  Observation: {len(obs)} chars, 67={has_67}, 专利={has_patent}, 83={has_83}")
        messages.append({"role": "user", "content": f"Observation: {obs}"})
