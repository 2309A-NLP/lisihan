"""
招股书问答智能体 - ReAct Agent (Web服务)
基于向量检索的招股说明书智能问答
"""
import sys, os, json, threading
sys.path.insert(0, os.path.dirname(__file__))
from flask import Flask, request, render_template, jsonify

from react_agent import react_agent

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = (data.get('question') or data.get('query') or '').strip()
    if not question:
        return jsonify({'answer': '请输入问题', 'type': ''})
    
    try:
        answer, _ = react_agent(question)
        is_error = answer.startswith('【')
        return jsonify({
            'answer': answer,
            'type': 'AI',
            'error': 'error' if is_error else ''
        })
    except Exception as e:
        return jsonify({
            'answer': f'出错: {str(e)}',
            'type': '',
            'error': str(e)
        })

if __name__ == '__main__':
    import webbrowser
    port = 5003
    print(f"招股书问答智能体启动中... http://localhost:{port}")
    threading.Timer(0.5, lambda: webbrowser.open(f'http://localhost:{port}')).start()
    app.run(host='localhost', port=port, debug=False)
