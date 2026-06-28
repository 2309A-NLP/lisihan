from flask import Flask, jsonify, render_template, request

from agent import Agent
from db import (
    api_summary,
    delete_record,
    get_budget,
    get_record_by_id,
    get_remaining_budget,
    init_db,
    query_records,
    set_budget,
    update_record,
)


# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
init_db()
app = Flask(__name__)
agent = Agent()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "") or payload.get("query", "")).strip()
    try:
        return jsonify({"reply": agent.process(message), "summary": api_summary()})
    except Exception as exc:
        return jsonify({"reply": f"处理失败：{exc}\n还需要其他帮助吗？"}), 500


@app.get("/api/records")
def records():
    rows = query_records(
        member=request.args.get("member") or None,
        item_keyword=request.args.get("item_keyword") or request.args.get("keyword") or None,
        month=request.args.get("month") or None,
        type_=request.args.get("type") or None,
        start_date=request.args.get("start_date") or None,
        end_date=request.args.get("end_date") or None,
        record_id=request.args.get("id") or None,
        limit=int(request.args.get("limit", "100")),
    )
    return jsonify({"records": rows})


@app.delete("/api/records/<int:record_id>")
def delete_record_api(record_id):
    if not get_record_by_id(record_id):
        return jsonify({"message": f"没有找到记录#{record_id}"}), 404
    delete_record(record_id)
    return jsonify({"message": f"已删除记录#{record_id}"})


@app.put("/api/records/<int:record_id>")
def update_record_api(record_id):
    payload = request.get_json(silent=True) or {}
    field = payload.get("field")
    value = payload.get("value")
    if field is None or value is None:
        return jsonify({"message": "请提供 field 和 value"}), 400
    try:
        changed = update_record(record_id, field, value)
    except ValueError as exc:
        return jsonify({"message": str(exc)}), 400
    if not changed:
        return jsonify({"message": f"没有找到记录#{record_id}，或内容没有变化"}), 404
    return jsonify({"message": f"已将记录#{record_id}的{field}修改为{value}"})


@app.get("/api/summary")
def summary():
    return jsonify(api_summary(month=request.args.get("month") or None, type_=request.args.get("type") or None))


@app.get("/api/budget")
def budget_get():
    month = request.args.get("month") or None
    return jsonify({"budget": get_budget(month), "remaining": get_remaining_budget(month)})


@app.post("/api/budget")
def budget_post():
    payload = request.get_json(silent=True) or {}
    month = payload.get("month")
    budget = payload.get("budget")
    if not month or budget is None:
        return jsonify({"message": "请提供 month 和 budget，例如 {'month': '2026-06', 'budget': 5000}"}), 400
    return jsonify({"budget": set_budget(month, budget), "remaining": get_remaining_budget(month)})


def create_app():
    return app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=True)
