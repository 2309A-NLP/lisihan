from app import app, create_app


# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
if __name__ == "__main__":
    create_app()
    app.run(host="0.0.0.0", port=8080, debug=True)
