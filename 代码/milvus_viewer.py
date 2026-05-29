# -*- coding: utf-8 -*-
"""Milvus 数据可视化页面。

运行方式:
    streamlit run milvus_viewer.py
"""

import pandas as pd
import streamlit as st
from pymilvus import MilvusClient

from config.config import settings


URI = f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
KNOWLEDGE_COLLECTION = settings.MILVUS_COLLECTION
MEMORY_COLLECTION = settings.MILVUS_MEMORY_COLLECTION

@st.cache_resource(show_spinner=False)
def get_client() -> MilvusClient:
    """创建并缓存 Milvus 客户端，避免 Streamlit 每次刷新都重复建连接。"""
    return MilvusClient(uri=URI, timeout=10)


def collection_count(client: MilvusClient, collection_name: str) -> int:
    """统计集合总量；先 flush 再读 stats，保证刚写入的数据也能显示。"""
    if not client.has_collection(collection_name):
        return 0

    try:
        client.flush(collection_name)
    except Exception:
        pass

    try:
        stats = client.get_collection_stats(collection_name)
        if isinstance(stats, dict) and stats.get("row_count") is not None:
            return int(stats["row_count"])
    except Exception:
        pass

    rows = client.query(
        collection_name=collection_name,
        filter="id >= 0",
        output_fields=["count(*)"],
    )
    if rows and rows[0].get("count(*)") is not None:
        return int(rows[0]["count(*)"])
    return 0


def safe_query(client: MilvusClient, collection_name: str, output_fields: list[str], limit: int, expr: str = "") -> list[dict]:
    """查询集合数据；字段不兼容时退回到最基础的 id/content 字段。"""
    query_expr = expr or "id >= 0"
    try:
        rows = client.query(
            collection_name=collection_name,
            filter=query_expr,
            output_fields=output_fields,
            limit=limit,
        )
        return list(rows)
    except Exception:
        rows = client.query(
            collection_name=collection_name,
            filter="id >= 0",
            output_fields=["id", "content"],
            limit=limit,
        )
        return list(rows)


def trim_for_table(rows: list[dict]) -> list[dict]:
    """隐藏向量字段并截断过长文本，让表格更容易阅读。"""
    cleaned = []
    for row in rows:
        item = {key: value for key, value in row.items() if key != "embedding"}
        content = str(item.get("content", ""))
        if len(content) > 180:
            item["content"] = content[:180] + "..."
        cleaned.append(item)
    return cleaned


def get_collection_fields(collection_name: str) -> tuple[str, list[str]]:
    """根据集合类型返回页面标签和默认展示字段。"""
    if collection_name == KNOWLEDGE_COLLECTION:
        return "知识库", ["id", "content", "role_ids", "source_file"]
    if collection_name == MEMORY_COLLECTION:
        return "长期记忆", ["id", "user_id", "role_id", "conversation_id", "sender", "content", "timestamp"]
    return "自定义集合", ["id", "content"]


st.set_page_config(page_title="Milvus 可视化", layout="wide")
st.title("Milvus 可视化")
st.caption(f"连接地址: {URI}")

try:
    client = get_client()
    all_collections = client.list_collections()
except Exception as exc:
    st.error(f"连接 Milvus 失败: {exc}")
    st.stop()

configured_collections = [KNOWLEDGE_COLLECTION, MEMORY_COLLECTION]
collections = [name for name in configured_collections if name in all_collections]
collections += [name for name in all_collections if name not in collections]

if not collections:
    st.warning("Milvus 中还没有集合。")
    st.stop()

summary_cols = st.columns(2)
for index, name in enumerate(configured_collections):
    label, _ = get_collection_fields(name)
    exists = name in all_collections
    total = collection_count(client, name) if exists else 0
    summary_cols[index].metric(f"{label}总数量", total, help=name)

selected_col = st.selectbox("选择集合", collections)
label, output_fields = get_collection_fields(selected_col)
total_count = collection_count(client, selected_col)

st.subheader(f"{label}: {selected_col}")
st.write(f"总数据量: {total_count}")

with st.sidebar:
    st.header("查询选项")
    role_id = st.number_input("角色ID，0表示全部", min_value=0, value=0, step=1)
    user_id = st.number_input("用户ID，0表示全部", min_value=0, value=0, step=1)
    limit = st.slider("显示条数", 1, 200, 20)

expr = "id >= 0"
if selected_col == MEMORY_COLLECTION:
    filters = []
    if role_id > 0:
        filters.append(f"role_id == {int(role_id)}")
    if user_id > 0:
        filters.append(f"user_id == {int(user_id)}")
    expr = " and ".join(filters) if filters else "id >= 0"

rows = safe_query(client, selected_col, output_fields, max(limit, 200 if role_id > 0 else limit), expr)

if selected_col == KNOWLEDGE_COLLECTION and role_id > 0:
    rows = [row for row in rows if int(role_id) in [int(value) for value in row.get("role_ids", [])]]

rows = rows[:limit]

if rows:
    df = pd.DataFrame(trim_for_table(rows))
    st.dataframe(df, use_container_width=True, hide_index=True)

    selected_id = st.selectbox("选择ID查看详情", df["id"].tolist())
    detail = safe_query(client, selected_col, ["*"], 1, f"id == {int(selected_id)}")
    if detail:
        detail_item = {key: value for key, value in detail[0].items() if key != "embedding"}
        st.json(detail_item)
else:
    st.warning("没有找到数据。")

st.subheader("向量搜索")
search_text = st.text_input("输入搜索文本")
if search_text:
    try:
        if selected_col == MEMORY_COLLECTION:
            from app.core.memory import memory_manager

            vector = memory_manager._build_embedding(search_text)
            search_filter = expr if expr != "id >= 0" else ""
        else:
            from app.core.rag import rag_system

            vector = rag_system.get_embedding(search_text)
            search_filter = ""

        search_results = client.search(
            collection_name=selected_col,
            data=[vector],
            filter=search_filter,
            limit=limit,
            output_fields=output_fields,
            search_params={"metric_type": "L2", "params": {"nprobe": 10}},
        )
        hits = []
        for group in search_results:
            for hit in group:
                entity = dict(hit.get("entity", {}))
                entity["distance"] = hit.get("distance")
                hits.append(entity)
        st.dataframe(pd.DataFrame(trim_for_table(hits)), use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"向量搜索失败: {exc}")
