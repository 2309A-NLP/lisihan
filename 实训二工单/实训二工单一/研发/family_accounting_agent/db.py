import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal


# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "money.db")
NOTE_TABLE = "money_notes"
BUDGET_TABLE = "budget_config"
ALLOWED_UPDATE_FIELDS = {"date", "member", "item", "amount", "type"}


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
    finally:
        conn.close()


def _normalize_row(row):
    if not row:
        return None
    result = {}
    for key in row.keys():
        value = row[key]
        if isinstance(value, Decimal):
            result[key] = float(value)
        elif isinstance(value, (date, datetime)):
            result[key] = value.isoformat()
        else:
            result[key] = value
    return result


def _month_range(month):
    month = month or datetime.now().strftime("%Y-%m")
    start = datetime.strptime(month, "%Y-%m").date()
    if start.month == 12:
        end = date(start.year + 1, 1, 1)
    else:
        end = date(start.year, start.month + 1, 1)
    return start.isoformat(), end.isoformat()


def _signed_amount(amount, record_type):
    value = abs(float(amount))
    return value if record_type == "收入" else -value


def init_db():
    with _connect() as conn:
        conn.executescript(f"""
            CREATE TABLE IF NOT EXISTS {NOTE_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                member TEXT NOT NULL,
                item TEXT NOT NULL,
                amount REAL NOT NULL,
                type TEXT NOT NULL CHECK(type IN ('收入','支出')),
                created_at TEXT DEFAULT (datetime('now','localtime'))
            );
            CREATE TABLE IF NOT EXISTS {BUDGET_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                month TEXT NOT NULL UNIQUE,
                budget_amount REAL NOT NULL,
                updated_at TEXT DEFAULT (datetime('now','localtime'))
            );
        """)
        conn.commit()


def add_record(date_, member, item, amount, type_):
    if type_ not in {"收入", "支出"}:
        raise ValueError('type 必须是"收入"或"支出"')
    with _connect() as conn:
        cursor = conn.execute(
            f"INSERT INTO {NOTE_TABLE} (date, member, item, amount, type) VALUES (?, ?, ?, ?, ?)",
            (date_, member, item, _signed_amount(amount, type_), type_),
        )
        record_id = cursor.lastrowid
        conn.commit()
        return record_id


def query_records(
    member=None,
    item_keyword=None,
    month=None,
    type_=None,
    start_date=None,
    end_date=None,
    limit=100,
    record_id=None,
    keyword=None,
):
    if keyword and not item_keyword:
        item_keyword = keyword

    sql = f"SELECT id, date, member, item, amount, type, created_at FROM {NOTE_TABLE} WHERE 1=1"
    params = []

    if record_id:
        sql += " AND id = ?"
        params.append(int(record_id))
    if member:
        sql += " AND member = ?"
        params.append(member)
    if item_keyword:
        sql += " AND item LIKE ?"
        params.append(f"%{item_keyword}%")
    if type_:
        sql += " AND type = ?"
        params.append(type_)
    if month:
        month_start, month_end = _month_range(month)
        sql += " AND date >= ? AND date < ?"
        params.extend([month_start, month_end])
    if start_date:
        sql += " AND date >= ?"
        params.append(start_date)
    if end_date:
        sql += " AND date <= ?"
        params.append(end_date)

    sql += " ORDER BY date DESC, id DESC LIMIT ?"
    params.append(int(limit))

    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
        return [_normalize_row(row) for row in rows]


def get_record_by_id(record_id):
    rows = query_records(record_id=record_id, limit=1)
    return rows[0] if rows else None


def delete_record(record_id):
    with _connect() as conn:
        cursor = conn.execute(f"DELETE FROM {NOTE_TABLE} WHERE id = ?", (int(record_id),))
        affected = cursor.rowcount
        conn.commit()
        return bool(affected)


def update_record(record_id, field, new_value):
    if field not in ALLOWED_UPDATE_FIELDS:
        raise ValueError("不支持修改该字段")

    existing = get_record_by_id(record_id)
    if not existing:
        return False

    if field == "type":
        if new_value not in {"收入", "支出"}:
            raise ValueError('type 必须是"收入"或"支出"')
        amount = _signed_amount(existing["amount"], new_value)
        sql = f"UPDATE {NOTE_TABLE} SET type = ?, amount = ? WHERE id = ?"
        params = (new_value, amount, int(record_id))
    elif field == "amount":
        amount = _signed_amount(new_value, existing["type"])
        sql = f"UPDATE {NOTE_TABLE} SET amount = ? WHERE id = ?"
        params = (amount, int(record_id))
    else:
        sql = f"UPDATE {NOTE_TABLE} SET \"{field}\" = ? WHERE id = ?"
        params = (new_value, int(record_id))

    with _connect() as conn:
        cursor = conn.execute(sql, params)
        affected = cursor.rowcount
        conn.commit()
        return bool(affected)


def get_summary(member=None, month=None, type_=None):
    month = month or datetime.now().strftime("%Y-%m")
    month_start, month_end = _month_range(month)
    filters = ["date >= ?", "date < ?"]
    params = [month_start, month_end]

    if member:
        filters.append("member = ?")
        params.append(member)
    if type_:
        filters.append("type = ?")
        params.append(type_)

    where_clause = " AND ".join(filters)
    with _connect() as conn:
        row = conn.execute(
            f"""
            SELECT
                COALESCE(SUM(CASE WHEN type = '收入' THEN amount ELSE 0 END), 0) AS income,
                COALESCE(SUM(CASE WHEN type = '支出' THEN ABS(amount) ELSE 0 END), 0) AS expense,
                COALESCE(SUM(amount), 0) AS net,
                COUNT(*) AS count
            FROM {NOTE_TABLE}
            WHERE {where_clause}
            """,
            params,
        ).fetchone()
        summary = _normalize_row(row) if row else None
    return {"month": month, **(summary or {})}


def set_budget(month, budget_amount):
    with _connect() as conn:
        conn.execute(
            f"""
            INSERT INTO {BUDGET_TABLE} (month, budget_amount)
            VALUES (?, ?)
            ON CONFLICT(month) DO UPDATE SET budget_amount = excluded.budget_amount,
                                              updated_at = datetime('now','localtime')
            """,
            (month, abs(float(budget_amount))),
        )
        conn.commit()
    return get_budget(month)


def get_budget(month):
    month = month or datetime.now().strftime("%Y-%m")
    with _connect() as conn:
        row = conn.execute(
            f"SELECT month, budget_amount, updated_at FROM {BUDGET_TABLE} WHERE month = ?",
            (month,),
        ).fetchone()
        return _normalize_row(row) if row else None


def get_remaining_budget(month):
    month = month or datetime.now().strftime("%Y-%m")
    budget = get_budget(month)
    if not budget:
        return None
    summary = get_summary(month=month)
    expense = float(summary.get("expense", 0))
    budget_amount = float(budget["budget_amount"])
    return {
        "month": month,
        "budget": round(budget_amount, 2),
        "expense": round(expense, 2),
        "remaining": round(budget_amount - expense, 2),
    }


def clear_all_data():
    with _connect() as conn:
        conn.execute(f"DELETE FROM {NOTE_TABLE}")
        conn.execute(f"DELETE FROM {BUDGET_TABLE}")
        conn.execute(f"DELETE FROM sqlite_sequence WHERE name IN ('{NOTE_TABLE}', '{BUDGET_TABLE}')")
        conn.commit()


def api_summary(month=None, type_=None):
    month = month or datetime.now().strftime("%Y-%m")
    today = datetime.now().strftime("%Y-%m-%d")
    expense_rows = query_records(month=month, type_="支出", limit=1000)
    income_rows = query_records(month=month, type_="收入", limit=1000)
    today_rows = query_records(start_date=today, end_date=today, type_="支出", limit=1000)
    rows = query_records(month=month, type_=type_, limit=1000) if type_ else query_records(month=month, limit=1000)

    by_member = {}
    for row in expense_rows:
        by_member[row["member"]] = by_member.get(row["member"], 0) + abs(float(row["amount"]))

    return {
        "month": month,
        "month_total": round(sum(abs(float(row["amount"])) for row in expense_rows), 2),
        "month_income": round(sum(float(row["amount"]) for row in income_rows), 2),
        "today_total": round(sum(abs(float(row["amount"])) for row in today_rows), 2),
        "count": len(rows),
        "by_member": [
            {"member": member, "total": round(total, 2)}
            for member, total in sorted(by_member.items(), key=lambda item: item[1], reverse=True)
        ],
        "recent": query_records(limit=12),
        "budget": get_remaining_budget(month),
    }


class DB:
    def __init__(self):
        init_db()

    def add_record(self, date_, member, item, amount, type_):
        return add_record(date_, member, item, amount, type_)

    def query_records(self, **kwargs):
        return query_records(**kwargs)

    def get_record_by_id(self, record_id):
        return get_record_by_id(record_id)

    def delete_record(self, record_id):
        return delete_record(record_id)

    def update_record(self, record_id, field, new_value):
        return update_record(record_id, field, new_value)

    def get_summary(self, member=None, month=None, type_=None):
        return get_summary(member=member, month=month, type_=type_)

    def set_budget(self, month, budget_amount):
        return set_budget(month, budget_amount)

    def get_budget(self, month):
        return get_budget(month)

    def get_remaining_budget(self, month):
        return get_remaining_budget(month)

    def clear_all_data(self):
        return clear_all_data()

    def api_summary(self, month=None, type_=None):
        return api_summary(month=month, type_=type_)

    def close(self):
        pass
