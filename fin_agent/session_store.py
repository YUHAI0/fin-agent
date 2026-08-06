"""多标签会话的磁盘持久化：索引 + 正文分离。

目录结构：
    {config_dir}/sessions/index.json      元数据索引
    {config_dir}/sessions/<id>.json       单会话正文
"""
import json
import os
import threading
import time
import uuid

from fin_agent.config import Config

INDEX_VERSION = 1
DEFAULT_TITLE = "新会话"
PREVIEW_LIMIT = 40

_LOCK = threading.RLock()


def sessions_dir():
    path = os.path.join(Config.get_config_dir(), "sessions")
    os.makedirs(path, exist_ok=True)
    return path


def _index_path():
    return os.path.join(sessions_dir(), "index.json")


def _session_path(session_id):
    return os.path.join(sessions_dir(), f"{session_id}.json")


def _load_index():
    path = _index_path()
    if not os.path.exists(path):
        return {"version": INDEX_VERSION, "sessions": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "sessions" not in data:
            return {"version": INDEX_VERSION, "sessions": []}
        return data
    except Exception:
        return {"version": INDEX_VERSION, "sessions": []}


def _save_index(index):
    with open(_index_path(), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def _find(index, session_id):
    for entry in index["sessions"]:
        if entry.get("id") == session_id:
            return entry
    return None


def _sorted(entries):
    """置顶优先，其余按 updated_at 倒序。"""
    return sorted(entries, key=lambda e: (0 if e.get("pinned") else 1, -e.get("updated_at", 0)))


def list_sessions(offset=0, limit=30):
    with _LOCK:
        index = _load_index()
        ordered = _sorted(index["sessions"])
        return {
            "sessions": ordered[offset:offset + limit],
            "total": len(ordered),
        }


def create_session(title=DEFAULT_TITLE):
    with _LOCK:
        index = _load_index()
        now = int(time.time())
        entry = {
            "id": str(uuid.uuid4()),
            "title": title or DEFAULT_TITLE,
            "created_at": now,
            "updated_at": now,
            "pinned": False,
            "message_count": 0,
            "preview": "",
        }
        index["sessions"].append(entry)
        _save_index(index)
        _write_body(entry["id"], {"id": entry["id"], "llm_history": [], "ui_messages": []})
        return entry


def _write_body(session_id, body):
    with open(_session_path(session_id), "w", encoding="utf-8") as f:
        json.dump(body, f, ensure_ascii=False, indent=2)


def _read_body(session_id):
    path = _session_path(session_id)
    if not os.path.exists(path):
        raise KeyError(session_id)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_session(session_id):
    with _LOCK:
        body = _read_body(session_id)
        body.setdefault("llm_history", [])
        body.setdefault("ui_messages", [])
        return body


def delete_session(session_id):
    with _LOCK:
        index = _load_index()
        before = len(index["sessions"])
        index["sessions"] = [e for e in index["sessions"] if e.get("id") != session_id]
        _save_index(index)
        path = _session_path(session_id)
        if os.path.exists(path):
            os.remove(path)
        return len(index["sessions"]) < before


def rename_session(session_id, title):
    with _LOCK:
        index = _load_index()
        entry = _find(index, session_id)
        if entry is None:
            return False
        entry["title"] = (title or DEFAULT_TITLE).strip()[:60]
        entry["updated_at"] = int(time.time())
        _save_index(index)
        return True


def set_pinned(session_id, pinned):
    with _LOCK:
        index = _load_index()
        entry = _find(index, session_id)
        if entry is None:
            return False
        entry["pinned"] = bool(pinned)
        _save_index(index)
        return True


def save_llm_history(session_id, llm_history):
    with _LOCK:
        try:
            body = _read_body(session_id)
        except KeyError:
            body = {"id": session_id, "llm_history": [], "ui_messages": []}
        body["llm_history"] = llm_history
        _write_body(session_id, body)
        _touch(session_id, message_count=_count_messages(llm_history))


def save_ui_messages(session_id, ui_messages):
    with _LOCK:
        try:
            body = _read_body(session_id)
        except KeyError:
            body = {"id": session_id, "llm_history": [], "ui_messages": []}
        body["ui_messages"] = ui_messages
        _write_body(session_id, body)
        _touch(session_id, preview=_first_user_text(ui_messages))


def _count_messages(llm_history):
    return len([m for m in llm_history if m.get("role") in ("user", "assistant")])


def _first_user_text(ui_messages):
    for msg in ui_messages or []:
        if msg.get("role") == "user":
            return str(msg.get("content", ""))[:PREVIEW_LIMIT]
    return ""


def _touch(session_id, message_count=None, preview=None):
    index = _load_index()
    entry = _find(index, session_id)
    if entry is None:
        entry = {
            "id": session_id,
            "title": DEFAULT_TITLE,
            "created_at": int(time.time()),
            "pinned": False,
            "message_count": 0,
            "preview": "",
        }
        index["sessions"].append(entry)
    entry["updated_at"] = int(time.time())
    if message_count is not None:
        entry["message_count"] = message_count
    if preview:
        entry["preview"] = preview
    _save_index(index)


def search_sessions(keyword, max_scan=200, time_budget=2.0):
    """全文搜索。最多扫 max_scan 个会话，超过 time_budget 秒即截断。"""
    keyword = (keyword or "").strip()
    if not keyword:
        return {"sessions": [], "truncated": False}

    with _LOCK:
        index = _load_index()
        ordered = _sorted(index["sessions"])

    started = time.time()
    hits = []
    truncated = False
    for i, entry in enumerate(ordered):
        if i >= max_scan:
            truncated = True
            break
        if time.time() - started > time_budget:
            truncated = True
            break
        if keyword in entry.get("title", ""):
            hits.append(entry)
            continue
        try:
            raw = json.dumps(_read_body(entry["id"]).get("ui_messages", []), ensure_ascii=False)
        except Exception:
            continue
        if keyword in raw:
            hits.append(entry)
    return {"sessions": hits, "truncated": truncated}
