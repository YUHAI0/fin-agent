"""供 Agent 管理本地新闻历史与订阅的工具。"""
import json

from fin_agent.news import (
    SOURCE_GLOBAL_CLS,
    SOURCE_GLOBAL_EM,
    SUPPORTED_NEWS_SOURCES,
)
from fin_agent.news_monitor import get_news_monitor, start_news_monitor
from fin_agent.news_store import (
    SUBSCRIPTION_TYPES,
    NewsSubscriptionStore,
    NotifiedNewsStore,
)


_UNSET = object()
_GLOBAL_NEWS_SOURCES = (SOURCE_GLOBAL_CLS, SOURCE_GLOBAL_EM)


def _result(data=None, message=None):
    payload = {"ok": True}
    if message:
        payload["message"] = message
    if data is not None:
        payload["data"] = data
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _error(message):
    return json.dumps(
        {"ok": False, "error": str(message)},
        ensure_ascii=False,
        indent=2,
    )


def _bool(value, field_name):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in ("true", "1", "yes", "on", "是", "启用"):
            return True
        if normalized in ("false", "0", "no", "off", "否", "停用", "暂停"):
            return False
    raise ValueError(f"{field_name} 必须是布尔值")


def _limit(value):
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValueError("limit 必须是整数")
    if value < 1 or value > 200:
        raise ValueError("limit 必须在 1 到 200 之间")
    return value


def _subscription_type(value):
    value = str(value or "").strip().casefold()
    if value not in SUBSCRIPTION_TYPES:
        raise ValueError("type 必须是 sector、topic 或 portfolio")
    return value


def _default_sources(subscription_type):
    if subscription_type == "portfolio":
        return list(SUPPORTED_NEWS_SOURCES)
    return list(_GLOBAL_NEWS_SOURCES)


def _normalize_sources_for_type(subscription_type, sources):
    """None、空数组或仅含空白值均表示使用该订阅类型的默认来源。"""
    if sources is None:
        return _default_sources(subscription_type)
    if isinstance(sources, str):
        sources = [sources]
    try:
        values = []
        for source in sources:
            source = str(source or "").strip()
            if source and source not in values:
                values.append(source)
    except TypeError:
        raise ValueError("sources 必须是新闻源数组")
    if not values:
        return _default_sources(subscription_type)

    invalid = set(values) - set(SUPPORTED_NEWS_SOURCES)
    if invalid:
        raise ValueError(f"不支持的新闻源：{', '.join(sorted(invalid))}")
    if subscription_type in ("sector", "topic"):
        forbidden = set(values) - set(_GLOBAL_NEWS_SOURCES)
        if forbidden:
            raise ValueError(
                "sector/topic 仅支持全局快讯源："
                f"{', '.join(_GLOBAL_NEWS_SOURCES)}；"
                "stock_news_em 只适用于 portfolio"
            )
    return values


def query_notified_news(
    unread=None,
    query=None,
    source=None,
    type=None,
    limit=20,
):
    """查询已由新闻监控记录并提醒过的新闻。"""
    try:
        unread_only = False if unread is None else _bool(unread, "unread")
        subscription_type = None
        if type is not None:
            subscription_type = _subscription_type(type)
        result = NotifiedNewsStore().list_news(
            source=str(source).strip() if source else None,
            unread_only=unread_only,
            subscription_type=subscription_type,
            query=str(query).strip() if query else None,
            limit=_limit(limit),
        )
        return _result(result)
    except (TypeError, ValueError, OSError) as exc:
        return _error(f"查询新闻失败：{exc}")


def create_news_subscription(
    type,
    name=None,
    keywords=None,
    exclude_keywords=None,
    sources=None,
):
    """创建板块、主题或动态持仓新闻订阅。"""
    try:
        subscription_type = _subscription_type(type)
        normalized_name = str(name or "").strip()
        if subscription_type in ("sector", "topic"):
            if not normalized_name and not keywords:
                raise ValueError("sector/topic 订阅至少需要 name 或 keywords")
            if not keywords:
                keywords = [normalized_name]
        elif not normalized_name:
            normalized_name = "全部持仓新闻"

        item = NewsSubscriptionStore().create_subscription(
            subscription_type=subscription_type,
            name=normalized_name,
            keywords=keywords,
            exclude_keywords=exclude_keywords,
            sources=_normalize_sources_for_type(subscription_type, sources),
        )
        message = "新闻订阅已创建"
        if subscription_type == "portfolio":
            message += "；该订阅会动态跟随全部组合中的持仓"
        return _result(item, message)
    except (TypeError, ValueError, OSError) as exc:
        return _error(f"创建新闻订阅失败：{exc}")


def list_news_subscriptions(enabled=None, type=None):
    """列出本地新闻订阅。"""
    try:
        if enabled is not None:
            enabled = _bool(enabled, "enabled")
        subscription_type = None
        if type is not None:
            subscription_type = _subscription_type(type)
        items = NewsSubscriptionStore().list_subscriptions(
            enabled=enabled,
            subscription_type=subscription_type,
        )
        return _result({"items": items, "total": len(items)})
    except (TypeError, ValueError, OSError) as exc:
        return _error(f"列出新闻订阅失败：{exc}")


def update_news_subscription(
    subscription_id,
    name=_UNSET,
    keywords=_UNSET,
    exclude_keywords=_UNSET,
    sources=_UNSET,
    enabled=_UNSET,
):
    """更新新闻订阅的可编辑字段。"""
    try:
        store = NewsSubscriptionStore()
        subscription_id = str(subscription_id).strip()
        existing = store.get_subscription(subscription_id)
        if existing is None:
            return _error(f"未找到新闻订阅：{subscription_id}")
        changes = {}
        for key, value in (
            ("name", name),
            ("keywords", keywords),
            ("exclude_keywords", exclude_keywords),
        ):
            if value is not _UNSET:
                changes[key] = value
        if sources is not _UNSET:
            changes["sources"] = _normalize_sources_for_type(
                existing["type"],
                sources,
            )
        if enabled is not _UNSET:
            changes["enabled"] = _bool(enabled, "enabled")
        if not changes:
            raise ValueError("请至少提供一个要更新的字段")
        item = store.update_subscription(subscription_id, **changes)
        if item is None:
            return _error(f"未找到新闻订阅：{subscription_id}")
        return _result(item, "新闻订阅已更新")
    except (TypeError, ValueError, OSError) as exc:
        return _error(f"更新新闻订阅失败：{exc}")


def pause_news_subscription(subscription_id):
    """暂停新闻订阅。"""
    return _set_news_subscription_enabled(subscription_id, False)


def enable_news_subscription(subscription_id):
    """启用新闻订阅。"""
    return _set_news_subscription_enabled(subscription_id, True)


def _set_news_subscription_enabled(subscription_id, enabled):
    try:
        item = NewsSubscriptionStore().set_enabled(
            str(subscription_id).strip(),
            enabled,
        )
        if item is None:
            return _error(f"未找到新闻订阅：{subscription_id}")
        action = "启用" if enabled else "暂停"
        return _result(item, f"新闻订阅已{action}")
    except (TypeError, ValueError, OSError) as exc:
        action = "启用" if enabled else "暂停"
        return _error(f"{action}新闻订阅失败：{exc}")


def delete_news_subscription(subscription_id):
    """删除新闻订阅。"""
    try:
        subscription_id = str(subscription_id).strip()
        deleted = NewsSubscriptionStore().delete_subscription(subscription_id)
        if not deleted:
            return _error(f"未找到新闻订阅：{subscription_id}")
        return _result(
            {"subscription_id": subscription_id, "deleted": True},
            "新闻订阅已删除",
        )
    except (TypeError, ValueError, OSError) as exc:
        return _error(f"删除新闻订阅失败：{exc}")


def refresh_news():
    """请求当前进程内的新闻监控立即执行一轮刷新。"""
    try:
        monitor = get_news_monitor()
        monitor_started = monitor is None
        if monitor is None:
            monitor = start_news_monitor()
        refresh_requested = monitor.refresh()
        status = monitor.status()
        accepted = monitor_started or refresh_requested
        if not accepted:
            return _result(
                {"accepted": False, "status": status},
                "新闻刷新正在进行，本次未重复启动",
            )
        return _result(
            {
                "accepted": True,
                "monitor_started": monitor_started,
                "refresh_requested": refresh_requested,
                "status": status,
            },
            "已接受后台刷新；若有匹配的新消息将通过桌面通知提醒",
        )
    except (RuntimeError, TypeError, ValueError, OSError) as exc:
        return _error(f"手动刷新新闻失败：{exc}")


_TYPE_PROPERTY = {
    "type": "string",
    "enum": list(SUBSCRIPTION_TYPES),
    "description": "订阅类型：sector（板块）、topic（主题）或 portfolio（全部组合持仓）。",
}
_SOURCES_PROPERTY = {
    "type": "array",
    "items": {"type": "string", "enum": list(SUPPORTED_NEWS_SOURCES)},
    "description": (
        "可选新闻源。省略或传空数组均使用类型默认值：sector/topic 仅使用"
        " stock_info_global_cls、stock_info_global_em；portfolio 默认使用全部三源。"
    ),
}
_KEYWORDS_PROPERTY = {
    "type": "array",
    "items": {"type": "string"},
    "description": "用于匹配标题和摘要的关键词。",
}
_EXCLUDE_KEYWORDS_PROPERTY = {
    "type": "array",
    "items": {"type": "string"},
    "description": "命中后排除新闻的关键词。",
}
_SUBSCRIPTION_ID_PROPERTY = {
    "type": "string",
    "description": "新闻订阅 ID。",
}


NEWS_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "query_notified_news",
            "description": "查询本地已提醒新闻，可按未读、关键词、来源和订阅类型筛选。",
            "parameters": {
                "type": "object",
                "properties": {
                    "unread": {
                        "type": "boolean",
                        "description": "true 时仅返回未读新闻；false 或省略时不限制。",
                    },
                    "query": {
                        "type": "string",
                        "description": "在标题、摘要、来源和股票代码中搜索。",
                    },
                    "source": {
                        "type": "string",
                        "enum": list(SUPPORTED_NEWS_SOURCES),
                        "description": "新闻来源。",
                    },
                    "type": _TYPE_PROPERTY,
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 200,
                        "default": 20,
                        "description": "最多返回的新闻条数。",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_news_subscription",
            "description": (
                "创建新闻订阅。sector/topic 用 name 和 keywords 匹配；"
                "portfolio 自动动态跟随全部组合持仓且不接受 symbols。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "type": _TYPE_PROPERTY,
                    "name": {
                        "type": "string",
                        "description": "订阅名称；sector/topic 建议填写板块或主题名。",
                    },
                    "keywords": _KEYWORDS_PROPERTY,
                    "exclude_keywords": _EXCLUDE_KEYWORDS_PROPERTY,
                    "sources": _SOURCES_PROPERTY,
                },
                "required": ["type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_news_subscriptions",
            "description": "列出新闻订阅，可按启用状态和订阅类型筛选。",
            "parameters": {
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "description": "按启用状态筛选。",
                    },
                    "type": _TYPE_PROPERTY,
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_news_subscription",
            "description": "更新新闻订阅；只传需要变更的字段，也可通过 enabled 暂停或启用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subscription_id": _SUBSCRIPTION_ID_PROPERTY,
                    "name": {"type": "string", "description": "新的订阅名称。"},
                    "keywords": _KEYWORDS_PROPERTY,
                    "exclude_keywords": _EXCLUDE_KEYWORDS_PROPERTY,
                    "sources": _SOURCES_PROPERTY,
                    "enabled": {
                        "type": "boolean",
                        "description": "是否启用订阅。",
                    },
                },
                "required": ["subscription_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pause_news_subscription",
            "description": "暂停指定新闻订阅，保留订阅配置。",
            "parameters": {
                "type": "object",
                "properties": {"subscription_id": _SUBSCRIPTION_ID_PROPERTY},
                "required": ["subscription_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "enable_news_subscription",
            "description": "重新启用指定新闻订阅。",
            "parameters": {
                "type": "object",
                "properties": {"subscription_id": _SUBSCRIPTION_ID_PROPERTY},
                "required": ["subscription_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_news_subscription",
            "description": "永久删除指定新闻订阅。",
            "parameters": {
                "type": "object",
                "properties": {"subscription_id": _SUBSCRIPTION_ID_PROPERTY},
                "required": ["subscription_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "refresh_news",
            "description": "请求本地新闻监控立即在后台刷新一次，不直接发起 HTTP 请求。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]
