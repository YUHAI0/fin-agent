"""新闻订阅、历史与监控状态的本地 JSON 存储。"""
import datetime
import difflib
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import uuid

from fin_agent.config import Config
from fin_agent.datasources.normalize import to_ts_code
from fin_agent.news import SOURCE_STOCK_NEWS_EM, SUPPORTED_NEWS_SOURCES


logger = logging.getLogger(__name__)


SUBSCRIPTION_TYPES = ("sector", "topic", "portfolio")
# sector/topic 缺少个股上下文，stock_news_em 仅适用于 portfolio；与
# fin_agent.tools.news_tools._normalize_sources_for_type 保持一致的类型约束，
# 确保无论走 Agent 工具还是 HTTP API 都无法写入不适用该类型的来源。
_GLOBAL_ONLY_SOURCES = tuple(s for s in SUPPORTED_NEWS_SOURCES if s != SOURCE_STOCK_NEWS_EM)


def _allowed_sources_for_type(subscription_type):
    if subscription_type == "portfolio":
        return SUPPORTED_NEWS_SOURCES
    return _GLOBAL_ONLY_SOURCES


_SUBSCRIPTION_LOCK = threading.RLock()
_HISTORY_LOCK = threading.RLock()
_STATE_LOCK = threading.RLock()


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _config_path(filename):
    directory = Config.get_config_dir()
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, filename)


def _backup_corrupted_file(path, error):
    """损坏恢复：备份无法解析的存储文件后再重置为空，避免用户数据静默丢失且不可追溯。

    固定使用单一备份文件名（覆盖写入）而非按时间戳生成新文件：损坏文件在被
    修复前可能被反复只读命中（如状态文件每轮轮询都会 can_fetch），若按次
    生成新备份会在长期未写入时无限堆积磁盘文件。
    """
    try:
        if os.path.exists(path):
            backup_path = f"{path}.corrupted.bak"
            shutil.copy2(path, backup_path)
            logger.warning(
                "本地存储文件损坏，已备份至 %s 并重置为空：%s：%s",
                backup_path, path, error,
            )
        else:
            logger.warning("本地存储文件读取失败：%s：%s", path, error)
    except OSError:
        logger.warning("本地存储文件损坏且备份失败，已重置为空：%s：%s", path, error)


def _load_json(path, default, validate=None):
    """读取本地 JSON 存储。

    只有 JSON 语法错误（json.JSONDecodeError）或未通过 `validate` 的
    根结构/嵌套 schema 校验才视为“损坏”：备份后重置为空。文件被占用、
    权限不足等 OSError（包括 open() 抛出的）一律原样抛出——绝不能被当成
    “空数据”返回，否则调用方后续的写入会把还完好的数据静默冲掉。
    """
    if not os.path.exists(path):
        return default()
    with open(path, "r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as exc:
            _backup_corrupted_file(path, exc)
            return default()
    if not isinstance(data, dict) or (validate is not None and not _safe_validate(data, validate)):
        _backup_corrupted_file(path, ValueError("JSON schema 校验失败"))
        return default()
    return data


def _safe_validate(data, validate):
    try:
        return bool(validate(data))
    except Exception:
        return False


def _atomic_write_json(path, data):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            prefix=f".{os.path.basename(path)}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = handle.name
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _string_list(values):
    if isinstance(values, str):
        values = [values]
    result = []
    seen = set()
    for value in values or []:
        if value is None:
            continue
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _symbol_list(values):
    symbols = []
    for value in _string_list(values):
        try:
            symbols.append(to_ts_code(value))
        except ValueError:
            continue
    return sorted(set(symbols))


def _is_string_list(value):
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def _valid_subscription_item(item):
    """订阅深校验：核心字段自建库起就一直存在，缺失即视为损坏；`symbols`
    是 portfolio 类型天然没有的字段，缺失属于正常情况，不参与校验。"""
    if not isinstance(item, dict):
        return False
    if not isinstance(item.get("id"), str) or not item["id"]:
        return False
    if item.get("type") not in SUBSCRIPTION_TYPES:
        return False
    if not isinstance(item.get("name"), str):
        return False
    if not isinstance(item.get("enabled"), bool):
        return False
    sources = item.get("sources")
    if not isinstance(sources, list) or not all(
        isinstance(s, str) and s in SUPPORTED_NEWS_SOURCES for s in sources
    ):
        return False
    if not _is_string_list(item.get("keywords")):
        return False
    if not _is_string_list(item.get("exclude_keywords")):
        return False
    symbols = item.get("symbols")
    if symbols is not None and not _is_string_list(symbols):
        return False
    return True


def _valid_history_item(item):
    """新闻历史深校验：关键字段类型必须正确；聚合类数组字段（related_sources
    等）是后续迭代逐步补齐的，旧文件里可能缺失，缺失时按向后兼容处理。"""
    if not isinstance(item, dict):
        return False
    if not isinstance(item.get("id"), str) or not item["id"]:
        return False
    if not isinstance(item.get("title"), str):
        return False
    if not isinstance(item.get("source"), str):
        return False
    if not isinstance(item.get("read"), bool):
        return False
    if not isinstance(item.get("notification_pending"), bool):
        return False
    for key in (
        "related_sources", "matched_subscription_ids",
        "matched_symbols", "pending_subscription_ids",
    ):
        value = item.get(key)
        if value is not None and not _is_string_list(value):
            return False
    sentiment = item.get("sentiment")
    if sentiment is not None and sentiment not in ("bullish", "bearish", "neutral"):
        return False
    labeled_at = item.get("sentiment_labeled_at")
    if labeled_at is not None and not isinstance(labeled_at, str):
        return False
    return True


def _valid_fetch_state(state):
    """来源/单股抓取退避状态节点：字段均为运行期写入，允许缺失（首次抓取前）。"""
    if not isinstance(state, dict):
        return False
    failure_count = state.get("failure_count", 0)
    if isinstance(failure_count, bool) or not isinstance(failure_count, int):
        return False
    for key in ("next_fetch_at", "last_success", "last_error"):
        value = state.get(key)
        if value is not None and not isinstance(value, str):
            return False
    return True


def _valid_sources_map(value):
    return isinstance(value, dict) and all(
        isinstance(source, str) and _valid_fetch_state(state)
        for source, state in value.items()
    )


def _valid_symbol_sources_map(value):
    if not isinstance(value, dict):
        return False
    for source, symbol_states in value.items():
        if not isinstance(source, str) or not isinstance(symbol_states, dict):
            return False
        for symbol, state in symbol_states.items():
            if not isinstance(symbol, str) or not _valid_fetch_state(state):
                return False
    return True


def _valid_subscription_baselines_map(value):
    if not isinstance(value, dict):
        return False
    for subscription_id, sources in value.items():
        if not isinstance(subscription_id, str) or not isinstance(sources, dict):
            return False
        for source, baseline in sources.items():
            if not isinstance(source, str) or not isinstance(baseline, dict):
                return False
            item_ids = baseline.get("item_ids")
            if item_ids is not None and not _is_string_list(item_ids):
                return False
            updated_at = baseline.get("updated_at")
            if updated_at is not None and not isinstance(updated_at, str):
                return False
    return True


def _valid_symbol_name_cache_map(value):
    if not isinstance(value, dict):
        return False
    for symbol, entry in value.items():
        if not isinstance(symbol, str) or not isinstance(entry, dict):
            return False
        name = entry.get("name")
        if name is not None and not isinstance(name, str):
            return False
        fetched_at = entry.get("fetched_at")
        if fetched_at is not None and not isinstance(fetched_at, str):
            return False
    return True


class NewsSubscriptionStore:
    def __init__(self, file_path=None):
        self.file_path = file_path or _config_path("news_subscriptions.json")

    @staticmethod
    def _empty():
        return {"version": 1, "subscriptions": []}

    @staticmethod
    def _valid(data):
        subscriptions = data.get("subscriptions")
        return isinstance(subscriptions, list) and all(
            _valid_subscription_item(item) for item in subscriptions
        )

    def _load(self):
        return _load_json(self.file_path, self._empty, validate=self._valid)

    def _save(self, data):
        _atomic_write_json(self.file_path, data)

    def list_subscriptions(self, enabled=None, subscription_type=None):
        with _SUBSCRIPTION_LOCK:
            items = [dict(item) for item in self._load()["subscriptions"]]
        if enabled is not None:
            items = [item for item in items if item.get("enabled", True) == bool(enabled)]
        if subscription_type:
            items = [item for item in items if item.get("type") == subscription_type]
        return items

    def get_subscription(self, subscription_id):
        with _SUBSCRIPTION_LOCK:
            for item in self._load()["subscriptions"]:
                if item.get("id") == subscription_id:
                    return dict(item)
        return None

    def create_subscription(
        self,
        subscription_type,
        name,
        keywords=None,
        exclude_keywords=None,
        sources=None,
        enabled=True,
        symbols=None,
    ):
        if subscription_type not in SUBSCRIPTION_TYPES:
            raise ValueError(f"不支持的订阅类型：{subscription_type}")
        if subscription_type == "portfolio" and symbols is not None:
            raise ValueError("portfolio 订阅不支持 symbols 字段")
        now = _now()
        item = {
            "id": str(uuid.uuid4()),
            "type": subscription_type,
            "name": str(name or "").strip() or subscription_type,
            "enabled": bool(enabled),
            "keywords": _string_list(keywords),
            "exclude_keywords": _string_list(exclude_keywords),
            "sources": self._normalize_sources(sources, subscription_type),
            "created_at": now,
            "updated_at": now,
        }
        if subscription_type != "portfolio":
            item["symbols"] = _symbol_list(symbols)
        with _SUBSCRIPTION_LOCK:
            data = self._load()
            data["subscriptions"].append(item)
            self._save(data)
        return dict(item)

    def update_subscription(self, subscription_id, **changes):
        allowed = {
            "name", "enabled", "keywords", "exclude_keywords", "sources", "symbols",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"不支持更新字段：{', '.join(sorted(unknown))}")
        with _SUBSCRIPTION_LOCK:
            data = self._load()
            target = next(
                (item for item in data["subscriptions"] if item.get("id") == subscription_id),
                None,
            )
            if target is None:
                return None
            if "name" in changes:
                target["name"] = str(changes["name"] or "").strip() or target["type"]
            if "enabled" in changes:
                target["enabled"] = bool(changes["enabled"])
            for key in ("keywords", "exclude_keywords"):
                if key in changes:
                    target[key] = _string_list(changes[key])
            if "sources" in changes:
                target["sources"] = self._normalize_sources(
                    changes["sources"], target.get("type")
                )
            if "symbols" in changes and target.get("type") == "portfolio":
                raise ValueError("portfolio 订阅不支持 symbols 字段")
            if "symbols" in changes:
                target["symbols"] = _symbol_list(changes["symbols"])
            target["updated_at"] = _now()
            self._save(data)
            return dict(target)

    def set_enabled(self, subscription_id, enabled):
        return self.update_subscription(subscription_id, enabled=enabled)

    def delete_subscription(self, subscription_id):
        with _SUBSCRIPTION_LOCK:
            data = self._load()
            before = len(data["subscriptions"])
            data["subscriptions"] = [
                item for item in data["subscriptions"]
                if item.get("id") != subscription_id
            ]
            if len(data["subscriptions"]) == before:
                return False
            self._save(data)
            return True

    @staticmethod
    def _normalize_sources(sources, subscription_type):
        """None、空数组或仅含空白值均视为“使用该类型默认来源”。

        与 fin_agent.tools.news_tools._normalize_sources_for_type 保持一致；
        否则空列表会被原样存储，而 _fetch_sources/_match_new_items 里的
        `subscription.get("sources") or SUPPORTED_NEWS_SOURCES` 兜底会在运行期
        把它当作“全部来源”处理，导致 sector/topic 订阅绕过 stock_news_em 限制。
        """
        allowed = _allowed_sources_for_type(subscription_type)
        values = _string_list(sources) if sources is not None else []
        if not values:
            values = list(allowed)
        invalid = set(values) - set(SUPPORTED_NEWS_SOURCES)
        if invalid:
            raise ValueError(f"不支持的新闻源：{', '.join(sorted(invalid))}")
        disallowed = set(values) - set(allowed)
        if disallowed:
            raise ValueError(
                f"{subscription_type} 订阅不支持来源：{', '.join(sorted(disallowed))}"
            )
        return values


class NotifiedNewsStore:
    def __init__(self, file_path=None):
        self.file_path = file_path or _config_path("notified_news.json")

    @staticmethod
    def _empty():
        return {"version": 1, "items": []}

    @staticmethod
    def _valid(data):
        items = data.get("items")
        return isinstance(items, list) and all(
            _valid_history_item(item) for item in items
        )

    def _load(self):
        return _load_json(self.file_path, self._empty, validate=self._valid)

    def upsert(self, news_item, matched_subscription_ids=None, matched_symbols=None):
        stored = self.upsert_many([{
            "item": news_item,
            "matched_subscription_ids": matched_subscription_ids,
            "matched_symbols": matched_symbols,
        }])
        return stored[0]["item"], stored[0]["created"]

    def upsert_many(self, entries):
        """一次锁、一次读取、一次原子替换完成整批新闻合并。"""
        with _HISTORY_LOCK:
            data = self._load()
            by_id = {item.get("id"): item for item in data["items"] if item.get("id")}
            by_fingerprint = {
                item.get("fingerprint"): item
                for item in data["items"]
                if item.get("fingerprint")
                and item.get("fingerprint_version") == 2
            }
            by_title_day = {}
            for item in data["items"]:
                key = item.get("title_day_fingerprint")
                if key and item.get("fingerprint_version") == 2:
                    by_title_day.setdefault(key, []).append(item)
            now = _now()
            results = []
            for entry in entries:
                news_item = entry.get("item")
                incoming = (
                    news_item.to_dict()
                    if hasattr(news_item, "to_dict")
                    else dict(news_item)
                )
                news_id = incoming.get("id")
                if not news_id:
                    raise ValueError("新闻缺少 id")
                fingerprint = incoming.get("fingerprint")
                title_day_fingerprint = incoming.get("title_day_fingerprint")
                existing = by_id.get(news_id)
                if (
                    existing is None
                    and fingerprint
                    and incoming.get("fingerprint_version") == 2
                ):
                    existing = by_fingerprint.get(fingerprint)
                if existing is None and title_day_fingerprint:
                    existing = next((
                        candidate
                        for candidate in by_title_day.get(
                            title_day_fingerprint, []
                        )
                        if _summaries_compatible(
                            candidate.get("summary"),
                            incoming.get("summary"),
                        )
                    ), None)
                created = existing is None
                if existing is None:
                    existing = incoming
                    existing["read"] = False
                    existing["notification_pending"] = True
                    existing["pending_subscription_ids"] = []
                    existing["notified_at"] = None
                    existing["matched_subscription_ids"] = []
                    existing["matched_symbols"] = []
                    existing["related_sources"] = []
                    existing["source_alias_ids"] = []
                    data["items"].append(existing)
                    by_id[news_id] = existing
                    if fingerprint and incoming.get("fingerprint_version") == 2:
                        by_fingerprint[fingerprint] = existing
                    if (
                        title_day_fingerprint
                        and incoming.get("fingerprint_version") == 2
                    ):
                        by_title_day.setdefault(
                            title_day_fingerprint, []
                        ).append(existing)
                else:
                    if (
                        fingerprint
                        and incoming.get("fingerprint_version") == 2
                        and existing.get("fingerprint_version") != 2
                    ):
                        existing["fingerprint"] = fingerprint
                        existing["title_day_fingerprint"] = title_day_fingerprint
                        existing["fingerprint_version"] = 2
                        by_fingerprint[fingerprint] = existing
                        if title_day_fingerprint:
                            by_title_day.setdefault(
                                title_day_fingerprint, []
                            ).append(existing)
                    existing["summary"] = existing.get("summary") or incoming.get("summary", "")
                    existing["url"] = existing.get("url") or incoming.get("url", "")
                    existing["published_at"] = (
                        existing.get("published_at") or incoming.get("published_at", "")
                    )
                    existing["symbols"] = sorted(set(
                        _string_list(existing.get("symbols"))
                        + _string_list(incoming.get("symbols"))
                    ))
                needs_notification = bool(
                    created or existing.get("notification_pending", False)
                )
                existing["related_sources"] = sorted(set(
                    _string_list(existing.get("related_sources"))
                    + [existing.get("source"), incoming.get("source")]
                ))
                existing["source_alias_ids"] = sorted(set(
                    _string_list(existing.get("source_alias_ids"))
                    + [existing.get("id"), news_id]
                ))
                existing["matched_subscription_ids"] = sorted(set(
                    _string_list(existing.get("matched_subscription_ids"))
                    + _string_list(entry.get("matched_subscription_ids"))
                ))
                existing["matched_symbols"] = sorted(set(
                    _string_list(existing.get("matched_symbols"))
                    + _string_list(entry.get("matched_symbols"))
                ))
                if needs_notification:
                    existing["notification_pending"] = True
                    existing["pending_subscription_ids"] = sorted(set(
                        _string_list(existing.get("pending_subscription_ids"))
                        + _string_list(entry.get("matched_subscription_ids"))
                    ))
                existing["updated_at"] = now
                results.append({
                    "item": dict(existing),
                    "created": created,
                    "needs_notification": needs_notification,
                })
            data["items"].sort(
                key=lambda item: (item.get("published_at", ""), item.get("id", "")),
                reverse=True,
            )
            self._save(data)
            return results

    def list_pending_notifications(self, limit=200):
        with _HISTORY_LOCK:
            items = [
                dict(item) for item in self._load()["items"]
                if item.get("notification_pending", False)
            ]
        return items[:max(int(limit), 0)]

    def list_unlabeled_sentiment(self, limit=50):
        with _HISTORY_LOCK:
            items = [
                dict(item) for item in self._load()["items"]
                if item.get("sentiment") not in ("bullish", "bearish", "neutral")
            ]
        return items[:max(int(limit), 0)]

    def update_sentiment(self, news_id, sentiment):
        if sentiment not in ("bullish", "bearish", "neutral"):
            return False
        with _HISTORY_LOCK:
            data = self._load()
            changed = False
            now = _now()
            for item in data["items"]:
                if item.get("id") != news_id:
                    continue
                item["sentiment"] = sentiment
                item["sentiment_labeled_at"] = now
                item["updated_at"] = now
                changed = True
                break
            if changed:
                self._save(data)
            return changed

    def mark_notifications_dispatched(self, news_ids):
        news_ids = set(_string_list(news_ids))
        if not news_ids:
            return 0
        with _HISTORY_LOCK:
            data = self._load()
            changed = 0
            now = _now()
            for item in data["items"]:
                if item.get("id") not in news_ids:
                    continue
                if item.get("notification_pending", False):
                    item["notification_pending"] = False
                    item["pending_subscription_ids"] = []
                    item["notified_at"] = now
                    item["updated_at"] = now
                    changed += 1
            if changed:
                self._save(data)
            return changed

    append_or_upsert = upsert

    def list_news(
        self,
        source=None,
        unread_only=False,
        subscription_id=None,
        subscription_type=None,
        query=None,
        symbol=None,
        news_id=None,
        offset=0,
        limit=100,
    ):
        with _HISTORY_LOCK:
            items = [dict(item) for item in self._load()["items"]]
        if news_id:
            items = [item for item in items if item.get("id") == news_id]
        if source:
            items = [
                item for item in items
                if item.get("source") == source
                or source in item.get("related_sources", [])
            ]
        if unread_only:
            items = [item for item in items if not item.get("read", False)]
        if subscription_id:
            items = [
                item for item in items
                if subscription_id in item.get("matched_subscription_ids", [])
            ]
        if subscription_type:
            subscription_ids = {
                item.get("id")
                for item in NewsSubscriptionStore().list_subscriptions(
                    subscription_type=subscription_type
                )
            }
            items = [
                item for item in items
                if subscription_ids.intersection(
                    item.get("matched_subscription_ids", [])
                )
            ]
        if query:
            needle = str(query).strip().casefold()
            items = [
                item for item in items
                if needle in "\n".join((
                    str(item.get("title", "")),
                    str(item.get("summary", "")),
                    str(item.get("source", "")),
                    " ".join(item.get("symbols", [])),
                )).casefold()
            ]
        if symbol:
            normalized_symbol = _symbol_list([symbol])
            symbol = normalized_symbol[0] if normalized_symbol else str(symbol)
            items = [
                item for item in items
                if symbol in item.get("matched_symbols", [])
            ]
        offset = max(int(offset or 0), 0)
        limit = max(int(limit or 0), 0)
        return {"items": items[offset:offset + limit], "total": len(items)}

    def mark_read(self, news_id, read=True):
        with _HISTORY_LOCK:
            data = self._load()
            target = next(
                (item for item in data["items"] if item.get("id") == news_id),
                None,
            )
            if target is None:
                return False
            if target.get("read", False) != bool(read):
                target["read"] = bool(read)
                target["updated_at"] = _now()
                self._save(data)
            return True

    def mark_read_many(self, news_ids, read=True):
        news_ids = set(_string_list(news_ids))
        if not news_ids:
            return 0
        with _HISTORY_LOCK:
            data = self._load()
            changed = 0
            now = _now()
            for item in data["items"]:
                if item.get("id") not in news_ids:
                    continue
                if item.get("read", False) != bool(read):
                    item["read"] = bool(read)
                    item["updated_at"] = now
                    changed += 1
            if changed:
                self._save(data)
            return changed

    def mark_all_read(self):
        with _HISTORY_LOCK:
            data = self._load()
            changed = 0
            now = _now()
            for item in data["items"]:
                if not item.get("read", False):
                    item["read"] = True
                    item["updated_at"] = now
                    changed += 1
            if changed:
                self._save(data)
            return changed

    def clear(self):
        with _HISTORY_LOCK:
            count = len(self._load()["items"])
            self._save(self._empty())
            return count

    def unread_count(self):
        with _HISTORY_LOCK:
            return sum(
                1 for item in self._load()["items"]
                if not item.get("read", False)
            )

    def cleanup(self, retention_days=30):
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            days=max(int(retention_days), 0)
        )
        with _HISTORY_LOCK:
            data = self._load()
            kept = []
            for item in data["items"]:
                if item.get("notification_pending", False):
                    kept.append(item)
                    continue
                timestamp = _parse_datetime(
                    item.get("published_at") or item.get("notified_at")
                )
                if timestamp is None or timestamp >= cutoff:
                    kept.append(item)
            removed = len(data["items"]) - len(kept)
            if removed:
                data["items"] = kept
                self._save(data)
            return removed

    def _save(self, data):
        _atomic_write_json(self.file_path, data)


class NewsMonitorStateStore:
    BASELINE_LIMIT = 2000

    def __init__(self, file_path=None):
        self.file_path = file_path or _config_path("news_monitor_state.json")

    @staticmethod
    def _empty():
        return {
            "version": 2,
            "sources": {},
            "symbol_sources": {},
            "subscription_baselines": {},
            "symbol_name_cache": {},
        }

    @staticmethod
    def _valid(data):
        # 四个顶层键在缺失时一律按“旧文件/尚未写入过”处理（返回空字典兜底），
        # 不因为缺少某个较晚版本才引入的键就判为损坏；_load() 之后会 setdefault。
        return (
            _valid_sources_map(data.get("sources", {}))
            and _valid_symbol_sources_map(data.get("symbol_sources", {}))
            and _valid_subscription_baselines_map(data.get("subscription_baselines", {}))
            and _valid_symbol_name_cache_map(data.get("symbol_name_cache", {}))
        )

    def _load(self):
        data = _load_json(self.file_path, self._empty, validate=self._valid)
        data["version"] = 2
        data.setdefault("sources", {})
        data.setdefault("symbol_sources", {})
        data.setdefault("subscription_baselines", {})
        data.setdefault("symbol_name_cache", {})
        for state in data["sources"].values():
            state.pop("baseline_item_ids", None)
        for source_states in data["symbol_sources"].values():
            for state in source_states.values():
                state.pop("baseline_item_ids", None)
        return data

    def can_fetch(self, source, now=None):
        now = now or datetime.datetime.now(datetime.timezone.utc)
        with _STATE_LOCK:
            state = self._load()["sources"].get(source, {})
        next_fetch = _parse_datetime(state.get("next_fetch_at"))
        return next_fetch is None or now >= next_fetch

    def can_fetch_symbol(self, source, symbol, now=None):
        now = now or datetime.datetime.now(datetime.timezone.utc)
        with _STATE_LOCK:
            state = (
                self._load()["symbol_sources"]
                .get(source, {})
                .get(symbol, {})
            )
        next_fetch = _parse_datetime(state.get("next_fetch_at"))
        return next_fetch is None or now >= next_fetch

    def get_source_health(self):
        """来源/单股抓取退避状态快照，供 monitor status 暴露给 UI 观察。

        只返回有失败记录或仍处于退避期的条目，避免持仓量大的账户让状态膨胀。
        """
        with _STATE_LOCK:
            data = self._load()
            sources = {
                source: _health_entry(state)
                for source, state in data["sources"].items()
                if _is_unhealthy(state)
            }
            symbol_sources = {}
            for source, symbol_states in data["symbol_sources"].items():
                failing = {
                    symbol: _health_entry(state)
                    for symbol, state in symbol_states.items()
                    if _is_unhealthy(state)
                }
                if failing:
                    symbol_sources[source] = failing
            return {"sources": sources, "symbol_sources": symbol_sources}

    def get_symbol_name_cache(self, symbols=None, max_age_hours=24):
        """返回持仓 symbol->name 缓存；给定 `symbols` 时一并返回其中缺失/过期的子集。

        供全局快讯的公司名匹配使用。名称来自行情接口，一天刷新一次；
        `stale` 为空时调用方应直接沿用已有缓存，不必发起网络请求。
        """
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=max_age_hours)
        with _STATE_LOCK:
            cache = self._load()["symbol_name_cache"]
            names = {
                symbol: entry.get("name")
                for symbol, entry in cache.items()
                if isinstance(entry, dict) and entry.get("name")
            }
            stale = []
            if symbols is not None:
                for symbol in symbols:
                    entry = cache.get(symbol)
                    if not isinstance(entry, dict) or not entry.get("name"):
                        stale.append(symbol)
                        continue
                    fetched_at = _parse_datetime(entry.get("fetched_at"))
                    if fetched_at is None or fetched_at < cutoff:
                        stale.append(symbol)
            return names, stale

    def update_symbol_name_cache(self, updates):
        """合并写入 symbol->name 缓存，记录 fetched_at 供下次判断是否过期。"""
        if not updates:
            return
        with _STATE_LOCK:
            data = self._load()
            cache = data["symbol_name_cache"]
            now = _now()
            for symbol, name in updates.items():
                if not symbol or not name:
                    continue
                cache[symbol] = {"name": name, "fetched_at": now}
            _atomic_write_json(self.file_path, data)

    def apply_fetch_status(
        self,
        source_successes=None,
        source_failures=None,
        symbol_successes=None,
        symbol_failures=None,
        base_interval_minutes=5,
    ):
        """批量记录来源及单股抓取状态，只写一次状态文件。"""
        with _STATE_LOCK:
            data = self._load()
            for source in (source_successes or {}):
                state = data["sources"].setdefault(source, {})
                state.update({
                    "failure_count": 0,
                    "next_fetch_at": None,
                    "last_success": _now(),
                    "last_error": None,
                })
            for source, error in (source_failures or {}).items():
                self._apply_failure(
                    data["sources"].setdefault(source, {}),
                    error,
                    base_interval_minutes,
                )
            symbol_states = data["symbol_sources"]
            for key in (symbol_successes or {}):
                source, symbol = key
                state = symbol_states.setdefault(source, {}).setdefault(symbol, {})
                state.update({
                    "failure_count": 0,
                    "next_fetch_at": None,
                    "last_success": _now(),
                    "last_error": None,
                })
            for key, error in (symbol_failures or {}).items():
                source, symbol = key
                state = symbol_states.setdefault(source, {}).setdefault(symbol, {})
                self._apply_failure(state, error, base_interval_minutes)
            _atomic_write_json(self.file_path, data)

    def get_subscription_baselines(self):
        with _STATE_LOCK:
            data = self._load()
            return {
                subscription_id: {
                    source: set(baseline.get("item_ids", []))
                    for source, baseline in sources.items()
                }
                for subscription_id, sources
                in data["subscription_baselines"].items()
            }

    def merge_subscription_baselines(self, updates):
        """按订阅/来源做有界历史并集，避免短返回使基线缩水。"""
        with _STATE_LOCK:
            data = self._load()
            subscriptions = data["subscription_baselines"]
            for update in updates:
                subscription_id = update["subscription_id"]
                source = update["source"]
                baseline = subscriptions.setdefault(
                    subscription_id, {}
                ).setdefault(source, {})
                baseline["item_ids"] = self._bounded_union(
                    baseline.get("item_ids"), update.get("item_ids"),
                )
                baseline["updated_at"] = _now()
            _atomic_write_json(self.file_path, data)

    def prune_subscriptions(self, active_subscription_ids):
        active = set(active_subscription_ids)
        with _STATE_LOCK:
            data = self._load()
            baselines = data["subscription_baselines"]
            stale = [subscription_id for subscription_id in baselines if subscription_id not in active]
            if not stale:
                return 0
            for subscription_id in stale:
                del baselines[subscription_id]
            _atomic_write_json(self.file_path, data)
            return len(stale)

    @classmethod
    def _bounded_union(cls, old_values, new_values):
        ordered = {}
        for value in _string_list(old_values) + _string_list(new_values):
            ordered.pop(value, None)
            ordered[value] = None
        return list(ordered.keys())[-cls.BASELINE_LIMIT:]

    @staticmethod
    def _apply_failure(state, error, base_interval_minutes):
        failures = int(state.get("failure_count", 0)) + 1
        delay_minutes = min(
            max(int(base_interval_minutes), 1) * (2 ** min(failures - 1, 10)),
            360,
        )
        next_fetch = (
            datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(minutes=delay_minutes)
        )
        state.update({
            "failure_count": failures,
            "next_fetch_at": next_fetch.isoformat(timespec="seconds"),
            "last_error": str(error)[:1000],
        })


def _is_unhealthy(state):
    return bool(state.get("failure_count", 0)) or bool(state.get("next_fetch_at"))


def _health_entry(state):
    return {
        "failure_count": state.get("failure_count", 0),
        "next_fetch_at": state.get("next_fetch_at"),
        "last_success": state.get("last_success"),
        "last_error": state.get("last_error"),
    }


def _summaries_compatible(left, right):
    left = re.sub(r"[\W_]+", "", str(left or "").casefold())[:300]
    right = re.sub(r"[\W_]+", "", str(right or "").casefold())[:300]
    if not left or not right:
        return True
    return difflib.SequenceMatcher(None, left, right).ratio() >= 0.45


def _parse_datetime(value):
    if not value:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed.astimezone(datetime.timezone.utc)
