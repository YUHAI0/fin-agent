"""不受交易时段限制的独立新闻轮询守护线程。"""
import datetime
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait

from fin_agent.config import Config
from fin_agent.news import (
    SOURCE_STOCK_NEWS_EM,
    SUPPORTED_NEWS_SOURCES,
    AkshareNewsAdapter,
)
from fin_agent.news_store import (
    NewsMonitorStateStore,
    NewsSubscriptionStore,
    NotifiedNewsStore,
)
from fin_agent.portfolio import PortfolioManager


logger = logging.getLogger(__name__)
_MONITOR_LOCK = threading.Lock()
_MONITOR_INSTANCE = None

# 常见公司后缀，用于把行情接口返回的正式名称规范化为快讯里常见的简称，
# 例如“贵州茅台酒股份有限公司” -> “贵州茅台”。
_COMPANY_NAME_SUFFIXES = (
    "股份有限公司", "集团股份有限公司", "控股集团有限公司", "有限责任公司",
    "集团有限公司", "控股有限公司", "股份公司", "集团股份", "有限公司",
    "控股集团", "科技集团", "集团", "控股", "股份",
)
# 规范化后短于该长度的“简称”不参与全局快讯匹配，避免常见短词误报
# （例如两字简称很容易和无关报道里的常见词撞车）。
_MIN_COMPANY_NAME_LENGTH = 3
_WHITESPACE_RE = re.compile(r"[\s\u3000]+")


def _normalize_company_text(text):
    return _WHITESPACE_RE.sub("", text or "")


def _normalized_company_name(name):
    """把公司全称规范化为匹配用简称；过短或空则返回空字符串表示不可用。"""
    text = _normalize_company_text(name)
    if not text:
        return ""
    for suffix in _COMPANY_NAME_SUFFIXES:
        if text.endswith(suffix) and len(text) > len(suffix):
            text = text[: -len(suffix)]
            break
    if len(text) < _MIN_COMPANY_NAME_LENGTH:
        return ""
    return text


def _noop_notification_sink(payload):
    return None


class NewsMonitor:
    def __init__(
        self,
        notification_sink=None,
        adapter=None,
        subscription_store=None,
        history_store=None,
        state_store=None,
        fetch_timeout_seconds=20,
        fetch_workers=8,
        max_inflight_fetches=64,
        sink_timeout_seconds=10,
        sink_queue_limit=8,
    ):
        self.notification_sink = notification_sink or _noop_notification_sink
        self.adapter = adapter or AkshareNewsAdapter()
        self.subscription_store = subscription_store or NewsSubscriptionStore()
        self.history_store = history_store or NotifiedNewsStore()
        self.state_store = state_store or NewsMonitorStateStore()
        self.fetch_timeout_seconds = max(float(fetch_timeout_seconds), 1.0)
        self.sink_timeout_seconds = max(float(sink_timeout_seconds), 1.0)
        self._stop_event = threading.Event()
        self._run_lock = threading.Lock()
        self._thread = None
        self._fetch_executor = ThreadPoolExecutor(
            max_workers=max(int(fetch_workers), 1),
            thread_name_prefix="news-fetch",
        )
        self._max_inflight_fetches = max(int(max_inflight_fetches), 1)
        self._fetch_inflight = {}
        self._symbol_cursor = 0
        self._sink_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="news-notification",
        )
        self._sink_slots = threading.BoundedSemaphore(
            max(int(sink_queue_limit), 1)
        )
        self._sink_tasks = []
        self._sink_lock = threading.Lock()
        self._sink_item_ids = set()
        self._closed = False
        self._close_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._cycle_running = False
        self._last_started_at = None
        self._last_completed_at = None
        self._last_error = None
        self._refresh_thread = None

    def start(self):
        if self._closed:
            raise RuntimeError("已关闭的 NewsMonitor 不能重新启动")
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self.run_forever,
            name="fin-agent-news-monitor",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self):
        self._stop_event.set()

    def shutdown(self, timeout=None):
        """停止调度并有限等待当前轮询，executor 只做非阻塞收尾。"""
        self.stop()
        timeout = (
            self.fetch_timeout_seconds + 2
            if timeout is None
            else max(float(timeout), 0)
        )
        deadline = time.monotonic() + timeout
        current = threading.current_thread()
        for thread in (self._thread, self._refresh_thread):
            if thread is None or thread is current or not thread.is_alive():
                continue
            thread.join(timeout=max(deadline - time.monotonic(), 0))
        self.close()

    def status(self):
        Config.load()
        with self._status_lock:
            return {
                "running": bool(
                    self._thread is not None
                    and self._thread.is_alive()
                    and not self._stop_event.is_set()
                ),
                "cycle_running": self._cycle_running,
                "closed": self._closed,
                "poll_interval_minutes": Config.NEWS_POLL_INTERVAL_MINUTES,
                "last_started_at": self._last_started_at,
                "last_completed_at": self._last_completed_at,
                "last_error": self._last_error,
                "source_health": self.state_store.get_source_health(),
            }

    def refresh(self):
        with self._status_lock:
            if self._closed or self._stop_event.is_set():
                raise RuntimeError("新闻监控已停止或正在关闭")
            if self._cycle_running or (
                self._refresh_thread is not None
                and self._refresh_thread.is_alive()
            ):
                return False
            self._refresh_thread = threading.Thread(
                target=self._run_manual_refresh,
                name="fin-agent-news-refresh",
                daemon=True,
            )
            self._refresh_thread.start()
            return True

    def _run_manual_refresh(self):
        try:
            self.run_once()
        except Exception:
            logger.exception("Manual news refresh failed")

    def run_forever(self):
        timeout = 0
        while not self._stop_event.wait(timeout):
            try:
                self.run_once()
            except Exception:
                logger.exception("News monitor cycle failed")
            Config.load()
            timeout = Config.NEWS_POLL_INTERVAL_MINUTES * 60

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._fetch_executor.shutdown(wait=False, cancel_futures=True)
            self._sink_executor.shutdown(wait=False, cancel_futures=True)

    def run_once(self):
        if not self._run_lock.acquire(blocking=False):
            return None
        with self._status_lock:
            self._cycle_running = True
            self._last_started_at = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(timespec="seconds")
            self._last_error = None
        try:
            Config.load()
            self._reap_sink_tasks()
            from fin_agent.news_sentiment import get_sentiment_labeler
            get_sentiment_labeler().enqueue_backlog(self.history_store, limit=3)
            all_subscriptions = self.subscription_store.list_subscriptions()
            retry_payload = self._dispatch_pending_notifications(
                all_subscriptions
            )
            self.history_store.cleanup(retention_days=30)
            self.state_store.prune_subscriptions(
                subscription.get("id") for subscription in all_subscriptions
                if subscription.get("id")
            )
            subscriptions = [
                subscription for subscription in all_subscriptions
                if subscription.get("enabled", True)
            ]
            if not subscriptions:
                return retry_payload

            portfolio_symbols = self._portfolio_symbols()
            symbol_names = self._refresh_symbol_name_cache(portfolio_symbols)
            fetched, fetch_status = self._fetch_sources(
                subscriptions, portfolio_symbols,
            )
            if any(fetch_status.values()):
                self.state_store.apply_fetch_status(
                    **fetch_status,
                    base_interval_minutes=Config.NEWS_POLL_INTERVAL_MINUTES,
                )
            pending, baseline_updates = self._match_new_items(
                subscriptions,
                fetched,
                portfolio_symbols,
                {
                    key[1]
                    for key in fetch_status["symbol_successes"]
                    if key[0] == SOURCE_STOCK_NEWS_EM
                },
                symbol_names,
            )
            if not pending:
                if baseline_updates:
                    self.state_store.merge_subscription_baselines(
                        baseline_updates
                    )
                return retry_payload

            entries = [{
                "item": match["item"],
                "matched_subscription_ids": match["subscription_ids"],
                "matched_symbols": match["matched_symbols"],
            } for match in pending.values()]
            try:
                upsert_results = self.history_store.upsert_many(entries)
                get_sentiment_labeler().enqueue_from_upsert_results(
                    upsert_results, self.history_store,
                )
                self.state_store.merge_subscription_baselines(baseline_updates)
            except Exception:
                raise
            return (
                self._dispatch_pending_notifications(all_subscriptions)
                or retry_payload
            )
        except Exception as exc:
            with self._status_lock:
                self._last_error = str(exc)
            raise
        finally:
            with self._status_lock:
                self._cycle_running = False
                self._last_completed_at = datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(timespec="seconds")
            self._run_lock.release()

    def _fetch_sources(self, subscriptions, portfolio_symbols):
        requested_sources = set()
        stock_symbols = set()
        for subscription in subscriptions:
            sources = subscription.get("sources") or SUPPORTED_NEWS_SOURCES
            requested_sources.update(sources)
            if SOURCE_STOCK_NEWS_EM not in sources:
                continue
            if subscription.get("type") == "portfolio":
                stock_symbols.update(portfolio_symbols)
            else:
                stock_symbols.update(subscription.get("symbols") or [])

        desired = [
            (source, None)
            for source in SUPPORTED_NEWS_SOURCES
            if source != SOURCE_STOCK_NEWS_EM
            and source in requested_sources
            and self.state_store.can_fetch(source)
        ]
        if SOURCE_STOCK_NEWS_EM in requested_sources:
            symbols = sorted(stock_symbols)
            if symbols:
                offset = self._symbol_cursor % len(symbols)
                symbols = symbols[offset:] + symbols[:offset]
                self._symbol_cursor = (offset + 1) % len(symbols)
            desired.extend(
                (SOURCE_STOCK_NEWS_EM, symbol)
                for symbol in symbols
                if self.state_store.can_fetch_symbol(
                    SOURCE_STOCK_NEWS_EM, symbol,
                )
            )

        capacity = self._max_inflight_fetches - len(self._fetch_inflight)
        submitted = []
        for key in desired:
            if capacity <= 0:
                break
            if key in self._fetch_inflight:
                continue
            meta = {
                "future": None,
                "submitted_at": time.monotonic(),
                "started_at": None,
                "timeout_logged": False,
            }
            future = self._fetch_executor.submit(self._execute_fetch, key, meta)
            meta["future"] = future
            self._fetch_inflight[key] = meta
            submitted.append(future)
            capacity -= 1

        if submitted:
            wait(submitted, timeout=self.fetch_timeout_seconds)

        fetched = {}
        source_successes = {}
        source_failures = {}
        symbol_successes = {}
        symbol_failures = {}
        now = time.monotonic()
        for key, meta in list(self._fetch_inflight.items()):
            future = meta["future"]
            source, symbol = key
            if future.done():
                del self._fetch_inflight[key]
                try:
                    items = future.result()
                except Exception as exc:
                    if meta["timeout_logged"]:
                        logger.warning(
                            "Timed-out news fetch %s/%s later failed: %s",
                            source, symbol or "global", exc,
                        )
                        continue
                    if symbol is None:
                        source_failures[source] = exc
                        logger.warning("News source %s failed: %s", source, exc)
                    else:
                        symbol_failures[key] = exc
                        logger.warning(
                            "News source %s symbol %s failed: %s",
                            source, symbol, exc,
                        )
                    continue
                fetched.setdefault(source, []).extend(items)
                if symbol is None:
                    source_successes[source] = True
                else:
                    symbol_successes[key] = True
                    source_successes[source] = True
                continue

            started_at = meta.get("started_at")
            if (
                started_at is not None
                and now - started_at >= self.fetch_timeout_seconds
                and not meta["timeout_logged"]
            ):
                meta["timeout_logged"] = True
                error = TimeoutError(
                    f"{source}/{symbol or 'global'} 抓取超过 "
                    f"{self.fetch_timeout_seconds:g} 秒，保留在途任务"
                )
                if symbol is None:
                    source_failures[source] = error
                else:
                    symbol_failures[key] = error
                logger.warning("%s", error)

        for source, items in fetched.items():
            by_id = {}
            for item in items:
                existing = by_id.get(item.id)
                if existing is None:
                    by_id[item.id] = item
                else:
                    existing.symbols = sorted(
                        set(existing.symbols) | set(item.symbols)
                    )
            fetched[source] = list(by_id.values())
        return fetched, {
            "source_successes": source_successes,
            "source_failures": source_failures,
            "symbol_successes": symbol_successes,
            "symbol_failures": symbol_failures,
        }

    def _execute_fetch(self, key, meta):
        meta["started_at"] = time.monotonic()
        source, symbol = key
        if symbol is not None:
            return self.adapter.fetch_stock_symbol(symbol)
        return self.adapter.fetch(source)

    def _match_new_items(
        self,
        subscriptions,
        fetched,
        portfolio_symbols,
        completed_stock_symbols,
        symbol_names=None,
    ):
        symbol_names = symbol_names or {}
        pending = {}
        baseline_updates = []
        baselines = self.state_store.get_subscription_baselines()
        for subscription in subscriptions:
            subscription_id = subscription.get("id")
            if not subscription_id:
                continue
            for source in subscription.get("sources") or SUPPORTED_NEWS_SOURCES:
                if source not in fetched:
                    continue
                subscription_baselines = baselines.get(subscription_id, {})
                batches = self._subscription_source_batches(
                    subscription,
                    source,
                    fetched[source],
                    portfolio_symbols,
                    completed_stock_symbols,
                )
                for baseline_source, items in batches:
                    current_ids = [item.id for item in items]
                    baseline_updates.append({
                        "subscription_id": subscription_id,
                        "source": baseline_source,
                        "item_ids": current_ids,
                    })
                    if baseline_source not in subscription_baselines:
                        continue
                    baseline = subscription_baselines[baseline_source]
                    for item in items:
                        if item.id in baseline:
                            continue
                        matched_symbols = self._matched_symbols(
                            subscription, item, portfolio_symbols, symbol_names,
                        )
                        if not self._matches(
                            subscription, item, portfolio_symbols, matched_symbols,
                        ):
                            continue
                        match = pending.setdefault(item.id, {
                            "item": item,
                            "subscription_ids": set(),
                            "matched_symbols": set(),
                        })
                        match["subscription_ids"].add(subscription_id)
                        match["matched_symbols"].update(matched_symbols)
        return pending, baseline_updates

    @staticmethod
    def _subscription_source_batches(
        subscription, source, items, portfolio_symbols, completed_stock_symbols,
    ):
        if source != SOURCE_STOCK_NEWS_EM:
            return [(source, items)]
        targets = (
            set(portfolio_symbols)
            if subscription.get("type") == "portfolio"
            else set(subscription.get("symbols") or [])
        )
        grouped = {}
        for symbol in targets & set(completed_stock_symbols):
            grouped[symbol] = []
        for item in items:
            for symbol in set(item.symbols) & targets:
                grouped.setdefault(symbol, []).append(item)
        return [
            (f"{source}:{symbol}", symbol_items)
            for symbol, symbol_items in grouped.items()
        ]

    @staticmethod
    def _matches(subscription, item, portfolio_symbols, matched_symbols):
        haystack = f"{item.title}\n{item.summary}".casefold()
        excluded = [
            keyword.casefold()
            for keyword in subscription.get("exclude_keywords") or []
            if keyword
        ]
        if any(keyword in haystack for keyword in excluded):
            return False

        keywords = [
            keyword.casefold()
            for keyword in subscription.get("keywords") or []
            if keyword
        ]
        keyword_match = any(keyword in haystack for keyword in keywords)
        subscription_type = subscription.get("type")
        if subscription_type == "portfolio":
            if not matched_symbols:
                return False
            return keyword_match if keywords else True

        configured_symbols = set(subscription.get("symbols") or [])
        symbol_match = bool(configured_symbols & set(item.symbols))
        if not keywords and not configured_symbols:
            return False
        return keyword_match or symbol_match

    @staticmethod
    def _matched_symbols(subscription, item, portfolio_symbols, symbol_names=None):
        if subscription.get("type") == "portfolio":
            candidates = set(portfolio_symbols)
        else:
            candidates = set(subscription.get("symbols") or [])
        matched = set(item.symbols) & candidates
        # 全球快讯（非 stock_news_em）不含明确代码上下文，须补充公司名匹配；
        # 仅对持仓订阅生效，且只用规范化后的简称，避免短词模糊误报。
        if (
            symbol_names
            and subscription.get("type") == "portfolio"
            and item.source != SOURCE_STOCK_NEWS_EM
        ):
            haystack = _normalize_company_text(f"{item.title}\n{item.summary}")
            for symbol in candidates - matched:
                normalized_name = _normalized_company_name(symbol_names.get(symbol))
                if normalized_name and normalized_name in haystack:
                    matched.add(symbol)
        return sorted(matched)

    def _refresh_symbol_name_cache(self, portfolio_symbols):
        """刷新持仓 symbol->name 缓存（一天一次），供全局快讯公司名匹配使用。

        失败时沿用旧缓存，不影响 stock_news_em 等按代码抓取的个股新闻源。
        """
        names, stale = self.state_store.get_symbol_name_cache(
            portfolio_symbols, max_age_hours=24,
        )
        if not stale:
            return names
        try:
            from fin_agent.datasources import get_provider

            frame = get_provider().get_realtime_price(stale)
        except Exception as exc:
            logger.warning(
                "刷新持仓名称缓存失败，沿用旧缓存（待刷新 %d 个）：%s",
                len(stale), exc,
            )
            return names
        updates = {}
        if frame is not None and not getattr(frame, "empty", True):
            for _, row in frame.iterrows():
                ts_code = row.get("ts_code")
                name = row.get("name")
                if ts_code and name and str(name).strip():
                    updates[ts_code] = str(name).strip()
        if updates:
            self.state_store.update_symbol_name_cache(updates)
            names = dict(names)
            names.update(updates)
        return names

    @staticmethod
    def _portfolio_symbols():
        manager = PortfolioManager()
        symbols = set()
        for portfolio in manager.data.get("portfolios", {}).values():
            symbols.update((portfolio.get("positions") or {}).keys())
        return sorted(symbols)

    @staticmethod
    def _build_payload(subscriptions, items):
        by_subscription = {
            subscription["id"]: {
                "subscription_id": subscription["id"],
                "subscription_name": subscription.get("name", ""),
                "subscription_type": subscription.get("type"),
                "items": [],
            }
            for subscription in subscriptions
        }
        for item in items:
            for subscription_id in item.get(
                "_notification_subscription_ids", []
            ):
                group = by_subscription.get(subscription_id)
                if group is not None:
                    payload_item = dict(item)
                    payload_item.pop("_notification_subscription_ids", None)
                    group["items"].append(payload_item)
        groups = [group for group in by_subscription.values() if group["items"]]
        for group in groups:
            group["items"].sort(
                key=lambda item: (item.get("published_at", ""), item.get("id", "")),
                reverse=True,
            )
        grouped_ids = {
            item["id"]
            for group in groups
            for item in group["items"]
        }
        ungrouped = []
        for item in items:
            if item.get("id") in grouped_ids:
                continue
            payload_item = dict(item)
            payload_item.pop("_notification_subscription_ids", None)
            ungrouped.append(payload_item)
        if ungrouped:
            groups.append({
                "subscription_id": None,
                "subscription_name": "已删除或停用的订阅",
                "subscription_type": "unknown",
                "items": ungrouped,
            })
        return {
            "type": "news_digest",
            "created_at": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(timespec="seconds"),
            "item_count": len(items),
            "groups": groups,
        }

    def _dispatch_pending_notifications(self, subscriptions):
        with self._sink_lock:
            inflight_ids = set(self._sink_item_ids)
        items = [
            item for item in self.history_store.list_pending_notifications()
            if item.get("id") not in inflight_ids
        ]
        if not items:
            return None
        if not self._sink_slots.acquire(blocking=False):
            logger.warning(
                "News notification queue backlog: pending=%d, in_flight=%d, "
                "no dispatch slot available",
                len(items), len(inflight_ids),
            )
            return None

        item_ids = [item["id"] for item in items]
        for item in items:
            item["_notification_subscription_ids"] = item.get(
                "pending_subscription_ids", []
            )
        payload = self._build_payload(subscriptions, items)
        with self._sink_lock:
            self._sink_item_ids.update(item_ids)
        try:
            self._dispatch_notification(payload, item_ids)
        except Exception:
            with self._sink_lock:
                self._sink_item_ids.difference_update(item_ids)
            self._sink_slots.release()
            logger.exception(
                "Failed to submit news notification; pending state retained"
            )
            return None
        return payload

    def _dispatch_notification(self, payload, item_ids):
        backlog = sum(
            1 for task in self._sink_tasks
            if not task["future"].done()
        )
        if backlog:
            logger.info(
                "Queueing news notification behind %d dispatch task(s)",
                backlog,
            )
        try:
            future = self._sink_executor.submit(
                self._run_notification_sink, payload, item_ids,
            )
        except Exception:
            raise
        self._sink_tasks.append({
            "future": future,
            "submitted_at": time.monotonic(),
            "timeout_logged": False,
            "item_count": len(item_ids),
        })

    def _run_notification_sink(self, payload, item_ids):
        try:
            result = self.notification_sink(payload)
            if result is False:
                raise RuntimeError("notification_sink returned False")
            if isinstance(result, dict) and result.get("ack_required"):
                # 新闻通知的“已投递”语义交给桌面端收到并展示后显式调用
                # POST /notifications/ack 来确认；这里只是成功入队，不能
                # 在此标记 dispatched，否则 ACK 机制形同虚设——一旦这里
                # 清掉 notification_pending，之后即便桌面端没收到/没展示，
                # 也不会再重试投递。
                return
            self.history_store.mark_notifications_dispatched(item_ids)
        except Exception:
            logger.exception(
                "News notification sink failed; pending state retained"
            )
        finally:
            with self._sink_lock:
                self._sink_item_ids.difference_update(item_ids)
            self._sink_slots.release()

    def _reap_sink_tasks(self):
        now = time.monotonic()
        active = []
        for task in self._sink_tasks:
            if task["future"].done():
                continue
            if (
                now - task["submitted_at"] >= self.sink_timeout_seconds
                and not task["timeout_logged"]
            ):
                task["timeout_logged"] = True
                logger.warning(
                    "News notification sink exceeded %.1f seconds; "
                    "task remains bounded and isolated (items=%d)",
                    self.sink_timeout_seconds, task["item_count"],
                )
            active.append(task)
        self._sink_tasks = active


def start_news_monitor(notification_sink=None):
    """启动进程内唯一的新闻守护线程。"""
    global _MONITOR_INSTANCE
    with _MONITOR_LOCK:
        if _MONITOR_INSTANCE is None:
            _MONITOR_INSTANCE = NewsMonitor(notification_sink=notification_sink)
        elif _MONITOR_INSTANCE._closed:
            thread = _MONITOR_INSTANCE._thread
            refresh_thread = _MONITOR_INSTANCE._refresh_thread
            if (
                (thread is not None and thread.is_alive())
                or (refresh_thread is not None and refresh_thread.is_alive())
            ):
                raise RuntimeError("新闻监控正在关闭，请稍后重试")
            _MONITOR_INSTANCE = NewsMonitor(notification_sink=notification_sink)
        elif notification_sink is not None:
            _MONITOR_INSTANCE.notification_sink = notification_sink
        _MONITOR_INSTANCE.start()
        return _MONITOR_INSTANCE


def get_news_monitor():
    with _MONITOR_LOCK:
        return _MONITOR_INSTANCE


def stop_news_monitor():
    """停止进程内新闻守护线程。"""
    global _MONITOR_INSTANCE
    with _MONITOR_LOCK:
        instance = _MONITOR_INSTANCE
        if instance is None:
            return
        instance.shutdown(timeout=instance.fetch_timeout_seconds + 2)
        thread_alive = (
            instance._thread is not None
            and instance._thread.is_alive()
        )
        refresh_alive = (
            instance._refresh_thread is not None
            and instance._refresh_thread.is_alive()
        )
        if not thread_alive and not refresh_alive:
            _MONITOR_INSTANCE = None
