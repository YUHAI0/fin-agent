import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor

from fin_agent.config import Config

logger = logging.getLogger(__name__)

VALID_SENTIMENTS = frozenset({"bullish", "bearish", "neutral"})

SENTIMENT_PROMPT = """你是 A 股财经新闻分析助手。根据新闻标题、摘要和关联股票，判断该消息对相关股票或板块的短期市场影响倾向。

仅输出一行 JSON，不要其他文字：
{{"sentiment":"bullish|bearish|neutral"}}

规则：
- bullish（利好）：明显正面预期，如业绩超预期、政策扶持、订单大增、涨价等
- bearish（利空）：明显负面，如亏损、监管处罚、减持、业绩下滑、风险警示等
- neutral（中性）：影响不明确、例行公告、或利好利空对冲

标题：{title}
摘要：{summary}
关联股票：{symbols}"""


def _parse_sentiment(text):
    text = (text or "").strip()
    if not text:
        return None
    try:
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            sentiment = str(data.get("sentiment", "")).strip().lower()
            if sentiment in VALID_SENTIMENTS:
                return sentiment
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    if "bullish" in text.casefold() or "利好" in text:
        return "bullish"
    if "bearish" in text.casefold() or "利空" in text:
        return "bearish"
    if "neutral" in text.casefold() or "中性" in text:
        return "neutral"
    return None


def label_news_sentiment(title, summary, symbols):
    """调用 LLM 标注单条新闻倾向；失败时返回 None。"""
    Config.load()
    if not Config.NEWS_SENTIMENT_ENABLED:
        return None
    try:
        from fin_agent.llm.factory import LLMFactory

        llm = LLMFactory.create_llm()
        symbol_text = ", ".join(symbols) if symbols else "无"
        prompt = SENTIMENT_PROMPT.format(
            title=title or "",
            summary=(summary or "")[:500],
            symbols=symbol_text,
        )
        message = llm.chat([{"role": "user", "content": prompt}], stream=False)
        content = (getattr(message, "content", "") or "").strip()
        return _parse_sentiment(content)
    except Exception as exc:
        logger.warning("News sentiment labeling failed: %s", exc)
        return None


class NewsSentimentLabeler:
    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls):
        with cls._init_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="news-sentiment",
                )
                inst._inflight_ids = set()
                inst._inflight_lock = threading.Lock()
                cls._instance = inst
            return cls._instance

    def enqueue_from_upsert_results(self, results, history_store):
        if not results:
            return
        Config.load()
        if not Config.NEWS_SENTIMENT_ENABLED:
            return
        for result in results:
            item = result.get("item") or {}
            if item.get("sentiment") in VALID_SENTIMENTS:
                continue
            self._enqueue_one(item, history_store)

    def enqueue_backlog(self, history_store, limit=5):
        Config.load()
        if not Config.NEWS_SENTIMENT_ENABLED:
            return
        for item in history_store.list_unlabeled_sentiment(limit=limit):
            self._enqueue_one(item, history_store)

    def _enqueue_one(self, item, history_store):
        news_id = item.get("id")
        if not news_id:
            return
        with self._inflight_lock:
            if news_id in self._inflight_ids:
                return
            self._inflight_ids.add(news_id)
        try:
            self._executor.submit(self._run_label, news_id, item, history_store)
        except Exception:
            with self._inflight_lock:
                self._inflight_ids.discard(news_id)
            logger.exception("Failed to submit sentiment labeling task for %s", news_id)

    def _run_label(self, news_id, item, history_store):
        try:
            Config.load()
            if not Config.NEWS_SENTIMENT_ENABLED:
                return
            sentiment = label_news_sentiment(
                item.get("title"),
                item.get("summary"),
                item.get("matched_symbols") or item.get("symbols") or [],
            )
            if sentiment:
                history_store.update_sentiment(news_id, sentiment)
        finally:
            with self._inflight_lock:
                self._inflight_ids.discard(news_id)


def get_sentiment_labeler():
    return NewsSentimentLabeler()
