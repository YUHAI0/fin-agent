"""新闻模型与 AKShare 新闻源适配。"""
import datetime
import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import List

import akshare as ak

from fin_agent.datasources.normalize import to_plain_code, to_ts_code


SOURCE_STOCK_NEWS_EM = "stock_news_em"
SOURCE_GLOBAL_CLS = "stock_info_global_cls"
SOURCE_GLOBAL_EM = "stock_info_global_em"
SUPPORTED_NEWS_SOURCES = (
    SOURCE_STOCK_NEWS_EM,
    SOURCE_GLOBAL_CLS,
    SOURCE_GLOBAL_EM,
)
CHINA_TIMEZONE = datetime.timezone(datetime.timedelta(hours=8))


@dataclass
class NewsItem:
    id: str
    source: str
    source_id: str
    title: str
    summary: str
    url: str
    published_at: str
    symbols: List[str] = field(default_factory=list)
    fingerprint: str = ""
    title_day_fingerprint: str = ""
    fingerprint_version: int = 2

    def to_dict(self):
        return asdict(self)


class AkshareNewsAdapter:
    """将三个 AKShare 接口归一为 NewsItem。"""

    def fetch(self, source: str):
        if source == SOURCE_STOCK_NEWS_EM:
            raise ValueError("stock_news_em 必须通过 fetch_stock_symbol 按股票抓取")
        if source == SOURCE_GLOBAL_CLS:
            return self._normalize_frame(ak.stock_info_global_cls(), source)
        if source == SOURCE_GLOBAL_EM:
            return self._normalize_frame(ak.stock_info_global_em(), source)
        raise ValueError(f"不支持的新闻源：{source}")

    def fetch_stock_symbol(self, raw_symbol):
        symbol = to_ts_code(raw_symbol)
        frame = ak.stock_news_em(symbol=to_plain_code(symbol))
        return self._normalize_frame(
            frame,
            SOURCE_STOCK_NEWS_EM,
            requested_symbol=symbol,
        )

    def _normalize_frame(self, frame, source, requested_symbol=None):
        if frame is None or getattr(frame, "empty", True):
            return []
        items = []
        for _, row in frame.iterrows():
            raw = row.to_dict()
            title = _text(_pick(raw, "新闻标题", "标题", "title"))
            summary = _text(_pick(raw, "新闻内容", "内容", "摘要", "summary"))
            url = _text(_pick(raw, "新闻链接", "链接", "url"))
            published_at = _published_at(raw)
            source_id = _text(_pick(
                raw, "新闻ID", "资讯代码", "文章编号", "id",
            ))
            symbols = (
                [requested_symbol]
                if requested_symbol
                else _extract_explicit_symbols(" ".join((title, summary)))
            )
            if not title and summary:
                title = summary[:80]
            if not title:
                continue
            if not source_id:
                source_id = _digest(source, title, url, published_at)
            item_id = f"{source}:{source_id}"
            fingerprint, title_day_fingerprint = news_fingerprints(
                title, summary, published_at,
            )
            items.append(NewsItem(
                id=item_id,
                source=source,
                source_id=source_id,
                title=title,
                summary=summary,
                url=url,
                published_at=published_at,
                symbols=symbols,
                fingerprint=fingerprint,
                title_day_fingerprint=title_day_fingerprint,
            ))
        return _deduplicate(items)


def _pick(row, *keys):
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        try:
            if value != value:
                continue
        except Exception:
            pass
        if str(value).strip():
            return value
    return None


def _text(value):
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() in ("nan", "nat", "none", "<na>") else text


def _published_at(row):
    value = _pick(row, "发布时间", "发布日期", "时间", "datetime", "date")
    date_value = _pick(row, "发布日期", "日期")
    time_value = _pick(row, "发布时间", "时间")
    if date_value is not None and time_value is not None:
        date_text = _text(date_value)
        time_text = _text(time_value)
        if date_text and date_text not in time_text:
            value = f"{date_text} {time_text}"
    if value is None:
        return ""
    if isinstance(value, datetime.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=CHINA_TIMEZONE)
        return value.isoformat(timespec="seconds")
    if isinstance(value, datetime.date):
        return datetime.datetime.combine(
            value, datetime.time.min, tzinfo=CHINA_TIMEZONE,
        ).isoformat(timespec="seconds")
    text = _text(value)
    try:
        parsed = datetime.datetime.fromisoformat(text.replace("/", "-"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=CHINA_TIMEZONE)
        return parsed.isoformat(timespec="seconds")
    except ValueError:
        return text


def _extract_explicit_symbols(text):
    """只识别带交易所或“股票代码”上下文的代码，避免把日期等六位数当股票。"""
    symbols = []
    patterns = (
        r"(?i)(?:SH|SZ|BJ)[.:\s-]*([0-9]{6})(?!\d)",
        r"(?i)([0-9]{6})[.\s-]*(?:SH|SZ|BJ)(?![A-Za-z])",
        r"(?:股票|证券)代码\s*[:：]?\s*([0-9]{6})(?!\d)",
    )
    for pattern in patterns:
        symbols.extend(re.findall(pattern, text or ""))
    normalized = []
    for plain in symbols:
        try:
            normalized.append(to_ts_code(plain))
        except ValueError:
            continue
    return sorted(set(normalized))


def news_fingerprints(title, summary="", published_at=""):
    """返回“标题+日期+摘要”主指纹及同日标题候选指纹。"""
    normalized_title = _normalize_text(title)
    if not normalized_title:
        return "", ""
    published_date = _beijing_date(published_at)
    normalized_summary = _normalize_text(summary)[:160]
    return (
        _digest(
            "news-v2",
            published_date,
            normalized_title,
            normalized_summary,
        ),
        _digest("news-title-day-v2", published_date, normalized_title),
    )


def news_fingerprint(title, summary="", published_at=""):
    return news_fingerprints(title, summary, published_at)[0]


def _normalize_text(value):
    return re.sub(r"[\W_]+", "", (value or "").casefold(), flags=re.UNICODE)


def _beijing_date(value):
    try:
        parsed = datetime.datetime.fromisoformat(
            str(value or "").replace("Z", "+00:00")
        )
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=CHINA_TIMEZONE)
        return parsed.astimezone(CHINA_TIMEZONE).date().isoformat()
    except ValueError:
        return datetime.datetime.now(CHINA_TIMEZONE).date().isoformat()


def _digest(*parts):
    raw = "\x1f".join(str(part or "") for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _deduplicate(items):
    by_id = {}
    for item in items:
        existing = by_id.get(item.id)
        if existing is None:
            by_id[item.id] = item
            continue
        existing.symbols = sorted(set(existing.symbols) | set(item.symbols))
    return sorted(
        by_id.values(),
        key=lambda item: (item.published_at, item.id),
        reverse=True,
    )
