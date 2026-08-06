"""行情数据源 provider 包。"""
import threading

from fin_agent.config import Config
from fin_agent.datasources.base import (
    CAP_DAILY_BASIC, CAP_DAILY_PRICE, CAP_INCOME_STATEMENT, CAP_INDEX_DAILY,
    CAP_REALTIME_PRICE, CAP_STOCK_BASIC, CAP_TRADE_CALENDAR,
    CapabilityNotSupported, MarketDataProvider,
)

_CACHE = {}
_LOCK = threading.Lock()


def _build(name):
    if name == "tushare":
        from fin_agent.datasources.tushare_provider import TushareProvider
        return TushareProvider()
    from fin_agent.datasources.akshare_provider import AkshareProvider
    return AkshareProvider()


def get_provider():
    """按 app_config.json 的 data_source 返回 provider 实例，按名称缓存。"""
    name = Config.DATA_SOURCE or "akshare"
    with _LOCK:
        if name not in _CACHE:
            _CACHE[name] = _build(name)
        return _CACHE[name]


def reset_provider_cache():
    """配置变更后清空缓存，使下次 get_provider 重新构建。"""
    with _LOCK:
        _CACHE.clear()


__all__ = [
    "get_provider", "reset_provider_cache", "MarketDataProvider", "CapabilityNotSupported",
    "CAP_STOCK_BASIC", "CAP_DAILY_PRICE", "CAP_REALTIME_PRICE", "CAP_DAILY_BASIC",
    "CAP_INCOME_STATEMENT", "CAP_INDEX_DAILY", "CAP_TRADE_CALENDAR",
]
