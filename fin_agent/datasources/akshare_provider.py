"""akshare 数据源实现。对外统一为 Tushare 风格的代码与列名。"""
import threading
import time

import akshare as ak
import pandas as pd

from fin_agent.datasources.base import (
    CAP_DAILY_BASIC, CAP_DAILY_PRICE, CAP_INCOME_STATEMENT, CAP_INDEX_DAILY,
    CAP_REALTIME_PRICE, CAP_STOCK_BASIC, CAP_TRADE_CALENDAR, MarketDataProvider,
)
from fin_agent.datasources.normalize import (
    DAILY_COLUMNS, REALTIME_COLUMNS, normalize_date, rename_and_select, to_plain_code, to_ts_code,
)

_SPOT_CACHE = {"data": None, "ts": 0.0}
_SPOT_LOCK = threading.Lock()
_SPOT_TTL_SECONDS = 5

_ADJ_MAP = {None: "", "": "", "qfq": "qfq", "hfq": "hfq"}

_DAILY_MAPPING = {
    "日期": "trade_date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "vol",
    "成交额": "amount",
    "涨跌幅": "pct_chg",
    "涨跌额": "change",
}


def _spot_snapshot():
    """全市场实时快照，5 秒内复用，避免组合估值时重复请求。"""
    with _SPOT_LOCK:
        now = time.time()
        if _SPOT_CACHE["data"] is not None and now - _SPOT_CACHE["ts"] < _SPOT_TTL_SECONDS:
            return _SPOT_CACHE["data"]
        df = ak.stock_zh_a_spot_em()
        _SPOT_CACHE["data"] = df
        _SPOT_CACHE["ts"] = now
        return df


class AkshareProvider(MarketDataProvider):
    name = "akshare"
    CAPABILITIES = {
        CAP_STOCK_BASIC, CAP_DAILY_PRICE, CAP_REALTIME_PRICE,
        CAP_DAILY_BASIC, CAP_INCOME_STATEMENT, CAP_INDEX_DAILY, CAP_TRADE_CALENDAR,
    }

    def get_stock_basic(self, ts_code=None, name=None):
        self.require(CAP_STOCK_BASIC)
        df = ak.stock_info_a_code_name()
        df = df.rename(columns={"code": "symbol", "name": "name"})
        df["ts_code"] = df["symbol"].apply(lambda c: _safe_ts_code(c))
        df = df.dropna(subset=["ts_code"])
        if ts_code:
            df = df[df["ts_code"] == to_ts_code(ts_code)]
        elif name:
            df = df[df["name"].str.contains(name, na=False)]
        for col in ("area", "industry", "market", "list_date"):
            df[col] = None
        return df[["ts_code", "symbol", "name", "area", "industry", "market", "list_date"]]

    def get_daily_price(self, ts_code, start_date=None, end_date=None, adj=None):
        self.require(CAP_DAILY_PRICE)
        full = to_ts_code(ts_code)
        df = ak.stock_zh_a_hist(
            symbol=to_plain_code(full),
            period="daily",
            start_date=normalize_date(start_date) or "19900101",
            end_date=normalize_date(end_date) or "20991231",
            adjust=_ADJ_MAP.get(adj, ""),
        )
        out = rename_and_select(df, _DAILY_MAPPING, DAILY_COLUMNS)
        if len(out):
            out["ts_code"] = full
            out["trade_date"] = out["trade_date"].apply(normalize_date)
            out["pre_close"] = out["close"] - out["change"]
        return out

    def get_realtime_price(self, ts_code):
        self.require(CAP_REALTIME_PRICE)
        codes = [ts_code] if isinstance(ts_code, str) else list(ts_code)
        wanted = {}
        for code in codes:
            full = to_ts_code(code)
            wanted[to_plain_code(full)] = full

        snapshot = _spot_snapshot()
        hit = snapshot[snapshot["代码"].isin(wanted.keys())]

        rows = []
        for _, row in hit.iterrows():
            plain = row["代码"]
            rows.append({
                "ts_code": wanted[plain],
                "name": row.get("名称"),
                "price": _to_float(row.get("最新价")),
                "pre_close": _to_float(row.get("昨收")),
                "change": _to_float(row.get("涨跌额")),
                "pct_chg": _to_float(row.get("涨跌幅")),
                "vol": _to_float(row.get("成交量")),
                "amount": _to_float(row.get("成交额")),
            })
        return pd.DataFrame(rows, columns=REALTIME_COLUMNS)

    def get_daily_basic(self, ts_code, start_date=None, end_date=None):
        self.require(CAP_DAILY_BASIC)
        full = to_ts_code(ts_code)
        # stock_a_indicator_lg 已随乐咕数据源下线；东财 stock_value_em 提供日频估值序列
        df = ak.stock_value_em(symbol=to_plain_code(full))
        mapping = {
            "数据日期": "trade_date",
            "PE(静)": "pe",
            "PE(TTM)": "pe_ttm",
            "市净率": "pb",
            "市销率": "ps_ttm",
            "总市值": "total_mv",
            "流通市值": "circ_mv",
        }
        columns = ["ts_code", "trade_date", "pe", "pe_ttm", "pb", "ps_ttm", "dv_ratio", "total_mv", "circ_mv"]
        out = rename_and_select(df, mapping, columns)
        if len(out):
            out["ts_code"] = full
            out["trade_date"] = out["trade_date"].apply(normalize_date)
            # 东财市值为元，Tushare 风格统一为万元
            out["total_mv"] = out["total_mv"].apply(lambda v: None if v is None else float(v) / 10000.0)
            out["circ_mv"] = out["circ_mv"].apply(lambda v: None if v is None else float(v) / 10000.0)
            start = normalize_date(start_date)
            end = normalize_date(end_date)
            if start:
                out = out[out["trade_date"] >= start]
            if end:
                out = out[out["trade_date"] <= end]
        return out

    def get_income_statement(self, ts_code, start_date=None, end_date=None):
        self.require(CAP_INCOME_STATEMENT)
        full = to_ts_code(ts_code)
        prefix = "SH" if full.endswith(".SH") else ("SZ" if full.endswith(".SZ") else "BJ")
        df = ak.stock_profit_sheet_by_report_em(symbol=f"{prefix}{to_plain_code(full)}")
        mapping = {
            "REPORT_DATE": "end_date",
            "TOTAL_OPERATE_INCOME": "total_revenue",
            "OPERATE_INCOME": "revenue",
            "OPERATE_PROFIT": "operate_profit",
            "TOTAL_PROFIT": "total_profit",
            "NETPROFIT": "n_income",
        }
        columns = ["ts_code", "end_date", "total_revenue", "revenue", "operate_profit", "total_profit", "n_income"]
        out = rename_and_select(df, mapping, columns)
        if len(out):
            out["ts_code"] = full
            out["end_date"] = out["end_date"].apply(normalize_date)
            start = normalize_date(start_date)
            end = normalize_date(end_date)
            if start:
                out = out[out["end_date"] >= start]
            if end:
                out = out[out["end_date"] <= end]
        return out

    def get_index_daily(self, ts_code, start_date=None, end_date=None):
        self.require(CAP_INDEX_DAILY)
        plain = to_plain_code(ts_code)
        df = ak.index_zh_a_hist(
            symbol=plain,
            period="daily",
            start_date=normalize_date(start_date) or "19900101",
            end_date=normalize_date(end_date) or "20991231",
        )
        out = rename_and_select(df, _DAILY_MAPPING, DAILY_COLUMNS)
        if len(out):
            out["ts_code"] = str(ts_code).upper()
            out["trade_date"] = out["trade_date"].apply(normalize_date)
            out["pre_close"] = out["close"] - out["change"]
        return out

    def get_trade_calendar(self, start_date=None, end_date=None):
        self.require(CAP_TRADE_CALENDAR)
        df = ak.tool_trade_date_hist_sina()
        out = pd.DataFrame({"cal_date": df["trade_date"].apply(normalize_date)})
        start = normalize_date(start_date)
        end = normalize_date(end_date)
        if start:
            out = out[out["cal_date"] >= start]
        if end:
            out = out[out["cal_date"] <= end]
        return out


def _safe_ts_code(code):
    try:
        return to_ts_code(code)
    except ValueError:
        return None


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
