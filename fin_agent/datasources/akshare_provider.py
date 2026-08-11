"""akshare 数据源实现。对外统一为 Tushare 风格的代码与列名。"""
import re

import akshare as ak
import pandas as pd
import requests

from fin_agent.datasources.base import (
    CAP_DAILY_BASIC, CAP_DAILY_PRICE, CAP_INCOME_STATEMENT, CAP_INDEX_DAILY,
    CAP_REALTIME_PRICE, CAP_STOCK_BASIC, CAP_TRADE_CALENDAR, MarketDataProvider,
)
from fin_agent.datasources.normalize import (
    DAILY_COLUMNS, REALTIME_COLUMNS, normalize_date, rename_and_select, to_plain_code, to_ts_code,
)

_QUOTE_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://finance.sina.com.cn",
}

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

_TX_DAILY_MAPPING = {
    "date": "trade_date",
    "open": "open",
    "close": "close",
    "high": "high",
    "low": "low",
    "volume": "vol",
    "amount": "amount",
}

_INDEX_SINA_MAPPING = {
    "date": "trade_date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "vol",
}


def _market_symbol(ts_code):
    """600519.SH -> sh600519；002594.SZ -> sz002594。"""
    full = to_ts_code(ts_code)
    plain = to_plain_code(full)
    if full.endswith(".SH"):
        return f"sh{plain}"
    if full.endswith(".BJ"):
        return f"bj{plain}"
    return f"sz{plain}"


def _plain_from_spot_code(code):
    text = str(code).strip()
    return re.sub(r"^(sh|sz|bj)", "", text, flags=re.IGNORECASE)


def _realtime_row(ts_code, name, price, pre_close, vol=0.0, amount=0.0):
    price = _to_float(price)
    pre_close = _to_float(pre_close)
    change = price - pre_close
    return {
        "ts_code": ts_code,
        "name": name,
        "price": price,
        "pre_close": pre_close,
        "change": round(change, 4),
        "pct_chg": round(change / pre_close * 100, 4) if pre_close else 0.0,
        "vol": _to_float(vol),
        "amount": _to_float(amount),
    }


def _fetch_quotes_sina(wanted):
    """新浪单票/批量行情。hq.sinajs.cn 比 ak.stock_zh_a_spot 全市场接口更稳。"""
    symbols = [_market_symbol(full) for full in wanted.values()]
    url = "https://hq.sinajs.cn/list=" + ",".join(symbols)
    resp = requests.get(url, headers=_QUOTE_HEADERS, timeout=10)
    resp.encoding = "gbk"
    by_plain = {to_plain_code(full): full for full in wanted.values()}
    rows = []
    for line in resp.text.splitlines():
        if '="' not in line:
            continue
        left, right = line.split('="', 1)
        payload = right.rstrip('";')
        if not payload:
            continue
        sym = left.rsplit("_", 1)[-1]
        plain = _plain_from_spot_code(sym)
        full = by_plain.get(plain)
        if not full:
            continue
        parts = payload.split(",")
        if len(parts) < 10:
            continue
        rows.append(_realtime_row(
            full, parts[0], parts[3], parts[2], vol=parts[8], amount=parts[9],
        ))
    return rows


def _fetch_quotes_tencent(wanted):
    """腾讯行情回退。"""
    symbols = [_market_symbol(full) for full in wanted.values()]
    url = "https://qt.gtimg.cn/q=" + ",".join(symbols)
    resp = requests.get(url, headers=_QUOTE_HEADERS, timeout=10)
    resp.encoding = "gbk"
    by_plain = {to_plain_code(full): full for full in wanted.values()}
    rows = []
    for chunk in resp.text.split(";"):
        chunk = chunk.strip()
        if '="' not in chunk:
            continue
        left, right = chunk.split('="', 1)
        payload = right.rstrip('"')
        if not payload:
            continue
        sym = left.rsplit("_", 1)[-1]
        plain = _plain_from_spot_code(sym)
        full = by_plain.get(plain)
        if not full:
            continue
        parts = payload.split("~")
        if len(parts) < 7:
            continue
        # 1=名称 3=现价 4=昨收 6=成交量(手)
        vol = _to_float(parts[6]) * 100  # 手 -> 股，与新浪口径对齐
        amount = _to_float(parts[37]) * 10000 if len(parts) > 37 else 0.0
        rows.append(_realtime_row(
            full, parts[1], parts[3], parts[4], vol=vol, amount=amount,
        ))
    return rows


def _with_derived_ohlc(out):
    """由收盘价序列补 pre_close / change / pct_chg。"""
    if not len(out):
        return out
    out = out.copy()
    closes = pd.to_numeric(out["close"], errors="coerce")
    out["pre_close"] = closes.shift(1)
    if out["pre_close"].isna().iloc[0] and len(closes):
        # 无前一日时用开盘近似，避免整列空
        opens = pd.to_numeric(out["open"], errors="coerce")
        out.loc[out.index[0], "pre_close"] = opens.iloc[0]
    out["change"] = closes - pd.to_numeric(out["pre_close"], errors="coerce")
    out["pct_chg"] = out.apply(
        lambda r: (float(r["change"]) / float(r["pre_close"]) * 100) if r["pre_close"] else 0.0,
        axis=1,
    )
    return out


class AkshareProvider(MarketDataProvider):
    name = "akshare"
    CAPABILITIES = {
        CAP_STOCK_BASIC, CAP_DAILY_PRICE, CAP_REALTIME_PRICE,
        CAP_DAILY_BASIC, CAP_INCOME_STATEMENT, CAP_INDEX_DAILY, CAP_TRADE_CALENDAR,
    }

    def get_stock_basic(self, ts_code=None, name=None):
        self.require(CAP_STOCK_BASIC)
        # 按代码查询走单票行情拿名称，避免拉全市场列表
        if ts_code and not name:
            full = to_ts_code(ts_code)
            plain = to_plain_code(full)
            wanted = {plain: full}
            rows = []
            try:
                rows = _fetch_quotes_sina(wanted)
            except Exception:
                rows = []
            if not rows:
                try:
                    rows = _fetch_quotes_tencent(wanted)
                except Exception:
                    rows = []
            if rows:
                return pd.DataFrame([{
                    "ts_code": full,
                    "symbol": plain,
                    "name": rows[0].get("name"),
                    "area": None,
                    "industry": None,
                    "market": None,
                    "list_date": None,
                }])

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
        plain = to_plain_code(full)
        start = normalize_date(start_date) or "19900101"
        end = normalize_date(end_date) or "20991231"
        adjust = _ADJ_MAP.get(adj, "")

        # 优先腾讯单票日线（本网络更稳）；失败再试东财
        out = pd.DataFrame(columns=DAILY_COLUMNS)
        try:
            df = ak.stock_zh_a_hist_tx(
                symbol=_market_symbol(full),
                start_date=start,
                end_date=end,
                adjust=adjust,
            )
            out = rename_and_select(df, _TX_DAILY_MAPPING, DAILY_COLUMNS)
            out = _with_derived_ohlc(out)
        except Exception:
            try:
                df = ak.stock_zh_a_hist(
                    symbol=plain,
                    period="daily",
                    start_date=start,
                    end_date=end,
                    adjust=adjust,
                )
                out = rename_and_select(df, _DAILY_MAPPING, DAILY_COLUMNS)
            except Exception:
                return out

        if len(out):
            out["ts_code"] = full
            out["trade_date"] = out["trade_date"].apply(normalize_date)
            if out["change"].notna().any() and out["pre_close"].isna().all():
                out["pre_close"] = out["close"] - out["change"]
            elif out["pre_close"].isna().any():
                out = _with_derived_ohlc(out)
        return out

    def get_realtime_price(self, ts_code):
        self.require(CAP_REALTIME_PRICE)
        codes = [ts_code] if isinstance(ts_code, str) else list(ts_code)
        wanted = {}
        for code in codes:
            full = to_ts_code(code)
            wanted[to_plain_code(full)] = full

        # 仅单票接口，不再回退全市场 spot（会触发东财多页 tqdm）
        rows = []
        try:
            rows = _fetch_quotes_sina(wanted)
        except Exception:
            rows = []
        if not rows:
            try:
                rows = _fetch_quotes_tencent(wanted)
            except Exception:
                rows = []
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
        plain = to_plain_code(full)
        columns = ["ts_code", "end_date", "total_revenue", "revenue", "operate_profit", "total_profit", "n_income"]
        out = pd.DataFrame(columns=columns)

        # 优先新浪单票利润表，避免东财多请求分页
        try:
            df = ak.stock_financial_report_sina(stock=plain, symbol="利润表")
            mapping = {
                "报告日": "end_date",
                "营业总收入": "total_revenue",
                "营业收入": "revenue",
                "营业利润": "operate_profit",
                "利润总额": "total_profit",
                "净利润": "n_income",
            }
            out = rename_and_select(df, mapping, columns)
        except Exception:
            try:
                prefix = "SH" if full.endswith(".SH") else ("SZ" if full.endswith(".SZ") else "BJ")
                df = ak.stock_profit_sheet_by_report_em(symbol=f"{prefix}{plain}")
                mapping = {
                    "REPORT_DATE": "end_date",
                    "TOTAL_OPERATE_INCOME": "total_revenue",
                    "OPERATE_INCOME": "revenue",
                    "OPERATE_PROFIT": "operate_profit",
                    "TOTAL_PROFIT": "total_profit",
                    "NETPROFIT": "n_income",
                }
                out = rename_and_select(df, mapping, columns)
            except Exception:
                return out

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
        start = normalize_date(start_date) or "19900101"
        end = normalize_date(end_date) or "20991231"
        code_out = str(ts_code).upper() if "." in str(ts_code) else to_ts_code(f"{plain}.SH")

        # 优先新浪单指数日线；失败再试东财
        try:
            symbol = _market_symbol(code_out)
            df = ak.stock_zh_index_daily(symbol=symbol)
            out = rename_and_select(df, _INDEX_SINA_MAPPING, DAILY_COLUMNS)
            if len(out):
                out["ts_code"] = code_out
                out["trade_date"] = out["trade_date"].apply(normalize_date)
                out = out[(out["trade_date"] >= start) & (out["trade_date"] <= end)]
                out = _with_derived_ohlc(out)
            return out
        except Exception:
            try:
                df = ak.index_zh_a_hist(
                    symbol=plain,
                    period="daily",
                    start_date=start,
                    end_date=end,
                )
                out = rename_and_select(df, _DAILY_MAPPING, DAILY_COLUMNS)
                if len(out):
                    out["ts_code"] = code_out
                    out["trade_date"] = out["trade_date"].apply(normalize_date)
                    out["pre_close"] = out["close"] - out["change"]
                return out
            except Exception:
                return pd.DataFrame(columns=DAILY_COLUMNS)

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
