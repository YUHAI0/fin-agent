"""Tushare 数据源实现。取数逻辑与原 tools/tushare_tools.py 保持一致。"""
import pandas as pd
import tushare as ts

from fin_agent.config import Config
from fin_agent.datasources.base import (
    CAP_DAILY_BASIC, CAP_DAILY_PRICE, CAP_INCOME_STATEMENT, CAP_INDEX_DAILY,
    CAP_REALTIME_PRICE, CAP_STOCK_BASIC, CAP_TRADE_CALENDAR, MarketDataProvider,
)
from fin_agent.datasources.normalize import (
    DAILY_COLUMNS, REALTIME_COLUMNS, normalize_date, rename_and_select, to_plain_code, to_ts_code,
)


class TushareProvider(MarketDataProvider):
    name = "tushare"
    CAPABILITIES = {
        CAP_STOCK_BASIC, CAP_DAILY_PRICE, CAP_REALTIME_PRICE,
        CAP_DAILY_BASIC, CAP_INCOME_STATEMENT, CAP_INDEX_DAILY, CAP_TRADE_CALENDAR,
    }

    def _pro(self):
        if not Config.TUSHARE_TOKEN:
            raise RuntimeError("未配置 Tushare Token，请在设置页填写后重试")
        ts.set_token(Config.TUSHARE_TOKEN)
        return ts.pro_api()

    def get_stock_basic(self, ts_code=None, name=None):
        self.require(CAP_STOCK_BASIC)
        fields = "ts_code,symbol,name,area,industry,market,list_date"
        if ts_code:
            df = self._pro().stock_basic(ts_code=to_ts_code(ts_code), fields=fields)
        elif name:
            df = self._pro().stock_basic(name=name, fields=fields)
        else:
            df = self._pro().stock_basic(list_status="L", fields=fields)
        return df

    def get_daily_price(self, ts_code, start_date=None, end_date=None, adj=None):
        self.require(CAP_DAILY_PRICE)
        code = to_ts_code(ts_code)
        df = ts.pro_bar(
            ts_code=code,
            adj=adj,
            start_date=normalize_date(start_date),
            end_date=normalize_date(end_date),
        )
        if df is None:
            df = pd.DataFrame()
        return rename_and_select(df, {}, DAILY_COLUMNS)

    def get_realtime_price(self, ts_code):
        self.require(CAP_REALTIME_PRICE)
        codes = [ts_code] if isinstance(ts_code, str) else list(ts_code)
        rows = []
        for code in codes:
            full = to_ts_code(code)
            df = ts.get_realtime_quotes(to_plain_code(full))
            if df is None or len(df) == 0:
                continue
            row = df.iloc[0]
            price = float(row["price"]) if row["price"] else 0.0
            pre_close = float(row["pre_close"]) if row["pre_close"] else 0.0
            change = price - pre_close
            rows.append({
                "ts_code": full,
                "name": row.get("name"),
                "price": price,
                "pre_close": pre_close,
                "change": round(change, 4),
                "pct_chg": round(change / pre_close * 100, 4) if pre_close else 0.0,
                "vol": float(row.get("volume") or 0),
                "amount": float(row.get("amount") or 0),
            })
        return pd.DataFrame(rows, columns=REALTIME_COLUMNS)

    def get_daily_basic(self, ts_code, start_date=None, end_date=None):
        self.require(CAP_DAILY_BASIC)
        df = self._pro().daily_basic(
            ts_code=to_ts_code(ts_code),
            start_date=normalize_date(start_date),
            end_date=normalize_date(end_date),
            fields="ts_code,trade_date,pe,pe_ttm,pb,ps_ttm,dv_ratio,total_mv,circ_mv",
        )
        return df

    def get_income_statement(self, ts_code, start_date=None, end_date=None):
        self.require(CAP_INCOME_STATEMENT)
        df = self._pro().income(
            ts_code=to_ts_code(ts_code),
            start_date=normalize_date(start_date),
            end_date=normalize_date(end_date),
            fields="ts_code,end_date,total_revenue,revenue,operate_profit,total_profit,n_income",
        )
        return df

    def get_index_daily(self, ts_code, start_date=None, end_date=None):
        self.require(CAP_INDEX_DAILY)
        df = self._pro().index_daily(
            ts_code=to_ts_code(ts_code) if "." in str(ts_code) else ts_code,
            start_date=normalize_date(start_date),
            end_date=normalize_date(end_date),
        )
        if df is None:
            df = pd.DataFrame()
        return rename_and_select(df, {}, DAILY_COLUMNS)

    def get_trade_calendar(self, start_date=None, end_date=None):
        self.require(CAP_TRADE_CALENDAR)
        df = self._pro().trade_cal(
            exchange="SSE",
            start_date=normalize_date(start_date),
            end_date=normalize_date(end_date),
            is_open="1",
            fields="cal_date",
        )
        return df
