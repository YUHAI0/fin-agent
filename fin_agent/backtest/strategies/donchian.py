from __future__ import annotations

from typing import Any, Mapping, Optional

import pandas as pd

from fin_agent.backtest.base import BacktestStrategy
from fin_agent.backtest.registry import register_strategy
from fin_agent.tools.technical_indicators import calculate_atr


@register_strategy
class DonchianBreakoutStrategy(BacktestStrategy):
    """唐奇安通道：突破 N 日高做多，跌破 N 日低平仓。"""

    name = "donchian_breakout"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        n = int(params.get("channel_period", 20))
        df["dc_upper"] = df["high"].rolling(n).max().shift(1)
        df["dc_lower"] = df["low"].rolling(n).min().shift(1)
        return df

    def signal(self, row, prev_row, params: Mapping[str, Any], context: Optional[Mapping[str, Any]] = None) -> int:
        if prev_row is None:
            return 0
        if self._na(row.get("dc_upper"), row.get("dc_lower")):
            return 0
        if row["close"] > row["dc_upper"]:
            return 1
        if row["close"] < row["dc_lower"]:
            return -1
        return 0


@register_strategy
class TurtleStrategy(BacktestStrategy):
    """海龟式：入场窗口突破 N 日高，出场窗口跌破 M 日低；可选 ATR 硬止损。"""

    name = "turtle"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        entry_n = int(params.get("entry_period", 20))
        exit_n = int(params.get("exit_period", 10))
        df["dc_high"] = df["high"].rolling(entry_n).max().shift(1)
        df["dc_exit"] = df["low"].rolling(exit_n).min().shift(1)
        atr_p = int(params.get("atr_period", 14))
        if float(params.get("atr_stop_mult", 0) or 0) > 0:
            calculate_atr(df, period=atr_p)
        return df

    def signal(self, row, prev_row, params: Mapping[str, Any], context: Optional[Mapping[str, Any]] = None) -> int:
        context = context or {}
        if prev_row is None:
            return 0
        pos = int(context.get("position") or 0)
        if self._na(row.get("dc_high"), row.get("dc_exit")):
            return 0
        if pos > 0 and row["close"] < row["dc_exit"]:
            return -1
        if pos == 0 and row["close"] > row["dc_high"]:
            return 1
        return 0

    def risk_exit(
        self,
        row: pd.Series,
        prev_row: Optional[pd.Series],
        params: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> int:
        mult = float(params.get("atr_stop_mult", 0) or 0)
        if mult <= 0:
            return 0
        ep = context.get("entry_price")
        if ep is None or self._na(row.get("atr")):
            return 0
        if row["close"] < float(ep) - mult * float(row["atr"]):
            return -1
        return 0
