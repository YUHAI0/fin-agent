from __future__ import annotations

from typing import Any, Mapping, Optional

import pandas as pd

from fin_agent.backtest.base import BacktestStrategy
from fin_agent.backtest.registry import register_strategy
from fin_agent.tools.technical_indicators import calculate_adx_di, calculate_macd


@register_strategy
class AdxMacdStrategy(BacktestStrategy):
    """ADX + DI 过滤趋势强度后，再用 MACD 金叉/死叉交易。"""

    name = "adx_macd"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        adx_p = int(params.get("adx_period", 14))
        calculate_adx_di(df, period=adx_p)
        fast_p = int(params.get("fast_period", 12))
        slow_p = int(params.get("slow_period", 26))
        signal_p = int(params.get("signal_period", 9))
        return calculate_macd(df, fast_period=fast_p, slow_period=slow_p, signal_period=signal_p)

    def signal(self, row, prev_row, params: Mapping[str, Any], context: Optional[Mapping[str, Any]] = None) -> int:
        context = context or {}
        if prev_row is None:
            return 0
        min_adx = float(params.get("min_adx", 22.0))
        if self._na(
            row.get("adx"),
            row.get("plus_di"),
            row.get("minus_di"),
            row.get("dif"),
            row.get("dea"),
            prev_row.get("dif"),
            prev_row.get("dea"),
        ):
            return 0
        trend_ok = row["adx"] >= min_adx and row["plus_di"] > row["minus_di"]
        pos = int(context.get("position") or 0)
        macd_buy = prev_row["dif"] <= prev_row["dea"] and row["dif"] > row["dea"]
        macd_sell = prev_row["dif"] >= prev_row["dea"] and row["dif"] < row["dea"]
        trend_bad = row["adx"] < min_adx or row["plus_di"] <= row["minus_di"]

        if pos > 0 and (macd_sell or trend_bad):
            return -1
        if pos == 0 and trend_ok and macd_buy:
            return 1
        return 0
