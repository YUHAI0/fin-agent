from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from fin_agent.backtest.base import BacktestStrategy
from fin_agent.backtest.registry import register_strategy
from fin_agent.tools.technical_indicators import calculate_boll


def _boll_enrich(df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    period = int(params.get("period", 20))
    std_dev = float(params.get("std_dev", 2))
    return calculate_boll(df, period=period, std_dev=std_dev)


@register_strategy
class BollReversionStrategy(BacktestStrategy):
    name = "boll_reversion"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        return _boll_enrich(df, params)

    def signal(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        if prev_row is None:
            return 0
        for col in ("close", "boll_lower", "boll_mid", "boll_upper"):
            if self._na(row.get(col), prev_row.get(col)):
                return 0
        if prev_row["close"] < prev_row["boll_lower"] and row["close"] >= row["boll_lower"]:
            return 1
        if prev_row["close"] < prev_row["boll_mid"] and row["close"] >= row["boll_mid"]:
            return -1
        return 0


@register_strategy
class BollBreakoutStrategy(BacktestStrategy):
    name = "boll_breakout"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        return _boll_enrich(df, params)

    def signal(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        if prev_row is None:
            return 0
        for col in ("close", "boll_lower", "boll_mid", "boll_upper"):
            if self._na(row.get(col), prev_row.get(col)):
                return 0
        if prev_row["close"] <= prev_row["boll_upper"] and row["close"] > row["boll_upper"]:
            return 1
        if prev_row["close"] >= prev_row["boll_mid"] and row["close"] < row["boll_mid"]:
            return -1
        return 0
