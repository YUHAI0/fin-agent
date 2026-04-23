from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from fin_agent.backtest.base import BacktestStrategy
from fin_agent.backtest.registry import register_strategy
from fin_agent.tools.technical_indicators import calculate_macd


@register_strategy
class MacdStrategy(BacktestStrategy):
    name = "macd"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        fast_p = int(params.get("fast_period", 12))
        slow_p = int(params.get("slow_period", 26))
        signal_p = int(params.get("signal_period", 9))
        return calculate_macd(df, fast_period=fast_p, slow_period=slow_p, signal_period=signal_p)

    def signal(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        if prev_row is None:
            return 0
        if self._na(row.get("dif"), row.get("dea"), prev_row.get("dif"), prev_row.get("dea")):
            return 0
        if prev_row["dif"] <= prev_row["dea"] and row["dif"] > row["dea"]:
            return 1
        if prev_row["dif"] >= prev_row["dea"] and row["dif"] < row["dea"]:
            return -1
        return 0
