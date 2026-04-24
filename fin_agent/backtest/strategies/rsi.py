from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from fin_agent.backtest.base import BacktestStrategy
from fin_agent.backtest.registry import register_strategy
from fin_agent.tools.technical_indicators import calculate_rsi


@register_strategy
class RsiStrategy(BacktestStrategy):
    name = "rsi"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        window = int(params.get("window", 14))
        return calculate_rsi(df, period=window)

    def signal(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        if prev_row is None:
            return 0
        lower = int(params.get("lower", 30))
        upper = int(params.get("upper", 70))
        if self._na(row.get("rsi"), prev_row.get("rsi")):
            return 0
        if prev_row["rsi"] >= lower and row["rsi"] < lower:
            return 1
        if prev_row["rsi"] <= upper and row["rsi"] > upper:
            return -1
        return 0
