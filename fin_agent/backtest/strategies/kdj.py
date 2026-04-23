from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from fin_agent.backtest.base import BacktestStrategy
from fin_agent.backtest.registry import register_strategy
from fin_agent.tools.technical_indicators import calculate_kdj


@register_strategy
class KdjStrategy(BacktestStrategy):
    name = "kdj"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        k_period = int(params.get("k_period", 9))
        d_period = int(params.get("d_period", 3))
        j_period = int(params.get("j_period", 3))
        return calculate_kdj(df, k_period=k_period, d_period=d_period, j_period=j_period)

    def signal(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        if prev_row is None:
            return 0
        for key in ("k", "d"):
            if self._na(row.get(key), prev_row.get(key)):
                return 0
        if prev_row["k"] <= prev_row["d"] and row["k"] > row["d"]:
            return 1
        if prev_row["k"] >= prev_row["d"] and row["k"] < row["d"]:
            return -1
        return 0
