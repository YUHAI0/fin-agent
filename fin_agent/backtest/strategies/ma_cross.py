from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from fin_agent.backtest.base import BacktestStrategy
from fin_agent.backtest.registry import register_strategy


@register_strategy
class MACrossStrategy(BacktestStrategy):
    name = "ma_cross"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        short_window = int(params.get("short_window", 5))
        long_window = int(params.get("long_window", 20))
        df["short_ma"] = df["close"].rolling(window=short_window).mean()
        df["long_ma"] = df["close"].rolling(window=long_window).mean()
        return df

    def signal(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        if prev_row is None:
            return 0
        for key in ("short_ma", "long_ma"):
            if self._na(row.get(key), prev_row.get(key)):
                return 0
        if prev_row["short_ma"] <= prev_row["long_ma"] and row["short_ma"] > row["long_ma"]:
            return 1
        if prev_row["short_ma"] >= prev_row["long_ma"] and row["short_ma"] < row["long_ma"]:
            return -1
        return 0
