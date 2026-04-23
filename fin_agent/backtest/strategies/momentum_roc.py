from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from fin_agent.backtest.base import BacktestStrategy
from fin_agent.backtest.registry import register_strategy


@register_strategy
class MomentumRocStrategy(BacktestStrategy):
    name = "momentum_roc"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        roc_window = int(params.get("roc_window", 10))
        df["roc"] = (df["close"] / df["close"].shift(roc_window) - 1) * 100
        return df

    def signal(self, row, prev_row, params: Mapping[str, Any], context=None) -> int:
        if prev_row is None:
            return 0
        if self._na(row.get("roc"), prev_row.get("roc")):
            return 0
        if prev_row["roc"] < 0 <= row["roc"]:
            return 1
        if prev_row["roc"] > 0 >= row["roc"]:
            return -1
        return 0
