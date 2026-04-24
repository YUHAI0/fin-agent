from __future__ import annotations

from typing import Any, Mapping, Optional

import pandas as pd

from fin_agent.backtest.base import BacktestStrategy
from fin_agent.backtest.registry import register_strategy
from fin_agent.tools.technical_indicators import calculate_rsi


@register_strategy
class RsiMa200Strategy(BacktestStrategy):
    """RSI 反转信号仅在价格在长期均线之上时做多（趋势过滤）。"""

    name = "rsi_ma200"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        ma_w = int(params.get("ma_window", 200))
        rsi_w = int(params.get("window", 14))
        df["trend_ma"] = df["close"].rolling(ma_w).mean()
        return calculate_rsi(df, period=rsi_w)

    def signal(self, row, prev_row, params: Mapping[str, Any], context: Optional[Mapping[str, Any]] = None) -> int:
        if prev_row is None:
            return 0
        lower = int(params.get("lower", 30))
        upper = int(params.get("upper", 70))
        if self._na(row.get("rsi"), prev_row.get("rsi"), row.get("trend_ma"), prev_row.get("trend_ma"), row.get("close")):
            return 0
        above_ma = float(row["close"]) > float(row["trend_ma"])
        if above_ma and prev_row["rsi"] < lower and row["rsi"] >= lower:
            return 1
        if prev_row["rsi"] > upper and row["rsi"] <= upper:
            return -1
        if prev_row["close"] >= prev_row["trend_ma"] and row["close"] < row["trend_ma"]:
            return -1
        return 0
