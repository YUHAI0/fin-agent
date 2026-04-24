from __future__ import annotations

from typing import Any, Mapping, Optional

import pandas as pd

from fin_agent.backtest.base import BacktestStrategy
from fin_agent.backtest.registry import register_strategy
from fin_agent.tools.technical_indicators import calculate_obv, calculate_vwap_ma


@register_strategy
class VolumeBreakoutStrategy(BacktestStrategy):
    """放量突破：收盘突破 N 日高且成交量大于 M 日均量倍数。"""

    name = "volume_breakout"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        n = int(params.get("breakout_period", 20))
        m = int(params.get("vol_ma_period", 20))
        exit_n = int(params.get("exit_period", n))
        df["range_high"] = df["high"].rolling(n).max().shift(1)
        df["range_low"] = df["low"].rolling(exit_n).min().shift(1)
        df["vol_ma"] = df["vol"].rolling(m).mean()
        return df

    def signal(self, row, prev_row, params: Mapping[str, Any], context: Optional[Mapping[str, Any]] = None) -> int:
        if prev_row is None:
            return 0
        mult = float(params.get("volume_mult", 1.5))
        if self._na(row.get("range_high"), row.get("range_low"), row.get("vol_ma")):
            return 0
        if row["vol"] > mult * row["vol_ma"] and row["close"] > row["range_high"]:
            return 1
        if row["close"] < row["range_low"]:
            return -1
        return 0


@register_strategy
class ObvCrossStrategy(BacktestStrategy):
    """OBV 上穿/下穿其均线。"""

    name = "obv_cross"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        calculate_obv(df)
        ma_p = int(params.get("obv_ma_period", 30))
        df["obv_ma"] = df["obv"].rolling(ma_p).mean()
        return df

    def signal(self, row, prev_row, params: Mapping[str, Any], context: Optional[Mapping[str, Any]] = None) -> int:
        if prev_row is None:
            return 0
        if self._na(row.get("obv"), row.get("obv_ma"), prev_row.get("obv"), prev_row.get("obv_ma")):
            return 0
        if prev_row["obv"] <= prev_row["obv_ma"] and row["obv"] > row["obv_ma"]:
            return 1
        if prev_row["obv"] >= prev_row["obv_ma"] and row["obv"] < row["obv_ma"]:
            return -1
        return 0


@register_strategy
class VwapDeviationStrategy(BacktestStrategy):
    """日频 VWAP 近似线偏离：低于均线过多买入，回到均线上方卖出。"""

    name = "vwap_deviation"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        return calculate_vwap_ma(df, period=int(params.get("period", 20)))

    def signal(self, row, prev_row, params: Mapping[str, Any], context: Optional[Mapping[str, Any]] = None) -> int:
        if prev_row is None:
            return 0
        dev = float(params.get("deviation", 0.008))
        if self._na(row.get("vwap_ma"), row.get("close")):
            return 0
        v = float(row["vwap_ma"])
        if row["close"] < v * (1.0 - dev):
            return 1
        if row["close"] > v:
            return -1
        return 0
