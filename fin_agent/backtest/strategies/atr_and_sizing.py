from __future__ import annotations

from typing import Any, Mapping, Optional

import pandas as pd

from fin_agent.backtest.base import BacktestStrategy
from fin_agent.backtest.registry import register_strategy
from fin_agent.tools.technical_indicators import calculate_atr


@register_strategy
class MaCrossAtrStopStrategy(BacktestStrategy):
    """双均线金叉死叉 + 持仓 ATR 倍数硬止损。"""

    name = "ma_cross_atr_stop"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        short_window = int(params.get("short_window", 5))
        long_window = int(params.get("long_window", 20))
        df["short_ma"] = df["close"].rolling(window=short_window).mean()
        df["long_ma"] = df["close"].rolling(window=long_window).mean()
        return calculate_atr(df, period=int(params.get("atr_period", 14)))

    def signal(self, row, prev_row, params: Mapping[str, Any], context: Optional[Mapping[str, Any]] = None) -> int:
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

    def risk_exit(
        self,
        row: pd.Series,
        prev_row: Optional[pd.Series],
        params: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> int:
        mult = float(params.get("atr_stop_mult", 2.0))
        ep = context.get("entry_price")
        if ep is None or self._na(row.get("atr")):
            return 0
        if row["close"] < float(ep) - mult * float(row["atr"]):
            return -1
        return 0


@register_strategy
class VolTargetMaCrossStrategy(BacktestStrategy):
    """双均线信号 + 按 ATR 占价比动态缩小仓位（波动目标近似）。"""

    name = "vol_target_ma_cross"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        short_window = int(params.get("short_window", 5))
        long_window = int(params.get("long_window", 20))
        df["short_ma"] = df["close"].rolling(window=short_window).mean()
        df["long_ma"] = df["close"].rolling(window=long_window).mean()
        return calculate_atr(df, period=int(params.get("atr_period", 14)))

    def signal(self, row, prev_row, params: Mapping[str, Any], context: Optional[Mapping[str, Any]] = None) -> int:
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

    def position_fraction(self, params: Mapping[str, Any], context: Mapping[str, Any]) -> float:
        sr = context.get("signal_row")
        if sr is None or self._na(sr.get("atr"), sr.get("close")):
            return 1.0
        risk_pct = float(params.get("risk_budget_pct", 2.0)) / 100.0
        atr_frac = max(1e-6, float(sr["atr"]) / float(sr["close"]))
        cap = float(params.get("max_fraction", 1.0))
        return min(cap, risk_pct / atr_frac)


@register_strategy
class KellyStyleMaCrossStrategy(BacktestStrategy):
    """双均线信号 + 固定仓位比例（简化固定风险/凯利上限）。"""

    name = "kelly_ma_cross"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        short_window = int(params.get("short_window", 5))
        long_window = int(params.get("long_window", 20))
        df["short_ma"] = df["close"].rolling(window=short_window).mean()
        df["long_ma"] = df["close"].rolling(window=long_window).mean()
        return df

    def signal(self, row, prev_row, params: Mapping[str, Any], context: Optional[Mapping[str, Any]] = None) -> int:
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

    def position_fraction(self, params: Mapping[str, Any], context: Mapping[str, Any]) -> float:
        return max(0.05, min(1.0, float(params.get("equity_fraction", 0.35))))
