from __future__ import annotations

from typing import Any, Mapping, Optional

import pandas as pd

from fin_agent.backtest.base import BacktestStrategy
from fin_agent.backtest.registry import register_strategy


@register_strategy
class EmaSmaBiasStrategy(BacktestStrategy):
    """EMA 与 SMA：价格上穿 EMA 且 EMA>SMA 且乖离达标买入；下穿 EMA 卖出。"""

    name = "ema_sma_bias"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        ema_span = int(params.get("ema_span", 12))
        sma_window = int(params.get("sma_window", 50))
        df["ema_fast"] = df["close"].ewm(span=ema_span, adjust=False).mean()
        df["sma_slow"] = df["close"].rolling(sma_window).mean()
        return df

    def signal(self, row, prev_row, params: Mapping[str, Any], context: Optional[Mapping[str, Any]] = None) -> int:
        if prev_row is None:
            return 0
        bias = float(params.get("bias_threshold", 0.0))
        if self._na(row.get("ema_fast"), row.get("sma_slow"), prev_row.get("close"), row.get("close")):
            return 0
        ema = float(row["ema_fast"])
        sma = float(row["sma_slow"])
        if ema <= sma:
            return 0
        bias_ok = (float(row["close"]) - ema) / (ema + 1e-12) >= bias
        cross_up = prev_row["close"] <= prev_row.get("ema_fast") and row["close"] > ema
        cross_dn = prev_row["close"] >= prev_row.get("ema_fast") and row["close"] < ema
        if cross_up and bias_ok:
            return 1
        if cross_dn:
            return -1
        return 0
