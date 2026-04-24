from __future__ import annotations

from typing import Any, Mapping, Optional

import pandas as pd

from fin_agent.backtest.base import BacktestStrategy
from fin_agent.backtest.registry import register_strategy


@register_strategy
class TripleMaStrategy(BacktestStrategy):
    """三均线：短上穿中且中在长期之上做多；短下穿中且中在长期之下平仓。"""

    name = "triple_ma"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        s = int(params.get("short_window", 5))
        m = int(params.get("mid_window", 20))
        l = int(params.get("long_window", 60))
        df["short_ma"] = df["close"].rolling(s).mean()
        df["mid_ma"] = df["close"].rolling(m).mean()
        df["long_ma"] = df["close"].rolling(l).mean()
        return df

    def signal(self, row, prev_row, params: Mapping[str, Any], context: Optional[Mapping[str, Any]] = None) -> int:
        if prev_row is None:
            return 0
        for k in ("short_ma", "mid_ma", "long_ma"):
            if self._na(row.get(k), prev_row.get(k)):
                return 0
        golden = prev_row["short_ma"] <= prev_row["mid_ma"] and row["short_ma"] > row["mid_ma"]
        aligned = row["mid_ma"] > row["long_ma"]
        dead = prev_row["short_ma"] >= prev_row["mid_ma"] and row["short_ma"] < row["mid_ma"]
        if golden and aligned:
            return 1
        if dead:
            return -1
        return 0
