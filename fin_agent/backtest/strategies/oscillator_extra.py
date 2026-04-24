from __future__ import annotations

from typing import Any, Mapping, Optional

import pandas as pd

from fin_agent.backtest.base import BacktestStrategy
from fin_agent.backtest.registry import register_strategy
from fin_agent.tools.technical_indicators import calculate_cci, calculate_stochastic, calculate_williams_r


@register_strategy
class CciStrategy(BacktestStrategy):
    """CCI：自下而上穿越 -100 买入，自上而下穿越 +100 卖出。"""

    name = "cci"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        return calculate_cci(df, period=int(params.get("period", 20)))

    def signal(self, row, prev_row, params: Mapping[str, Any], context: Optional[Mapping[str, Any]] = None) -> int:
        if prev_row is None:
            return 0
        if self._na(row.get("cci"), prev_row.get("cci")):
            return 0
        low_th = float(params.get("oversold", -100.0))
        high_th = float(params.get("overbought", 100.0))
        if prev_row["cci"] <= low_th and row["cci"] > low_th:
            return 1
        if prev_row["cci"] >= high_th and row["cci"] < high_th:
            return -1
        return 0


@register_strategy
class WilliamsRStrategy(BacktestStrategy):
    """Williams %R：超卖区上穿买入，超买区下穿卖出。"""

    name = "williams_r"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        return calculate_williams_r(df, period=int(params.get("period", 14)))

    def signal(self, row, prev_row, params: Mapping[str, Any], context: Optional[Mapping[str, Any]] = None) -> int:
        if prev_row is None:
            return 0
        if self._na(row.get("willr"), prev_row.get("willr")):
            return 0
        os_th = float(params.get("oversold", -80.0))
        ob_th = float(params.get("overbought", -20.0))
        if prev_row["willr"] <= os_th and row["willr"] > os_th:
            return 1
        if prev_row["willr"] >= ob_th and row["willr"] < ob_th:
            return -1
        return 0


@register_strategy
class StochasticStrategy(BacktestStrategy):
    """随机指标 %K/%D 金叉死叉（列 st_k / st_d），结合超卖超买区。"""

    name = "stochastic"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        k_p = int(params.get("k_period", 14))
        d_p = int(params.get("d_period", 3))
        return calculate_stochastic(df, k_period=k_p, d_period=d_p)

    def signal(self, row, prev_row, params: Mapping[str, Any], context: Optional[Mapping[str, Any]] = None) -> int:
        if prev_row is None:
            return 0
        if self._na(row.get("st_k"), row.get("st_d"), prev_row.get("st_k"), prev_row.get("st_d")):
            return 0
        os_lv = float(params.get("oversold_level", 30.0))
        ob_lv = float(params.get("overbought_level", 70.0))
        gold = prev_row["st_k"] <= prev_row["st_d"] and row["st_k"] > row["st_d"] and row["st_k"] < os_lv + 5
        dead = prev_row["st_k"] >= prev_row["st_d"] and row["st_k"] < row["st_d"] and row["st_k"] > ob_lv - 5
        if gold:
            return 1
        if dead:
            return -1
        return 0
