from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional

import pandas as pd


class BacktestStrategy(ABC):
    """单日 bar、全仓买卖语义：enrich 加列；signal 返回 1/-1/0；可选 risk_exit、position_fraction。"""

    name: str
    requires_multi_asset: bool = False

    @abstractmethod
    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        """在 df 上增加回测所需列并返回。"""

    @abstractmethod
    def signal(
        self,
        row: pd.Series,
        prev_row: Optional[pd.Series],
        params: Mapping[str, Any],
        context: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """prev_row 为 None 时应返回 0。context 含 position, entry_price, cash 等。"""

    def risk_exit(
        self,
        row: pd.Series,
        prev_row: Optional[pd.Series],
        params: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> int:
        """持仓时优先判断：返回 -1 表示强制平仓，否则 0。"""
        return 0

    def position_fraction(self, params: Mapping[str, Any], context: Mapping[str, Any]) -> float:
        """买入时用：实际仓位 = floor(满仓股数 * fraction, 一手)。默认 1.0。"""
        return float(params.get("position_fraction", 1.0))

    @classmethod
    def _na(cls, *values) -> bool:
        return any(pd.isna(v) for v in values)
