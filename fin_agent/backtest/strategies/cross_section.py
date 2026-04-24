from __future__ import annotations

from typing import Any, Mapping, Optional

import pandas as pd

from fin_agent.backtest.base import BacktestStrategy
from fin_agent.backtest.registry import register_strategy


@register_strategy
class CrossSectionMomentumStrategy(BacktestStrategy):
    """动量截面/多标的排序：当前引擎为单标的，注册占位并由引擎直接返回错误。"""

    name = "cross_section_momentum"
    requires_multi_asset = True

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        return df

    def signal(self, row, prev_row, params: Mapping[str, Any], context: Optional[Mapping[str, Any]] = None) -> int:
        return 0
