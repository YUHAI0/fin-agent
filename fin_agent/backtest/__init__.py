"""回测子系统：策略通过 register_strategy 注册，引擎按 name 解析。"""

from fin_agent.backtest.base import BacktestStrategy
from fin_agent.backtest.engine import BacktestEngine
from fin_agent.backtest.registry import (
    get_strategy_class,
    list_registered_strategies,
    register_strategy,
)
from fin_agent.backtest.runner import run_backtest

# 注册内置策略（副作用：填充 registry）
import fin_agent.backtest.strategies  # noqa: E402, F401

__all__ = [
    "BacktestStrategy",
    "BacktestEngine",
    "run_backtest",
    "register_strategy",
    "get_strategy_class",
    "list_registered_strategies",
]
