from __future__ import annotations

from typing import List, Type

from fin_agent.backtest.base import BacktestStrategy

_REGISTRY: dict[str, Type[BacktestStrategy]] = {}


def register_strategy(cls: Type[BacktestStrategy]) -> Type[BacktestStrategy]:
    if not issubclass(cls, BacktestStrategy):
        raise TypeError(f"{cls} must subclass BacktestStrategy")
    key = getattr(cls, "name", None)
    if not key or not isinstance(key, str):
        raise ValueError(f"{cls.__name__} must define non-empty string class attr 'name'")
    _REGISTRY[key] = cls
    return cls


def get_strategy_class(strategy_type: str) -> Type[BacktestStrategy]:
    cls = _REGISTRY.get(strategy_type)
    if cls is None:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown strategy '{strategy_type}'. Registered: {known}")
    return cls


def list_registered_strategies() -> List[str]:
    return sorted(_REGISTRY.keys())
