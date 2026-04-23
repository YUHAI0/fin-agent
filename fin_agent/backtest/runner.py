from __future__ import annotations

import json
from datetime import datetime, timedelta

import fin_agent.backtest.strategies  # noqa: F401  # 内置策略注册
from fin_agent.backtest.engine import BacktestEngine


def run_backtest(ts_code, strategy="ma_cross", start_date=None, end_date=None, params=None):
    """
    Wrapper for Tool usage.
    params: JSON string or dict of strategy parameters.
    """
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
    if not end_date:
        end_date = datetime.now().strftime("%Y%m%d")

    strategy_config = {"type": strategy}
    if params:
        if isinstance(params, str):
            try:
                strategy_config.update(json.loads(params))
            except json.JSONDecodeError:
                pass
        elif isinstance(params, dict):
            strategy_config.update(params)

    engine = BacktestEngine()
    try:
        result = engine.run(ts_code, start_date, end_date, strategy_config)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error running backtest: {str(e)}"
