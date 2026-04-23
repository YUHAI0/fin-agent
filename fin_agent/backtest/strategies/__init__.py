"""内置策略：导入子模块即完成 register_strategy 注册。"""

from fin_agent.backtest.strategies import adx_macd  # noqa: F401
from fin_agent.backtest.strategies import atr_and_sizing  # noqa: F401
from fin_agent.backtest.strategies import boll  # noqa: F401
from fin_agent.backtest.strategies import cross_section  # noqa: F401
from fin_agent.backtest.strategies import donchian  # noqa: F401
from fin_agent.backtest.strategies import ema_sma_bias  # noqa: F401
from fin_agent.backtest.strategies import kdj  # noqa: F401
from fin_agent.backtest.strategies import ma_cross  # noqa: F401
from fin_agent.backtest.strategies import macd  # noqa: F401
from fin_agent.backtest.strategies import momentum_roc  # noqa: F401
from fin_agent.backtest.strategies import oscillator_extra  # noqa: F401
from fin_agent.backtest.strategies import rsi  # noqa: F401
from fin_agent.backtest.strategies import rsi_ma200  # noqa: F401
from fin_agent.backtest.strategies import triple_ma  # noqa: F401
from fin_agent.backtest.strategies import volume_price  # noqa: F401
