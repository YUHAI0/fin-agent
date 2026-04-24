from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import tushare as ts

from fin_agent.config import Config
from fin_agent.backtest.registry import get_strategy_class


class BacktestEngine:
    def __init__(self, initial_capital=100000, commission=0.0003):
        self.initial_capital = initial_capital
        self.commission = commission
        self.cash = initial_capital
        self.position = 0
        self.history = []
        self.portfolio_values = []
        self.entry_price: float | None = None

    def _fetch_data(self, ts_code, start_date, end_date):
        try:
            ts.set_token(Config.TUSHARE_TOKEN)
            pro = ts.pro_api()
            warmup_start = (datetime.strptime(start_date, "%Y%m%d") - timedelta(days=400)).strftime("%Y%m%d")
            df = pro.daily(ts_code=ts_code, start_date=warmup_start, end_date=end_date)
            if df.empty:
                raise ValueError(f"No data found for {ts_code}")
            return df.sort_values("trade_date", ascending=True).reset_index(drop=True)
        except Exception as e:
            raise e

    def run(self, ts_code, start_date, end_date, strategy_config):
        strategy_type = strategy_config.get("type", "ma_cross")
        strategy_cls = get_strategy_class(strategy_type)
        if getattr(strategy_cls, "requires_multi_asset", False):
            return {
                "error": "该策略需要多标的截面回测，当前 run_backtest 仅支持单标的 Tushare 日线。",
                "strategy": strategy_type,
            }

        strategy = strategy_cls()

        df = self._fetch_data(ts_code, start_date, end_date)
        df = strategy.enrich(df.copy(), strategy_config)

        mask = df["trade_date"] >= start_date
        if not mask.any():
            return {"error": "No data in requested date range"}

        start_idx = mask.idxmax()
        if start_idx > 0:
            start_idx -= 1

        prev_row = None
        for _, row in df.iloc[start_idx:].iterrows():
            is_trading_period = row["trade_date"] >= start_date
            current_price = row["close"]

            if is_trading_period:
                total_value = self.cash + (self.position * current_price)
                self.portfolio_values.append({"trade_date": row["trade_date"], "value": total_value})

            context = {
                "position": self.position,
                "entry_price": self.entry_price,
                "cash": self.cash,
            }

            sig = 0
            if self.position > 0:
                sig = strategy.risk_exit(row, prev_row, strategy_config, context)
            if sig == 0:
                sig = strategy.signal(row, prev_row, strategy_config, context)

            if is_trading_period and sig != 0:
                if sig == 1 and self.cash > 0:
                    max_shares = int(self.cash / (current_price * (1 + self.commission)) / 100) * 100
                    sizing_ctx = {**context, "signal_row": row}
                    frac = float(strategy.position_fraction(strategy_config, sizing_ctx))
                    frac = max(0.0, min(1.0, frac))
                    max_shares = int(max_shares * frac / 100) * 100
                    if max_shares > 0:
                        cost = max_shares * current_price
                        comm = cost * self.commission
                        self.cash -= cost + comm
                        self.position += max_shares
                        self.entry_price = current_price
                        self.history.append(
                            {
                                "date": row["trade_date"],
                                "action": "BUY",
                                "price": current_price,
                                "shares": max_shares,
                                "commission": comm,
                            }
                        )
                elif sig == -1 and self.position > 0:
                    revenue = self.position * current_price
                    comm = revenue * self.commission
                    self.cash += revenue - comm
                    self.history.append(
                        {
                            "date": row["trade_date"],
                            "action": "SELL",
                            "price": current_price,
                            "shares": self.position,
                            "commission": comm,
                        }
                    )
                    self.position = 0
                    self.entry_price = None

            prev_row = row

        final_value = self.cash + (self.position * df.iloc[-1]["close"])
        total_return = (final_value - self.initial_capital) / self.initial_capital

        pv_df = pd.DataFrame(self.portfolio_values)
        if not pv_df.empty:
            pv_df["cummax"] = pv_df["value"].cummax()
            pv_df["drawdown"] = (pv_df["cummax"] - pv_df["value"]) / pv_df["cummax"]
            max_drawdown = pv_df["drawdown"].max()
        else:
            max_drawdown = 0

        equity_curve = []
        for pv in self.portfolio_values:
            td = pv.get("trade_date")
            val = float(pv.get("value", 0))
            try:
                td_str = str(int(float(td)))
            except (TypeError, ValueError):
                td_str = str(td) if td is not None else ""
            equity_curve.append(
                {
                    "trade_date": td_str,
                    "value": round(val, 2),
                    "return_pct": round((val / self.initial_capital - 1) * 100, 4),
                }
            )

        return {
            "ts_code": ts_code,
            "strategy": strategy_config.get("type"),
            "initial_capital": self.initial_capital,
            "final_value": round(final_value, 2),
            "total_return_pct": round(total_return * 100, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "trades_count": len(self.history),
            "trades": self.history[-5:],
            "equity_curve": equity_curve,
        }


import fin_agent.backtest.strategies  # noqa: E402, F401  # 仅从 engine 导入时也注册内置策略
