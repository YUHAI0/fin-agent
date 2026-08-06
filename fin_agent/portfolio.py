import datetime
import json
import os
from io import StringIO
from typing import Dict, List, Optional
import pandas as pd
from fin_agent.config import Config
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

DEFAULT_PORTFOLIO_ID = "default"
DEFAULT_PORTFOLIO_NAME = "默认组合"
SCHEMA_VERSION = 2


class PortfolioManager:
    def __init__(self, file_path: str = None):
        if file_path:
            self.file_path = file_path
        else:
            config_dir = Config.get_config_dir()
            os.makedirs(config_dir, exist_ok=True)
            self.file_path = os.path.join(config_dir, "portfolio.json")

        self.data = self._load_portfolio()

    def _empty(self):
        return {
            "version": SCHEMA_VERSION,
            "active_portfolio_id": DEFAULT_PORTFOLIO_ID,
            "portfolios": {
                DEFAULT_PORTFOLIO_ID: {
                    "name": DEFAULT_PORTFOLIO_NAME,
                    "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
                    "positions": {},
                }
            },
        }

    def _migrate(self, raw):
        """v1（顶层 positions）→ v2（多组合）。旧持仓包装进默认组合，字段原样保留。"""
        migrated = self._empty()
        positions = raw.get("positions") or {}
        for ts_code, item in positions.items():
            migrated["portfolios"][DEFAULT_PORTFOLIO_ID]["positions"][ts_code] = {
                "amount": item.get("amount", 0),
                "cost": item.get("cost", 0.0),
                "bought_at": item.get("bought_at", ""),
                "note": item.get("note", ""),
            }
        return migrated

    def _load_portfolio(self):
        if not os.path.exists(self.file_path):
            return self._empty()
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            return self._empty()

        if not isinstance(raw, dict):
            return self._empty()

        if raw.get("version") == SCHEMA_VERSION and "portfolios" in raw:
            return raw

        migrated = self._migrate(raw)
        self.data = migrated
        self._save_portfolio()
        return migrated

    def _save_portfolio(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def _resolve(self, portfolio_id=None):
        """按 id 或名称定位组合；不传则用当前活动组合。"""
        portfolios = self.data.get("portfolios", {})
        if not portfolio_id:
            portfolio_id = self.data.get("active_portfolio_id") or DEFAULT_PORTFOLIO_ID

        if portfolio_id in portfolios:
            return portfolio_id, portfolios[portfolio_id]

        for pid, portfolio in portfolios.items():
            if portfolio.get("name") == portfolio_id:
                return pid, portfolio

        raise ValueError(f"找不到组合：{portfolio_id}")

    def add_position(self, ts_code: str, amount: int, price: float):
        """
        Add a position (buy).
        :param ts_code: Stock code
        :param amount: Quantity (positive)
        :param price: Cost per share
        """
        if amount <= 0 or price <= 0:
            return "Error: Amount and price must be positive."

        positions = self.holdings.get("positions", {})
        
        if ts_code in positions:
            # Average cost
            current_amount = positions[ts_code]["amount"]
            current_cost = positions[ts_code]["cost"]
            total_cost = current_amount * current_cost + amount * price
            new_amount = current_amount + amount
            new_cost = total_cost / new_amount
            
            positions[ts_code]["amount"] = new_amount
            positions[ts_code]["cost"] = new_cost
        else:
            positions[ts_code] = {
                "amount": amount,
                "cost": price
            }
        
        self.holdings["positions"] = positions
        self._save_portfolio()
        return f"Successfully added {amount} shares of {ts_code} at {price:.2f}."

    def remove_position(self, ts_code: str, amount: int, price: float):
        """
        Remove a position (sell).
        :param ts_code: Stock code
        :param amount: Quantity to sell
        :param price: Sell price
        """
        positions = self.holdings.get("positions", {})
        
        if ts_code not in positions:
            return f"Error: You do not hold {ts_code}."
        
        current_amount = positions[ts_code]["amount"]
        
        if amount > current_amount:
            return f"Error: Insufficient shares. You have {current_amount}, trying to sell {amount}."
            
        # Update position
        if amount == current_amount:
            del positions[ts_code]
        else:
            positions[ts_code]["amount"] = current_amount - amount
            
        self.holdings["positions"] = positions
        self._save_portfolio()
        return f"Successfully sold {amount} shares of {ts_code} at {price:.2f}."

    def get_portfolio_status(self):
        """
        Get current portfolio status with real-time valuation.
        """
        # Import here to avoid circular dependency
        from fin_agent.tools.tushare_tools import get_realtime_price
        
        positions = self.holdings.get("positions", {})
        if not positions:
            return "Portfolio is empty."
            
        report = []
        total_market_value = 0.0
        total_cost_value = 0.0

        # Helper: avoid hanging forever on realtime quote (network/provider can stall on Windows)
        def get_realtime_price_safe(code: str, timeout_sec: float = 3.0):
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(get_realtime_price, code)
                try:
                    return fut.result(timeout=timeout_sec)
                except FuturesTimeoutError:
                    return "Error: Realtime quote timeout"
                except Exception as e:
                    return f"Error: {e}"
        
        for ts_code, data in positions.items():
            amount = data["amount"]
            cost = data["cost"]
            
            # Fetch real-time price
            # get_realtime_price returns a JSON string, we need to parse it
            try:
                price_json = get_realtime_price_safe(ts_code)
                if "Error" in price_json or "No realtime data" in price_json:
                    current_price = cost # Fallback to cost if fails? Or 0? Let's keep cost to avoid panic, but mark it
                    current_price_str = f"{cost} (Est.)"
                else:
                    df = pd.read_json(StringIO(price_json), orient='records')
                    if not df.empty:
                        # Tushare realtime returns 'price', 'bid', 'ask', etc.
                        # Sometimes 'price' is the current price.
                        current_price = float(df.iloc[0]['price'])
                        current_price_str = str(current_price)
                    else:
                        current_price = cost
                        current_price_str = f"{cost} (Est.)"
            except:
                current_price = cost
                current_price_str = f"{cost} (Est.)"
                
            market_value = amount * current_price
            cost_value = amount * cost
            pnl = market_value - cost_value
            pnl_pct = (pnl / cost_value) * 100 if cost_value != 0 else 0
            
            total_market_value += market_value
            total_cost_value += cost_value
            
            report.append({
                "ts_code": ts_code,
                "amount": amount,
                "cost": round(cost, 2),
                "current_price": round(current_price, 2),
                "market_value": round(market_value, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2)
            })
            
        total_pnl = total_market_value - total_cost_value
        total_pnl_pct = (total_pnl / total_cost_value) * 100 if total_cost_value != 0 else 0
        
        summary = {
            "positions": report,
            "total_market_value": round(total_market_value, 2),
            "total_cost_value": round(total_cost_value, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2)
        }
        
        return summary

    def clear_portfolio(self):
        self.holdings = {"positions": {}}
        self._save_portfolio()
        return "Portfolio cleared."
