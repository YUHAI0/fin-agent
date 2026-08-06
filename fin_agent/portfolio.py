import datetime
import json
import os
import uuid
from typing import Dict, List, Optional

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

    # ---------- 组合级操作 ----------

    def list_portfolios(self):
        items = []
        for pid, portfolio in self.data.get("portfolios", {}).items():
            items.append({
                "id": pid,
                "name": portfolio.get("name", pid),
                "created_at": portfolio.get("created_at", ""),
                "position_count": len(portfolio.get("positions", {})),
            })
        items.sort(key=lambda x: x["created_at"])
        return {
            "active_portfolio_id": self.data.get("active_portfolio_id", DEFAULT_PORTFOLIO_ID),
            "portfolios": items,
        }

    def create_portfolio(self, name):
        name = (name or "").strip()
        if not name:
            return "Error: 组合名称不能为空。"
        for portfolio in self.data["portfolios"].values():
            if portfolio.get("name") == name:
                return f"Error: 已存在同名组合「{name}」。"

        pid = str(uuid.uuid4())
        self.data["portfolios"][pid] = {
            "name": name,
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "positions": {},
        }
        self._save_portfolio()
        return pid

    def rename_portfolio(self, portfolio_id, name):
        name = (name or "").strip()
        if not name:
            return "Error: 组合名称不能为空。"
        pid, portfolio = self._resolve(portfolio_id)
        portfolio["name"] = name
        self._save_portfolio()
        return f"组合已重命名为「{name}」。"

    def delete_portfolio(self, portfolio_id):
        if len(self.data.get("portfolios", {})) <= 1:
            return "Error: 至少需要保留一个组合，无法删除最后一个。"
        pid, _ = self._resolve(portfolio_id)
        del self.data["portfolios"][pid]
        if self.data.get("active_portfolio_id") == pid:
            self.data["active_portfolio_id"] = next(iter(self.data["portfolios"]))
        self._save_portfolio()
        return "组合已删除。"

    def set_active_portfolio(self, portfolio_id):
        pid, _ = self._resolve(portfolio_id)
        self.data["active_portfolio_id"] = pid
        self._save_portfolio()
        return "已切换当前组合。"

    # ---------- 持仓级操作 ----------

    def add_position(self, ts_code, amount, price, portfolio=None):
        """买入：已持有则按加权平均更新成本。供 LLM 工具调用。"""
        if amount <= 0 or price <= 0:
            return "Error: Amount and price must be positive."

        _, target = self._resolve(portfolio)
        positions = target.setdefault("positions", {})

        if ts_code in positions:
            current_amount = positions[ts_code]["amount"]
            current_cost = positions[ts_code]["cost"]
            total_cost = current_amount * current_cost + amount * price
            new_amount = current_amount + amount
            positions[ts_code]["amount"] = new_amount
            positions[ts_code]["cost"] = total_cost / new_amount
        else:
            positions[ts_code] = {
                "amount": amount,
                "cost": price,
                "bought_at": datetime.date.today().isoformat(),
                "note": "",
            }

        self._save_portfolio()
        return f"Successfully added {amount} shares of {ts_code} at {price:.2f}."

    def create_position(self, ts_code, amount, cost, bought_at="", note="", portfolio=None):
        """新建持仓。已存在则拒绝，引导使用编辑。供 UI 调用。"""
        if amount <= 0 or cost <= 0:
            return "Error: 数量与成本必须为正数。"

        _, target = self._resolve(portfolio)
        positions = target.setdefault("positions", {})
        if ts_code in positions:
            return f"Error: 该组合中已存在 {ts_code}，请使用编辑功能修改。"

        positions[ts_code] = {
            "amount": amount,
            "cost": cost,
            "bought_at": bought_at or "",
            "note": note or "",
        }
        self._save_portfolio()
        return f"已添加持仓 {ts_code}。"

    def update_position(self, ts_code, amount, cost, bought_at="", note="", portfolio=None):
        """编辑持仓：直接覆盖，不做加权平均。供 UI 调用。"""
        if amount <= 0 or cost <= 0:
            return "Error: 数量与成本必须为正数。"

        _, target = self._resolve(portfolio)
        positions = target.setdefault("positions", {})
        if ts_code not in positions:
            return f"Error: 该组合中不存在 {ts_code}。"

        positions[ts_code] = {
            "amount": amount,
            "cost": cost,
            "bought_at": bought_at or "",
            "note": note or "",
        }
        self._save_portfolio()
        return f"已更新持仓 {ts_code}。"

    def remove_position(self, ts_code, amount, price, portfolio=None):
        """卖出指定数量。供 LLM 工具调用。"""
        _, target = self._resolve(portfolio)
        positions = target.setdefault("positions", {})

        if ts_code not in positions:
            return f"Error: You do not hold {ts_code}."

        current_amount = positions[ts_code]["amount"]
        if amount > current_amount:
            return f"Error: Insufficient shares. You have {current_amount}, trying to sell {amount}."

        if amount == current_amount:
            del positions[ts_code]
        else:
            positions[ts_code]["amount"] = current_amount - amount

        self._save_portfolio()
        return f"Successfully sold {amount} shares of {ts_code} at {price:.2f}."

    def delete_position(self, ts_code, portfolio=None):
        """整条删除持仓。供 UI 调用。"""
        _, target = self._resolve(portfolio)
        positions = target.setdefault("positions", {})
        if ts_code not in positions:
            return f"Error: 该组合中不存在 {ts_code}。"
        del positions[ts_code]
        self._save_portfolio()
        return f"已删除持仓 {ts_code}。"

    def clear_portfolio(self, portfolio=None):
        _, target = self._resolve(portfolio)
        target["positions"] = {}
        self._save_portfolio()
        return "Portfolio cleared."

    # ---------- 估值 ----------

    def get_portfolio_status(self, portfolio=None):
        """一次性批量取现价，计算市值与盈亏。"""
        pid, target = self._resolve(portfolio)
        positions = target.get("positions", {})
        if not positions:
            return {
                "portfolio_id": pid,
                "portfolio_name": target.get("name", pid),
                "positions": [],
                "total_market_value": 0.0,
                "total_cost_value": 0.0,
                "total_pnl": 0.0,
                "total_pnl_pct": 0.0,
            }

        price_map = self._batch_prices(list(positions.keys()))

        report = []
        total_market_value = 0.0
        total_cost_value = 0.0

        for ts_code, item in positions.items():
            amount = item["amount"]
            cost = item["cost"]
            quote = price_map.get(ts_code)
            current_price = quote["price"] if quote and quote["price"] else cost
            estimated = quote is None or not quote["price"]

            market_value = amount * current_price
            cost_value = amount * cost
            pnl = market_value - cost_value

            total_market_value += market_value
            total_cost_value += cost_value

            report.append({
                "ts_code": ts_code,
                "name": (quote or {}).get("name") or ts_code,
                "amount": amount,
                "cost": round(cost, 2),
                "current_price": round(current_price, 2),
                "estimated": estimated,
                "market_value": round(market_value, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl / cost_value * 100, 2) if cost_value else 0.0,
                "bought_at": item.get("bought_at", ""),
                "note": item.get("note", ""),
            })

        total_pnl = total_market_value - total_cost_value
        return {
            "portfolio_id": pid,
            "portfolio_name": target.get("name", pid),
            "positions": report,
            "total_market_value": round(total_market_value, 2),
            "total_cost_value": round(total_cost_value, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl / total_cost_value * 100, 2) if total_cost_value else 0.0,
        }

    def _batch_prices(self, ts_codes, timeout_sec=8.0):
        """批量取实时行情，整体超时后按空结果处理（估值回落到成本价）。"""
        from fin_agent.datasources import get_provider

        def fetch():
            df = get_provider().get_realtime_price(ts_codes)
            return {
                row["ts_code"]: {"price": float(row["price"] or 0), "name": row.get("name")}
                for _, row in df.iterrows()
            }

        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(fetch)
            try:
                return future.result(timeout=timeout_sec)
            except FuturesTimeoutError:
                return {}
            except Exception:
                return {}
