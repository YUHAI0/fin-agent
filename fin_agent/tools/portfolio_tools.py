import json

from fin_agent.portfolio import PortfolioManager


def _safe(call):
    """PortfolioManager._resolve 在组合不存在时抛 ValueError，转成可读文本回给模型。"""
    try:
        return call()
    except ValueError as e:
        return f"Error: {e}"


def add_portfolio_position(ts_code, amount, price, portfolio=None):
    """
    Add a stock to the simulated portfolio.
    :param ts_code: Stock code
    :param amount: Quantity
    :param price: Cost per share
    :param portfolio: Portfolio name or id, defaults to the active one
    """
    return _safe(lambda: PortfolioManager().add_position(
        ts_code, int(amount), float(price), portfolio=portfolio
    ))


def remove_portfolio_position(ts_code, amount, price, portfolio=None):
    """
    Remove a stock from the simulated portfolio.
    :param ts_code: Stock code
    :param amount: Quantity
    :param price: Sell price
    :param portfolio: Portfolio name or id, defaults to the active one
    """
    return _safe(lambda: PortfolioManager().remove_position(
        ts_code, int(amount), float(price), portfolio=portfolio
    ))


def get_portfolio_status(portfolio=None):
    """
    Get the current status of the portfolio, including real-time valuation and P&L.
    :param portfolio: Portfolio name or id, defaults to the active one
    """
    status = _safe(lambda: PortfolioManager().get_portfolio_status(portfolio=portfolio))
    if isinstance(status, str):
        return status
    return json.dumps(status, ensure_ascii=False, indent=2)


def clear_portfolio(portfolio=None):
    """
    Clear all positions in the portfolio.
    :param portfolio: Portfolio name or id, defaults to the active one
    """
    return _safe(lambda: PortfolioManager().clear_portfolio(portfolio=portfolio))

# Tool definitions
PORTFOLIO_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "add_portfolio_position",
            "description": "Add a stock position to the portfolio tracker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {
                        "type": "string",
                        "description": "Stock code (e.g., '000001.SZ')."
                    },
                    "amount": {
                        "type": "integer",
                        "description": "Number of shares."
                    },
                    "price": {
                        "type": "number",
                        "description": "Cost price per share."
                    },
                    "portfolio": {
                        "type": "string",
                        "description": "组合名称或 id。不传则作用于当前活动组合。"
                    },
                },
                "required": ["ts_code", "amount", "price"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remove_portfolio_position",
            "description": "Remove (sell) a stock position from the portfolio tracker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {
                        "type": "string",
                        "description": "Stock code."
                    },
                    "amount": {
                        "type": "integer",
                        "description": "Number of shares to sell."
                    },
                    "price": {
                        "type": "number",
                        "description": "Selling price per share."
                    },
                    "portfolio": {
                        "type": "string",
                        "description": "组合名称或 id。不传则作用于当前活动组合。"
                    },
                },
                "required": ["ts_code", "amount", "price"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_portfolio_status",
            "description": "Get the current portfolio holdings, value, and P&L status. Use this when the user asks about 'my portfolio', 'my holdings', '我的持仓', or '账户'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "portfolio": {
                        "type": "string",
                        "description": "组合名称或 id。不传则作用于当前活动组合。"
                    },
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "clear_portfolio",
            "description": "Clear all positions from the portfolio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "portfolio": {
                        "type": "string",
                        "description": "组合名称或 id。不传则作用于当前活动组合。"
                    },
                },
                "required": []
            }
        }
    }
]
