import json
from fin_agent.portfolio import PortfolioManager

# Global instance
pm = PortfolioManager()

def add_portfolio_position(ts_code, amount, price):
    """
    Add a stock to the simulated portfolio.
    :param ts_code: Stock code
    :param amount: Quantity
    :param price: Cost per share
    """
    return pm.add_position(ts_code, int(amount), float(price))

def remove_portfolio_position(ts_code, amount, price):
    """
    Remove a stock from the simulated portfolio.
    :param ts_code: Stock code
    :param amount: Quantity
    :param price: Sell price
    """
    return pm.remove_position(ts_code, int(amount), float(price))

def get_portfolio_status():
    """
    Get the current status of the portfolio, including real-time valuation and P&L.
    """
    status = pm.get_portfolio_status()
    if isinstance(status, str):
        return status
    return json.dumps(status, ensure_ascii=False, indent=2)

def clear_portfolio():
    """
    Clear all positions in the portfolio.
    """
    return pm.clear_portfolio()

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
                    }
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
                    }
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
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "clear_portfolio",
            "description": "Clear all positions and history from the portfolio.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

