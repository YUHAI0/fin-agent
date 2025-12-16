import tushare as ts
import pandas as pd
from fin_agent.config import Config
import json
from datetime import datetime, timedelta
from fin_agent.tools.technical_indicators import get_technical_indicators, get_technical_patterns
from fin_agent.backtest import run_backtest
from fin_agent.tools.portfolio_tools import (
    PORTFOLIO_TOOLS_SCHEMA, 
    add_portfolio_position, 
    remove_portfolio_position, 
    get_portfolio_status, 
    clear_portfolio
)
from fin_agent.tools.scheduler_tools import (
    SCHEDULER_TOOLS_SCHEMA,
    add_price_alert,
    list_alerts,
    remove_alert,
    update_alert,
    reset_email_config
)
from fin_agent.tools.profile_tools import (
    PROFILE_TOOLS_SCHEMA,
    update_user_profile,
    get_user_profile
)

# Initialize Tushare - will be re-initialized when called if Config updates
def get_pro():
    ts.set_token(Config.TUSHARE_TOKEN)
    return ts.pro_api()

def get_current_time():
    """Get current date and time."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

def get_stock_basic(ts_code=None, name=None):
    """
    Get basic stock information.
    :param ts_code: Stock code (e.g., 000001.SZ)
    :param name: Stock name (e.g., 平安银行)
    :return: DataFrame or dict string
    """
    try:
        pro = get_pro()
        # If name is provided but ts_code is not, try to find ts_code
        if name and not ts_code:
            # Getting all stocks and filtering might be slow, but it's a simple way
            df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
            df = df[df['name'] == name]
            if df.empty:
                return f"Error: Stock with name '{name}' not found."
            return df.to_json(orient='records', force_ascii=False)
        
        # Otherwise use ts_code or just list all (limit to some reasonable amount if needed, but pro.stock_basic returns all usually)
        # To be safe for LLM, usually we query by specific code or just return error if both missing
        if not ts_code:
             return "Error: Please provide either ts_code or name."

        df = pro.stock_basic(ts_code=ts_code, fields='ts_code,symbol,name,area,industry,list_date')
        if df.empty:
            return f"Error: Stock code '{ts_code}' not found."
        return df.to_json(orient='records', force_ascii=False)
    except Exception as e:
        return f"Error fetching stock basic info: {str(e)}"

def get_daily_price(ts_code, start_date=None, end_date=None):
    """
    Get daily stock price.
    :param ts_code: Stock code
    :param start_date: Start date (YYYYMMDD)
    :param end_date: End date (YYYYMMDD)
    :return: JSON string
    """
    if not start_date:
        # Default to last 30 days
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y%m%d')

    try:
        pro = get_pro()
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df.empty:
            return f"No data found for {ts_code} between {start_date} and {end_date}."
        
        # Ensure data is sorted by date descending
        df = df.sort_values('trade_date', ascending=False)
        return df.to_json(orient='records', force_ascii=False)
    except Exception as e:
        return f"Error fetching daily price: {str(e)}"

def get_realtime_price(ts_code):
    """
    Get realtime stock price using legacy Tushare interface.
    :param ts_code: Stock code (e.g., 000001.SZ -> 000001 for legacy)
    :return: JSON string
    """
    try:
        # Legacy interface takes code without suffix usually, but let's check input
        code = ts_code.split('.')[0] if '.' in ts_code else ts_code
        
        df = ts.get_realtime_quotes(code)
        if df is None or df.empty:
            return f"No realtime data found for {ts_code}."
            
        # Add ts_code back for clarity
        df['ts_code'] = ts_code
        return df.to_json(orient='records', force_ascii=False)
    except Exception as e:
        return f"Error fetching realtime price: {str(e)}"

def get_daily_basic(ts_code, start_date=None, end_date=None):
    """
    Get daily basic indicators (PE, PB, turnover, etc.).
    """
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y%m%d')
        
    try:
        pro = get_pro()
        df = pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date, 
                            fields='ts_code,trade_date,close,turnover_rate,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,total_share,float_share,free_share,total_mv,circ_mv')
        if df.empty:
             return f"No daily basic data found for {ts_code}."
        df = df.sort_values('trade_date', ascending=False)
        return df.to_json(orient='records', force_ascii=False)
    except Exception as e:
        return f"Error fetching daily basic info: {str(e)}"

def get_income_statement(ts_code, start_date=None, end_date=None):
    """
    Get income statement data (Revenue, Profit).
    """
    if not start_date:
        # Last 2 years
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y%m%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y%m%d')
        
    try:
        pro = get_pro()
        df = pro.income(ts_code=ts_code, start_date=start_date, end_date=end_date,
                       fields='ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,total_revenue,revenue,total_profit,n_income,n_income_attr_p')
        if df.empty:
            return f"No income statement data found for {ts_code}."
        df = df.sort_values('end_date', ascending=False)
        return df.to_json(orient='records', force_ascii=False)
    except Exception as e:
        return f"Error fetching income statement: {str(e)}"

def get_index_daily(ts_code, start_date=None, end_date=None):
    """
    Get daily index market data (e.g. 000001.SH, 399001.SZ).
    """
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y%m%d')
    
    try:
        pro = get_pro()
        df = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df.empty:
            return f"No index data found for {ts_code}."
        df = df.sort_values('trade_date', ascending=False)
        return df.to_json(orient='records', force_ascii=False)
    except Exception as e:
        return f"Error fetching index daily: {str(e)}"

def get_moneyflow(ts_code, start_date=None, end_date=None):
    """
    Get stock money flow (buy/sell volume by order size).
    """
    if not start_date:
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y%m%d')
        
    try:
        pro = get_pro()
        df = pro.moneyflow(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df.empty:
            return f"No money flow data found for {ts_code}."
        df = df.sort_values('trade_date', ascending=False)
        return df.to_json(orient='records', force_ascii=False)
    except Exception as e:
        return f"Error fetching money flow (check permission/points): {str(e)}"

def get_hsgt_top10(trade_date=None):
    """
    Get Northbound/Southbound top 10 turnover.
    """
    if not trade_date:
        # Tushare data might delay, try yesterday if today is empty or just let user specify
        trade_date = datetime.now().strftime('%Y%m%d')
    
    try:
        pro = get_pro()
        df = pro.hsgt_top10(trade_date=trade_date)
        if df.empty:
            # Try previous trading day if empty (simple retry logic)
            prev_date = (datetime.strptime(trade_date, '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')
            df = pro.hsgt_top10(trade_date=prev_date)
            if df.empty:
                 return f"No HSGT top 10 data found for {trade_date} or {prev_date}."
        return df.to_json(orient='records', force_ascii=False)
    except Exception as e:
        return f"Error fetching HSGT top 10: {str(e)}"

def get_limit_list(trade_date=None):
    """
    Get daily limit up/down list.
    """
    if not trade_date:
        trade_date = datetime.now().strftime('%Y%m%d')
    
    try:
        pro = get_pro()
        df = pro.limit_list(trade_date=trade_date)
        if df.empty:
             return f"No limit list data found for {trade_date}."
        return df.to_json(orient='records', force_ascii=False)
    except Exception as e:
        return f"Error fetching limit list: {str(e)}"

def get_top_list(trade_date=None):
    """
    Get daily dragon and tiger list.
    """
    if not trade_date:
        trade_date = datetime.now().strftime('%Y%m%d')
        
    try:
        pro = get_pro()
        df = pro.top_list(trade_date=trade_date)
        if df.empty:
            return f"No top list data found for {trade_date}."
        return df.to_json(orient='records', force_ascii=False)
    except Exception as e:
        return f"Error fetching top list: {str(e)}"

def get_forecast(ts_code, start_date=None, end_date=None):
    """
    Get financial forecast.
    """
    if not start_date:
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d') # Last 6 months
    if not end_date:
        end_date = datetime.now().strftime('%Y%m%d')
        
    try:
        pro = get_pro()
        df = pro.forecast(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df.empty:
            return f"No forecast data found for {ts_code}."
        df = df.sort_values('ann_date', ascending=False)
        return df.to_json(orient='records', force_ascii=False)
    except Exception as e:
        return f"Error fetching forecast: {str(e)}"

def get_concept_detail(concept_name=None, ts_code=None):
    """
    Get stocks in a concept or concepts of a stock.
    Supports fuzzy search for concept name.
    """
    try:
        pro = get_pro()
        
        # If searching for what concepts a stock belongs to
        if ts_code:
            df = pro.concept_detail(ts_code=ts_code)
            if df.empty:
                return f"No concept data found for stock {ts_code}."
            return df.to_json(orient='records', force_ascii=False)
            
        # If searching for stocks in a concept
        if concept_name:
            # First need to find concept ID
            # This is heavy, maybe optimization needed later. 
            # We fetch all concepts and filter.
            concepts = pro.concept()
            matched = concepts[concepts['name'].str.contains(concept_name)]
            
            if matched.empty:
                return f"No concept found matching '{concept_name}'."
            
            # If multiple matches, return list of potential matches or just the first one
            if len(matched) > 1:
                # If exact match exists, prefer it
                exact = matched[matched['name'] == concept_name]
                if not exact.empty:
                    concept_id = exact.iloc[0]['code']
                else:
                    # Just pick first or return list of names to ask user clarification
                    # For simplicity, we pick first but warn
                    concept_id = matched.iloc[0]['code']
            else:
                concept_id = matched.iloc[0]['code']
                
            # Get stocks for this concept
            df = pro.concept_detail(id=concept_id)
            if df.empty:
                 return f"No stocks found for concept {concept_name} (ID: {concept_id})."
            return df.to_json(orient='records', force_ascii=False)

        return "Error: Please provide either concept_name or ts_code."
        
    except Exception as e:
        return f"Error fetching concept detail: {str(e)}"

def screen_stocks(pe_min=None, pe_max=None, pb_min=None, pb_max=None, 
                  mv_min=None, mv_max=None, dv_min=None, 
                  turnover_min=None, turnover_max=None,
                  industry=None, limit=20):
    """
    Screen stocks based on fundamental and technical indicators.
    """
    try:
        pro = get_pro()
        
        # 1. Determine the latest trading date
        # We can't easily query "latest", so we check today, if empty, check yesterday, etc.
        # A more robust way is to use trade_cal or just try loop back a few days.
        now = datetime.now()
        found_date = None
        df_daily = pd.DataFrame()
        
        # Try last 5 days to find data
        for i in range(5):
            date_str = (now - timedelta(days=i)).strftime('%Y%m%d')
            # Fetch basic daily data for ALL stocks on this date
            # Note: Tushare limits might apply, but daily_basic usually allows full fetch for one date
            try:
                # We need to fetch enough fields for filtering
                # total_mv is in 10k CNY usually? Tushare docs say: total_mv: 总市值 （万元）
                df = pro.daily_basic(trade_date=date_str, 
                                   fields='ts_code,trade_date,close,pe,pe_ttm,pb,total_mv,turnover_rate,dv_ratio')
                if not df.empty:
                    df_daily = df
                    found_date = date_str
                    break
            except:
                continue
                
        if df_daily.empty:
            return "Error: Could not fetch daily basic data for the last 5 days."
            
        # 2. Filter by Industry if specified
        if industry:
            # Fuzzy match industry name in stock_basic
            # First get all stocks (cached ideally, but here we fetch)
            df_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
            
            # Filter stock_basic by industry
            # Check if industry is in the 'industry' column
            # Ensure column is string
            df_basic['industry'] = df_basic['industry'].astype(str)
            target_stocks = df_basic[df_basic['industry'].str.contains(industry, na=False)]
            
            if target_stocks.empty:
                return f"No stocks found in industry matching '{industry}'."
                
            # Filter df_daily to only include these stocks
            df_daily = df_daily[df_daily['ts_code'].isin(target_stocks['ts_code'])]
            
            if df_daily.empty:
                return f"No data found for stocks in industry '{industry}' on {found_date}."

        # 3. Apply numeric filters
        # Handle PE (using pe_ttm usually better, or just pe)
        if pe_min is not None:
            df_daily = df_daily[df_daily['pe_ttm'] >= float(pe_min)]
        if pe_max is not None:
            df_daily = df_daily[df_daily['pe_ttm'] <= float(pe_max)]
            
        if pb_min is not None:
            df_daily = df_daily[df_daily['pb'] >= float(pb_min)]
        if pb_max is not None:
            df_daily = df_daily[df_daily['pb'] <= float(pb_max)]
            
        # Market Value (total_mv is in 10k, usually we filter by 'Yi' (100 million))
        # Input mv_min usually implies 'Yi'. So 100 Yi = 100 * 10000 (unit in table)
        # Let's assume input is in 100 Million (Yi)
        if mv_min is not None:
            df_daily = df_daily[df_daily['total_mv'] >= float(mv_min) * 10000]
        if mv_max is not None:
            df_daily = df_daily[df_daily['total_mv'] <= float(mv_max) * 10000]
            
        if dv_min is not None:
            df_daily = df_daily[df_daily['dv_ratio'] >= float(dv_min)]
            
        if turnover_min is not None:
            df_daily = df_daily[df_daily['turnover_rate'] >= float(turnover_min)]
        if turnover_max is not None:
            df_daily = df_daily[df_daily['turnover_rate'] <= float(turnover_max)]
            
        # 4. Return results
        if df_daily.empty:
            return "No stocks found matching the criteria."
            
        # Add Name and Industry to result for better readability
        # If we didn't fetch stock_basic yet
        if not industry: # If industry was filtered, we already have target_stocks, but easier to just fetch specific codes or all again
             # Fetch names for the result codes
             codes = df_daily['ts_code'].tolist()
             # If too many codes, fetching all basic might be faster than chunks? 
             # stock_basic returns ~5000 rows, fast enough.
             df_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
        
        # Merge to get Name and Industry
        result = pd.merge(df_daily, df_basic[['ts_code', 'name', 'industry']], on='ts_code', how='left')
        
        # Sort by Market Value desc by default if no other sort implied? 
        # Or maybe PE asc? Let's sort by Total MV desc to show big companies first
        result = result.sort_values('total_mv', ascending=False)
        
        # Limit
        result = result.head(limit)
        
        # Format columns
        # total_mv convert to Yi for display? 
        # Let's just return JSON and let LLM interpret, but adding a hint is good.
        # We return raw data, LLM can format.
        
        return result.to_json(orient='records', force_ascii=False)

    except Exception as e:
        return f"Error executing stock screen: {str(e)}"

def reset_core_config():
    """
    Reset core configuration (Tushare Token & LLM).
    """
    try:
        print("Initiating core configuration reset...")
        Config.setup()
        return "Core configuration wizard finished. New settings are applied."
    except Exception as e:
        return f"Error resetting core config: {str(e)}"

# Tool definitions for LLM
BASE_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "reset_core_config",
            "description": "Reset or update core configuration (Tushare Token, LLM Provider, API Keys) interactively. Use this when the user wants to change API keys or providers.",
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
            "name": "get_current_time",
            "description": "Get the current system date and time. Use this when the user asks about 'today', 'now', or relative dates.",
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
            "name": "get_stock_basic",
            "description": "Get basic information about a stock, such as its industry, area, and listing date. You can search by stock name or code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {
                        "type": "string",
                        "description": "The stock code (e.g., '000001.SZ')."
                    },
                    "name": {
                        "type": "string",
                        "description": "The stock name (e.g., '平安银行')."
                    }
                },
                "required": [] 
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_daily_price",
            "description": "Get historical daily price data for a stock within a date range (Open, High, Low, Close, Vol).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {
                        "type": "string",
                        "description": "The stock code (e.g., '000001.SZ')."
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYYMMDD format. Defaults to 30 days ago."
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYYMMDD format. Defaults to today."
                    }
                },
                "required": ["ts_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_realtime_price",
            "description": "Get the latest real-time stock price data (current price, bid/ask, volume, etc.). Use this for the most up-to-date market snapshot.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {
                        "type": "string",
                        "description": "The stock code (e.g., '000001.SZ')."
                    }
                },
                "required": ["ts_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_daily_basic",
            "description": "Get daily basic indicators including PE (Price-to-Earnings), PB (Price-to-Book), Turnover Rate, and Market Value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {
                        "type": "string",
                        "description": "The stock code (e.g., '000001.SZ')."
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date (YYYYMMDD)."
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date (YYYYMMDD)."
                    }
                },
                "required": ["ts_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_income_statement",
            "description": "Get historical income statement data (Revenue, Net Income) to analyze financial performance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {
                        "type": "string",
                        "description": "The stock code (e.g., '000001.SZ')."
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date (YYYYMMDD). Defaults to 2 years ago."
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date (YYYYMMDD)."
                    }
                },
                "required": ["ts_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_index_daily",
            "description": "Get daily index market data (e.g. 000001.SH for Shanghai Composite, 399001.SZ for Shenzhen Component).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {
                        "type": "string",
                        "description": "The index code (e.g., '000001.SH', '399001.SZ')."
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date (YYYYMMDD)."
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date (YYYYMMDD)."
                    }
                },
                "required": ["ts_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_moneyflow",
            "description": "Get stock money flow data (buy/sell volume by order size). Useful for analyzing fund movement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {
                        "type": "string",
                        "description": "The stock code (e.g., '000001.SZ')."
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date (YYYYMMDD)."
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date (YYYYMMDD)."
                    }
                },
                "required": ["ts_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_hsgt_top10",
            "description": "Get top 10 turnover stocks for Northbound (Shanghai/Shenzhen-Hong Kong Connect) trading.",
            "parameters": {
                "type": "object",
                "properties": {
                    "trade_date": {
                        "type": "string",
                        "description": "Trade date (YYYYMMDD)."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_limit_list",
            "description": "Get the list of stocks that hit the daily limit up or down.",
            "parameters": {
                "type": "object",
                "properties": {
                    "trade_date": {
                        "type": "string",
                        "description": "Trade date (YYYYMMDD)."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_list",
            "description": "Get the Dragon and Tiger list (daily active/volatile stocks with detailed seat info).",
            "parameters": {
                "type": "object",
                "properties": {
                    "trade_date": {
                        "type": "string",
                        "description": "Trade date (YYYYMMDD)."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": "Get financial forecast/guidance published by the company.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {
                        "type": "string",
                        "description": "The stock code (e.g., '000001.SZ')."
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Announcement start date (YYYYMMDD)."
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Announcement end date (YYYYMMDD)."
                    }
                },
                "required": ["ts_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_concept_detail",
            "description": "Get stocks belonging to a specific concept (by name) OR get concepts for a specific stock (by code).",
            "parameters": {
                "type": "object",
                "properties": {
                    "concept_name": {
                        "type": "string",
                        "description": "The concept name (e.g., 'Sora概念', '锂电池'). Fuzzy matching supported."
                    },
                    "ts_code": {
                        "type": "string",
                        "description": "The stock code (e.g., '000001.SZ'). Use this to see what concepts a stock belongs to."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_technical_indicators",
            "description": "Calculate technical indicators (MACD, RSI, KDJ, BOLL) for a stock. Useful for technical analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {
                        "type": "string",
                        "description": "The stock code (e.g., '000001.SZ')."
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date (YYYYMMDD). Optional, defaults to returning recent data."
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date (YYYYMMDD). Optional."
                    }
                },
                "required": ["ts_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_technical_patterns",
            "description": "Automatically identify technical patterns (Golden Cross, Dead Cross, Overbought/Oversold, Bollinger Band Break) for a stock based on the latest data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {
                        "type": "string",
                        "description": "The stock code (e.g., '000001.SZ')."
                    }
                },
                "required": ["ts_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "screen_stocks",
            "description": "Smart stock picker/screener. Filter stocks based on fundamental indicators (PE, PB, Market Value, Dividend) and industry.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pe_min": {"type": "number", "description": "Minimum PE (TTM) ratio."},
                    "pe_max": {"type": "number", "description": "Maximum PE (TTM) ratio. Use this for 'low PE' criteria."},
                    "pb_min": {"type": "number", "description": "Minimum PB ratio."},
                    "pb_max": {"type": "number", "description": "Maximum PB ratio."},
                    "mv_min": {"type": "number", "description": "Minimum Market Value (in 100 Million/Yi CNY). e.g., 100 for 100 Yi."},
                    "mv_max": {"type": "number", "description": "Maximum Market Value (in 100 Million/Yi CNY)."},
                    "dv_min": {"type": "number", "description": "Minimum Dividend Yield (%). e.g., 3 for >3%."},
                    "turnover_min": {"type": "number", "description": "Minimum Turnover Rate (%)."},
                    "turnover_max": {"type": "number", "description": "Maximum Turnover Rate (%)."},
                    "net_profit_min": {"type": "number", "description": "Minimum Net Main Force Inflow (in 10k/Wan CNY). e.g., 1000 for 1000 Wan."},
                    "industry": {"type": "string", "description": "Industry name to filter by (fuzzy match). e.g., '银行', '半导体'."},
                    "limit": {"type": "integer", "description": "Max number of results to return. Default 20."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_backtest",
            "description": "Run a historical backtest for a trading strategy on a specific stock.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {
                        "type": "string",
                        "description": "The stock code (e.g., '000001.SZ')."
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["ma_cross", "macd", "rsi"],
                        "description": "Strategy type. 'ma_cross' (Moving Average Crossover), 'macd' (MACD Cross), 'rsi' (RSI Reversal)."
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Backtest start date (YYYYMMDD). Defaults to 1 year ago."
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Backtest end date (YYYYMMDD). Defaults to today."
                    },
                    "params": {
                        "type": "object",
                        "description": "Strategy parameters. e.g., {'short_window': 5, 'long_window': 20} for ma_cross."
                    }
                },
                "required": ["ts_code"]
            }
        }
    }
]

# Combine schemas
TOOLS_SCHEMA = BASE_TOOLS_SCHEMA + PORTFOLIO_TOOLS_SCHEMA + SCHEDULER_TOOLS_SCHEMA + PROFILE_TOOLS_SCHEMA

# Helper to execute tool calls
def execute_tool_call(tool_name, arguments):
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return "Error: Invalid JSON arguments."

    if tool_name == "get_current_time":
        return get_current_time()
    elif tool_name == "get_stock_basic":
        return get_stock_basic(**arguments)
    elif tool_name == "get_daily_price":
        return get_daily_price(**arguments)
    elif tool_name == "get_realtime_price":
        return get_realtime_price(**arguments)
    elif tool_name == "get_daily_basic":
        return get_daily_basic(**arguments)
    elif tool_name == "get_income_statement":
        return get_income_statement(**arguments)
    elif tool_name == "get_index_daily":
        return get_index_daily(**arguments)
    elif tool_name == "get_moneyflow":
        return get_moneyflow(**arguments)
    elif tool_name == "get_hsgt_top10":
        return get_hsgt_top10(**arguments)
    elif tool_name == "get_limit_list":
        return get_limit_list(**arguments)
    elif tool_name == "get_top_list":
        return get_top_list(**arguments)
    elif tool_name == "get_forecast":
        return get_forecast(**arguments)
    elif tool_name == "get_concept_detail":
        return get_concept_detail(**arguments)
    elif tool_name == "get_technical_indicators":
        return get_technical_indicators(**arguments)
    elif tool_name == "get_technical_patterns":
        return get_technical_patterns(**arguments)
    elif tool_name == "screen_stocks":
        return screen_stocks(**arguments)
    elif tool_name == "run_backtest":
        return run_backtest(**arguments)
    elif tool_name == "add_portfolio_position":
        return add_portfolio_position(**arguments)
    elif tool_name == "remove_portfolio_position":
        return remove_portfolio_position(**arguments)
    elif tool_name == "get_portfolio_status":
        return get_portfolio_status(**arguments)
    elif tool_name == "clear_portfolio":
        return clear_portfolio(**arguments)
    elif tool_name == "add_price_alert":
        return add_price_alert(**arguments)
    elif tool_name == "list_alerts":
        return list_alerts(**arguments)
    elif tool_name == "remove_alert":
        return remove_alert(**arguments)
    elif tool_name == "update_alert":
        return update_alert(**arguments)
    elif tool_name == "reset_email_config":
        return reset_email_config()
    elif tool_name == "reset_core_config":
        return reset_core_config()
    elif tool_name == "update_user_profile":
        return update_user_profile(**arguments)
    elif tool_name == "get_user_profile":
        return get_user_profile(**arguments)
    else:
        return f"Error: Tool '{tool_name}' not found."
