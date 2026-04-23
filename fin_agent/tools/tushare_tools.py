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

def get_long_tail_stocks(min_mv=10, max_mv=200, max_pe=40, max_pb=5, 
                        max_turnover=3.0, check_consolidation=False, 
                        check_volume_spike=False, limit=20):
    """
    Discovery of long-tail/neglected stocks.
    Criteria:
    - Small/Mid Cap (default 10-200 Yi)
    - Value (PE < 40, PB < 5)
    - Neglected (Turnover < 3%)
    - Optional: Long-term consolidation (Low volatility)
    - Optional: Abnormal volume (Volume spike)
    """
    try:
        pro = get_pro()
        
        # 1. Get latest trading date
        now = datetime.now()
        df_daily = pd.DataFrame()
        found_date = None
        
        # Try last 5 days to find data
        for i in range(5):
            date_str = (now - timedelta(days=i)).strftime('%Y%m%d')
            try:
                # Fetch daily basic
                df = pro.daily_basic(trade_date=date_str, 
                                   fields='ts_code,trade_date,close,pe_ttm,pb,total_mv,turnover_rate,volume_ratio')
                if not df.empty:
                    df_daily = df
                    found_date = date_str
                    break
            except:
                continue
        
        if df_daily.empty:
            return "Error: Could not fetch daily basic data."
            
        # 2. Basic Filtering
        # Market Value: Tushare total_mv is in 10k CNY. So 1 Yi = 10,000 unit.
        mv_min_val = float(min_mv) * 10000
        mv_max_val = float(max_mv) * 10000
        
        mask = (df_daily['total_mv'] >= mv_min_val) & \
               (df_daily['total_mv'] <= mv_max_val) & \
               (df_daily['pe_ttm'] > 0) & (df_daily['pe_ttm'] <= float(max_pe)) & \
               (df_daily['pb'] <= float(max_pb)) & \
               (df_daily['turnover_rate'] <= float(max_turnover))
               
        candidates = df_daily[mask].copy()
        
        if candidates.empty:
            return "No stocks found matching the basic long-tail criteria."
            
        # 3. Advanced Filtering (Consolidation / Volume Spike)
        # If requested, we limit candidates to top 50 (by lowest turnover) before fetching history to save time/quota
        if check_consolidation or check_volume_spike:
            # Sort by turnover first to process most neglected ones
            candidates = candidates.sort_values('turnover_rate').head(50)
            final_ts_codes = []
            
            # Start date for history (90 days for consolidation check)
            start_date = (datetime.strptime(found_date, '%Y%m%d') - timedelta(days=90)).strftime('%Y%m%d')
            
            for index, row in candidates.iterrows():
                try:
                    df_hist = pro.daily(ts_code=row['ts_code'], start_date=start_date, end_date=found_date)
                    if df_hist.empty or len(df_hist) < 20:
                        continue
                        
                    is_valid = True
                    
                    # Check Consolidation (Low Volatility)
                    # Std Dev of Close Price / Mean Close Price
                    if check_consolidation:
                        # Use last 60 days
                        df_window = df_hist.head(60)
                        if len(df_window) > 10:
                            volatility = df_window['close'].std() / df_window['close'].mean()
                            # Threshold: < 0.15 (15%) implies relatively stable
                            if volatility > 0.15: 
                                is_valid = False
                    
                    # Check Volume Spike
                    # Latest volume vs Avg volume of previous days
                    if is_valid and check_volume_spike:
                        # Ensure we have enough data
                        if len(df_hist) > 20:
                            latest_vol = df_hist.iloc[0]['vol']
                            # Avg of next 20 days (previous in time)
                            avg_vol = df_hist.iloc[1:21]['vol'].mean()
                            # Expect > 2.0x volume spike
                            if avg_vol == 0 or latest_vol / avg_vol < 2.0:
                                is_valid = False
                                
                    if is_valid:
                        final_ts_codes.append(row['ts_code'])
                        
                except Exception:
                    continue
            
            # Filter main df
            if not final_ts_codes:
                 return "No stocks found matching the advanced criteria (Consolidation/Volume Spike)."
            candidates = candidates[candidates['ts_code'].isin(final_ts_codes)]
        
        # 4. Final Result Preparation
        # Fetch names
        df_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry,area')
        
        result = pd.merge(candidates, df_basic, on='ts_code', how='left')
        
        # Sort logic
        if check_volume_spike:
            result = result.sort_values('volume_ratio', ascending=False)
        else:
            # Sort by Turnover (Ascending) for "most neglected"
            result = result.sort_values('turnover_rate', ascending=True)
            
        result = result.head(limit)
        
        return result.to_json(orient='records', force_ascii=False)

    except Exception as e:
        return f"Error executing long-tail stock discovery: {str(e)}"

def screen_stocks(pe_min=None, pe_max=None, pb_min=None, pb_max=None, 
                  mv_min=None, mv_max=None, dv_min=None, 
                  turnover_min=None, turnover_max=None,
                  net_profit_min=None,
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
            
        # Estimate Net Profit (TTM) from Total MV and PE TTM
        # Net Profit = Total MV / PE TTM
        # total_mv is in 10k, so result is in 10k
        if net_profit_min is not None:
            # Avoid division by zero or negative PE (loss) if we strictly want profit
            # If PE is negative, profit is negative.
            # We construct a temporary column for filtering
            # Handle potential zeros in pe_ttm to avoid inf
            # We only care if pe_ttm > 0 for positive profit check usually
            
            # Create a mask for valid calculation
            valid_pe = (df_daily['pe_ttm'] != 0) & (df_daily['pe_ttm'].notna()) & (df_daily['total_mv'].notna())
            
            # Calculate estimated profit for valid rows
            # Initialize with -inf or NaN
            estimated_profit = pd.Series(index=df_daily.index, dtype=float)
            estimated_profit[valid_pe] = df_daily.loc[valid_pe, 'total_mv'] / df_daily.loc[valid_pe, 'pe_ttm']
            
            # Filter
            # If net_profit_min >= 0, we imply we need positive profit, so negative PE rows (negative profit) are excluded naturally if we calculate correctly (MV is pos, PE neg -> Profit neg)
            df_daily = df_daily[estimated_profit >= float(net_profit_min)]

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

def get_hk_stock_basic(ts_code=None, name=None):
    """
    Get basic information about a Hong Kong stock.
    :param ts_code: Stock code (e.g., 00700.HK for Tencent)
    :param name: Stock name (e.g., 腾讯控股)
    :return: DataFrame or dict string
    """
    try:
        pro = get_pro()
        # If name is provided but ts_code is not, try to find ts_code
        if name and not ts_code:
            df = pro.hk_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
            df = df[df['name'].str.contains(name, na=False)]
            if df.empty:
                return f"Error: Hong Kong stock with name '{name}' not found."
            return df.to_json(orient='records', force_ascii=False)
        
        if not ts_code:
            return "Error: Please provide either ts_code or name."
        
        df = pro.hk_basic(ts_code=ts_code, fields='ts_code,symbol,name,area,industry,list_date')
        if df.empty:
            return f"Error: Hong Kong stock code '{ts_code}' not found."
        return df.to_json(orient='records', force_ascii=False)
    except Exception as e:
        return f"Error fetching Hong Kong stock basic info: {str(e)}"

def get_hk_daily_price(ts_code, start_date=None, end_date=None):
    """
    Get daily Hong Kong stock price.
    :param ts_code: Stock code (e.g., 00700.HK)
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
        df = pro.hk_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df.empty:
            return f"No data found for Hong Kong stock {ts_code} between {start_date} and {end_date}."
        
        # Ensure data is sorted by date descending
        df = df.sort_values('trade_date', ascending=False)
        return df.to_json(orient='records', force_ascii=False)
    except Exception as e:
        return f"Error fetching Hong Kong stock daily price: {str(e)}"

def get_us_stock_basic(ts_code=None, name=None):
    """
    Get basic information about a US stock.
    :param ts_code: Stock code (e.g., AAPL.O for Apple)
    :param name: Stock name (e.g., Apple Inc)
    :return: DataFrame or dict string
    """
    try:
        pro = get_pro()
        # If name is provided but ts_code is not, try to find ts_code
        if name and not ts_code:
            df = pro.us_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
            df = df[df['name'].str.contains(name, na=False, case=False)]
            if df.empty:
                return f"Error: US stock with name '{name}' not found."
            return df.to_json(orient='records', force_ascii=False)
        
        if not ts_code:
            return "Error: Please provide either ts_code or name."
        
        df = pro.us_basic(ts_code=ts_code, fields='ts_code,symbol,name,area,industry,list_date')
        if df.empty:
            return f"Error: US stock code '{ts_code}' not found."
        return df.to_json(orient='records', force_ascii=False)
    except Exception as e:
        return f"Error fetching US stock basic info: {str(e)}"

def get_us_daily_price(ts_code, start_date=None, end_date=None):
    """
    Get daily US stock price.
    :param ts_code: Stock code (e.g., AAPL.O)
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
        df = pro.us_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df.empty:
            return f"No data found for US stock {ts_code} between {start_date} and {end_date}."
        
        # Ensure data is sorted by date descending
        df = df.sort_values('trade_date', ascending=False)
        return df.to_json(orient='records', force_ascii=False)
    except Exception as e:
        return f"Error fetching US stock daily price: {str(e)}"

def get_hk_realtime_price(ts_code):
    """
    Get realtime Hong Kong stock price.
    Note: Tushare may not support realtime data for HK stocks. 
    If realtime is unavailable, returns the latest daily price data.
    :param ts_code: Stock code (e.g., 00700.HK)
    :return: JSON string
    """
    try:
        # Try legacy realtime interface first (may not work for HK stocks)
        # Extract code without suffix for legacy interface
        code = ts_code.replace('.HK', '')
        
        try:
            df = ts.get_realtime_quotes(code)
            if df is not None and not df.empty:
                df['ts_code'] = ts_code
                return df.to_json(orient='records', force_ascii=False)
        except:
            # Legacy interface may not support HK stocks, fallback to latest daily data
            pass
        
        # Fallback: Get latest daily price data
        pro = get_pro()
        # Try last 5 trading days to get the most recent data
        now = datetime.now()
        for i in range(5):
            date_str = (now - timedelta(days=i)).strftime('%Y%m%d')
            try:
                df = pro.hk_daily(ts_code=ts_code, trade_date=date_str)
                if not df.empty:
                    df = df.sort_values('trade_date', ascending=False)
                    # Add a note that this is daily data, not realtime
                    return df.to_json(orient='records', force_ascii=False) + " [Note: Latest daily data, not realtime]"
            except:
                continue
        
        return f"No realtime or recent daily data found for Hong Kong stock {ts_code}."
    except Exception as e:
        return f"Error fetching Hong Kong stock realtime price: {str(e)}"

def get_us_realtime_price(ts_code):
    """
    Get realtime US stock price.
    Note: Tushare may not support realtime data for US stocks.
    If realtime is unavailable, returns the latest daily price data.
    :param ts_code: Stock code (e.g., AAPL.O)
    :return: JSON string
    """
    try:
        # Try legacy realtime interface first (may not work for US stocks)
        # Extract symbol without suffix for legacy interface
        code = ts_code.replace('.O', '').replace('.N', '').replace('.A', '')
        
        try:
            df = ts.get_realtime_quotes(code)
            if df is not None and not df.empty:
                df['ts_code'] = ts_code
                return df.to_json(orient='records', force_ascii=False)
        except:
            # Legacy interface may not support US stocks, fallback to latest daily data
            pass
        
        # Fallback: Get latest daily price data
        pro = get_pro()
        # Try last 5 trading days to get the most recent data
        now = datetime.now()
        for i in range(5):
            date_str = (now - timedelta(days=i)).strftime('%Y%m%d')
            try:
                df = pro.us_daily(ts_code=ts_code, trade_date=date_str)
                if not df.empty:
                    df = df.sort_values('trade_date', ascending=False)
                    # Add a note that this is daily data, not realtime
                    return df.to_json(orient='records', force_ascii=False) + " [Note: Latest daily data, not realtime]"
            except:
                continue
        
        return f"No realtime or recent daily data found for US stock {ts_code}."
    except Exception as e:
        return f"Error fetching US stock realtime price: {str(e)}"

def get_etf_basic(ts_code=None, name=None):
    """
    Get basic information about an ETF.
    :param ts_code: ETF code (e.g., 510330.SH)
    :param name: ETF name (e.g., 沪深300ETF)
    :return: DataFrame or dict string
    """
    try:
        pro = get_pro()
        if name and not ts_code:
            df = pro.fund_basic(market='E', fields='ts_code,name,management,custodian,fund_type,found_date,list_date')
            df = df[df['name'].str.contains(name, na=False)]
            if df.empty:
                return f"Error: ETF with name '{name}' not found."
            return df.to_json(orient='records', force_ascii=False)
        
        if not ts_code:
            return "Error: Please provide either ts_code or name."
        
        df = pro.fund_basic(ts_code=ts_code, market='E', fields='ts_code,name,management,custodian,fund_type,found_date,list_date')
        if df.empty:
            return f"Error: ETF code '{ts_code}' not found."
        return df.to_json(orient='records', force_ascii=False)
    except Exception as e:
        return f"Error fetching ETF basic info: {str(e)}"

def get_etf_daily_price(ts_code, start_date=None, end_date=None):
    """
    Get daily ETF price.
    :param ts_code: ETF code (e.g., 510330.SH)
    :param start_date: Start date (YYYYMMDD)
    :param end_date: End date (YYYYMMDD)
    :return: JSON string
    """
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y%m%d')

    try:
        pro = get_pro()
        df = pro.fund_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df.empty:
            return f"No data found for ETF {ts_code} between {start_date} and {end_date}."
        
        df = df.sort_values('trade_date', ascending=False)
        return df.to_json(orient='records', force_ascii=False)
    except Exception as e:
        return f"Error fetching ETF daily price: {str(e)}"

def get_cb_basic(ts_code=None, name=None):
    """
    Get basic information about a convertible bond (可转债).
    :param ts_code: Convertible bond code (e.g., 123456.SH)
    :param name: Convertible bond name
    :return: DataFrame or dict string
    """
    try:
        pro = get_pro()
        if name and not ts_code:
            df = pro.cb_basic(fields='ts_code,bond_short_name,stk_code,stk_short_name,issue_type,issue_size,list_date')
            df = df[df['bond_short_name'].str.contains(name, na=False)]
            if df.empty:
                return f"Error: Convertible bond with name '{name}' not found."
            return df.to_json(orient='records', force_ascii=False)
        
        if not ts_code:
            return "Error: Please provide either ts_code or name."
        
        df = pro.cb_basic(ts_code=ts_code, fields='ts_code,bond_short_name,stk_code,stk_short_name,issue_type,issue_size,list_date')
        if df.empty:
            return f"Error: Convertible bond code '{ts_code}' not found."
        return df.to_json(orient='records', force_ascii=False)
    except Exception as e:
        return f"Error fetching convertible bond basic info: {str(e)}"

def get_cb_daily_price(ts_code, start_date=None, end_date=None):
    """
    Get daily convertible bond price.
    :param ts_code: Convertible bond code (e.g., 123456.SH)
    :param start_date: Start date (YYYYMMDD)
    :param end_date: End date (YYYYMMDD)
    :return: JSON string
    """
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y%m%d')

    try:
        pro = get_pro()
        df = pro.cb_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df.empty:
            return f"No data found for convertible bond {ts_code} between {start_date} and {end_date}."
        
        df = df.sort_values('trade_date', ascending=False)
        return df.to_json(orient='records', force_ascii=False)
    except Exception as e:
        return f"Error fetching convertible bond daily price: {str(e)}"

def get_futures_basic(ts_code=None, exchange=None, name=None):
    """
    Get basic information about a futures contract.
    :param ts_code: Futures code (e.g., CU2412.SHF)
    :param exchange: Exchange code (SHF, DCE, CZE, INE)
    :param name: Futures name
    :return: DataFrame or dict string
    """
    try:
        pro = get_pro()
        if name and not ts_code:
            df = pro.fut_basic(exchange=exchange or '', fields='ts_code,symbol,name,exchange,list_date,delist_date')
            df = df[df['name'].str.contains(name, na=False)]
            if df.empty:
                return f"Error: Futures contract with name '{name}' not found."
            return df.to_json(orient='records', force_ascii=False)
        
        if not ts_code:
            return "Error: Please provide either ts_code or name."
        
        df = pro.fut_basic(ts_code=ts_code, exchange=exchange or '', fields='ts_code,symbol,name,exchange,list_date,delist_date')
        if df.empty:
            return f"Error: Futures code '{ts_code}' not found."
        return df.to_json(orient='records', force_ascii=False)
    except Exception as e:
        return f"Error fetching futures basic info: {str(e)}"

def get_futures_daily_price(ts_code, start_date=None, end_date=None, exchange=None):
    """
    Get daily futures price.
    :param ts_code: Futures code (e.g., CU2412.SHF)
    :param start_date: Start date (YYYYMMDD)
    :param end_date: End date (YYYYMMDD)
    :param exchange: Exchange code (SHF, DCE, CZE, INE)
    :return: JSON string
    """
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y%m%d')

    try:
        pro = get_pro()
        df = pro.fut_daily(ts_code=ts_code, start_date=start_date, end_date=end_date, exchange=exchange)
        if df.empty:
            return f"No data found for futures {ts_code} between {start_date} and {end_date}."
        
        df = df.sort_values('trade_date', ascending=False)
        return df.to_json(orient='records', force_ascii=False)
    except Exception as e:
        return f"Error fetching futures daily price: {str(e)}"

def get_macro_gdp(period=None, start_period=None, end_period=None):
    """
    Get GDP (Gross Domestic Product) macroeconomic data.
    :param period: Specific period (YYYY format for year)
    :param start_period: Start period (YYYY format)
    :param end_period: End period (YYYY format)
    :return: JSON string
    """
    try:
        pro = get_pro()
        
        # Convert period formats to Tushare expected format (YYYY)
        def convert_period(p):
            """Convert period to YYYY format"""
            if not p:
                return None
            p_str = str(p).strip()
            # If it's YYYYQ format (e.g., 2024Q1), extract year
            if 'Q' in p_str.upper():
                return p_str.split('Q')[0]
            # If it's YYYY format, use as is
            if len(p_str) == 4 and p_str.isdigit():
                return p_str
            # If it's YYYYMM format, extract year
            if len(p_str) == 6 and p_str.isdigit():
                return p_str[:4]
            # Try to extract first 4 digits
            import re
            match = re.search(r'\d{4}', p_str)
            if match:
                return match.group()
            return p_str
        
        # Determine query parameters
        if period:
            # Single period query
            year = convert_period(period)
            if not year or len(year) != 4:
                return f"Error: Invalid period format '{period}'. Please use YYYY format (e.g., '2024')."
            try:
                df = pro.cn_gdp(start_date=year, end_date=year)
            except Exception as api_error:
                # Try alternative API call format
                try:
                    df = pro.cn_gdp(year=year)
                except:
                    return f"Error calling Tushare API: {str(api_error)}"
        elif start_period and end_period:
            # Range query
            start_year = convert_period(start_period)
            end_year = convert_period(end_period)
            if not start_year or not end_year or len(start_year) != 4 or len(end_year) != 4:
                return f"Error: Invalid period format. Please use YYYY format (e.g., '2020', '2024')."
            try:
                df = pro.cn_gdp(start_date=start_year, end_date=end_year)
            except Exception as api_error:
                return f"Error calling Tushare API: {str(api_error)}"
        else:
            # Default to last 5 years
            end_year = datetime.now().strftime('%Y')
            start_year = str(int(end_year) - 5)
            try:
                df = pro.cn_gdp(start_date=start_year, end_date=end_year)
            except Exception as api_error:
                return f"Error calling Tushare API: {str(api_error)}"
        
        if df.empty:
            return "No GDP data found for the specified period."
        
        # Sort by period - try different possible column names safely
        try:
            # Check available columns
            available_cols = list(df.columns)
            
            # Try to find a suitable column for sorting
            sort_column = None
            for col in ['period', 'quarter', 'year', 'date', 'time', 'end_date']:
                if col in available_cols:
                    sort_column = col
                    break
            
            if sort_column:
                df = df.sort_values(sort_column, ascending=False, na_position='last')
            else:
                # If no suitable column, try to sort by first column
                if len(available_cols) > 0:
                    df = df.sort_values(available_cols[0], ascending=False, na_position='last')
                else:
                    df = df.sort_index(ascending=False)
        except Exception as sort_error:
            # If sorting fails, just return unsorted data
            pass
        
        return df.to_json(orient='records', force_ascii=False)
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f"Error fetching GDP data: {str(e)}\nDetails: {error_detail[:200]}"

def get_macro_cpi(period=None, start_period=None, end_period=None):
    """
    Get CPI (Consumer Price Index) macroeconomic data.
    :param period: Specific period (YYYYMM format)
    :param start_period: Start period (YYYYMM)
    :param end_period: End period (YYYYMM)
    :return: JSON string
    """
    try:
        pro = get_pro()
        if period:
            # Single period query
            df = pro.cn_cpi(month=period)
        elif start_period and end_period:
            # Range query - Tushare cn_cpi uses start_m and end_m
            df = pro.cn_cpi(start_m=start_period, end_m=end_period)
        else:
            # Default to last 12 months
            end_period = datetime.now().strftime('%Y%m')
            start_period = (datetime.now() - timedelta(days=365)).strftime('%Y%m')
            df = pro.cn_cpi(start_m=start_period, end_m=end_period)
        
        if df.empty:
            return "No CPI data found."
        
        # Sort by month
        if 'month' in df.columns:
            df = df.sort_values('month', ascending=False)
        elif 'm' in df.columns:
            df = df.sort_values('m', ascending=False)
        else:
            df = df.sort_index(ascending=False)
        
        return df.to_json(orient='records', force_ascii=False)
    except Exception as e:
        return f"Error fetching CPI data: {str(e)}"

def get_macro_m2(period=None, start_period=None, end_period=None):
    """
    Get M2 money supply macroeconomic data.
    :param period: Specific period (YYYYMM format)
    :param start_period: Start period (YYYYMM)
    :param end_period: End period (YYYYMM)
    :return: JSON string
    """
    try:
        pro = get_pro()
        if period:
            # Single period query - Tushare cn_m uses month parameter
            df = pro.cn_m(month=period)
        elif start_period and end_period:
            # Range query - Tushare cn_m uses start_m and end_m
            df = pro.cn_m(start_m=start_period, end_m=end_period)
        else:
            # Default to last 12 months
            end_period = datetime.now().strftime('%Y%m')
            start_period = (datetime.now() - timedelta(days=365)).strftime('%Y%m')
            df = pro.cn_m(start_m=start_period, end_m=end_period)
        
        if df.empty:
            return "No M2 data found."
        
        # Sort by month
        if 'month' in df.columns:
            df = df.sort_values('month', ascending=False)
        elif 'm' in df.columns:
            df = df.sort_values('m', ascending=False)
        else:
            df = df.sort_index(ascending=False)
        
        return df.to_json(orient='records', force_ascii=False)
    except Exception as e:
        return f"Error fetching M2 data: {str(e)}"

def get_macro_interest_rate(period=None, start_period=None, end_period=None):
    """
    Get interest rate data (SHIBOR - Shanghai Interbank Offered Rate).
    :param period: Specific period (YYYYMMDD format)
    :param start_period: Start period (YYYYMMDD)
    :param end_period: End period (YYYYMMDD)
    :return: JSON string
    """
    try:
        pro = get_pro()
        if period:
            # Convert YYYYMM to YYYYMMDD if needed
            if len(period) == 6:
                period = period + '01'
            df = pro.shibor(date=period)
        elif start_period and end_period:
            # Convert YYYYMM to YYYYMMDD if needed
            if len(start_period) == 6:
                start_period = start_period + '01'
            if len(end_period) == 6:
                end_period = end_period + '28'  # Use end of month
            df = pro.shibor(start_date=start_period, end_date=end_period)
        else:
            # Default to last 30 days
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
            df = pro.shibor(start_date=start_date, end_date=end_date)
        
        if df.empty:
            return "No interest rate data found."
        
        df = df.sort_values('date', ascending=False)
        return df.to_json(orient='records', force_ascii=False)
    except Exception as e:
        return f"Error fetching interest rate data: {str(e)}"

def get_global_index_comparison(indices=None):
    """
    Compare global market indices.
    :param indices: List of index codes (e.g., ['000001.SH', '399001.SZ', 'HSI.HI', 'SPX.GI'])
    :return: JSON string with comparison data
    """
    try:
        pro = get_pro()
        if not indices:
            # Default indices: Shanghai Composite, Shenzhen Component, Hang Seng Index, S&P 500
            indices = ['000001.SH', '399001.SZ', 'HSI.HI', 'SPX.GI']
        
        results = []
        now = datetime.now()
        
        for idx_code in indices:
            try:
                # Try to get latest data (last 5 days)
                for i in range(5):
                    date_str = (now - timedelta(days=i)).strftime('%Y%m%d')
                    try:
                        df = pro.index_daily(ts_code=idx_code, trade_date=date_str)
                        if not df.empty:
                            latest = df.iloc[0].to_dict()
                            latest['index_code'] = idx_code
                            results.append(latest)
                            break
                    except:
                        continue
            except:
                continue
        
        if not results:
            return "No global index data found."
        
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return f"Error fetching global index comparison: {str(e)}"

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
            "name": "get_hk_stock_basic",
            "description": "Get basic information about a Hong Kong stock (e.g., industry, listing date). You can search by stock name or code (e.g., '00700.HK' for Tencent).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {
                        "type": "string",
                        "description": "The Hong Kong stock code (e.g., '00700.HK')."
                    },
                    "name": {
                        "type": "string",
                        "description": "The Hong Kong stock name (e.g., '腾讯控股')."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_hk_daily_price",
            "description": "Get historical daily price data for a Hong Kong stock within a date range (Open, High, Low, Close, Vol).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {
                        "type": "string",
                        "description": "The Hong Kong stock code (e.g., '00700.HK')."
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
            "name": "get_us_stock_basic",
            "description": "Get basic information about a US stock (e.g., industry, listing date). You can search by stock name or code (e.g., 'AAPL.O' for Apple).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {
                        "type": "string",
                        "description": "The US stock code (e.g., 'AAPL.O', 'TSLA.O')."
                    },
                    "name": {
                        "type": "string",
                        "description": "The US stock name (e.g., 'Apple Inc')."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_us_daily_price",
            "description": "Get historical daily price data for a US stock within a date range (Open, High, Low, Close, Vol).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {
                        "type": "string",
                        "description": "The US stock code (e.g., 'AAPL.O')."
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
            "name": "get_hk_realtime_price",
            "description": "Get the latest real-time price data for a Hong Kong stock. If realtime data is unavailable, returns the latest daily price data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {
                        "type": "string",
                        "description": "The Hong Kong stock code (e.g., '00700.HK')."
                    }
                },
                "required": ["ts_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_us_realtime_price",
            "description": "Get the latest real-time price data for a US stock. If realtime data is unavailable, returns the latest daily price data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {
                        "type": "string",
                        "description": "The US stock code (e.g., 'AAPL.O')."
                    }
                },
                "required": ["ts_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_etf_basic",
            "description": "Get basic information about an ETF (Exchange Traded Fund). You can search by ETF code or name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {
                        "type": "string",
                        "description": "The ETF code (e.g., '510330.SH')."
                    },
                    "name": {
                        "type": "string",
                        "description": "The ETF name (e.g., '沪深300ETF')."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_etf_daily_price",
            "description": "Get historical daily price data for an ETF within a date range (Open, High, Low, Close, Vol).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {
                        "type": "string",
                        "description": "The ETF code (e.g., '510330.SH')."
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
            "name": "get_cb_basic",
            "description": "Get basic information about a convertible bond (可转债). You can search by code or name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {
                        "type": "string",
                        "description": "The convertible bond code (e.g., '123456.SH')."
                    },
                    "name": {
                        "type": "string",
                        "description": "The convertible bond name."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_cb_daily_price",
            "description": "Get historical daily price data for a convertible bond within a date range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {
                        "type": "string",
                        "description": "The convertible bond code (e.g., '123456.SH')."
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
            "name": "get_futures_basic",
            "description": "Get basic information about a futures contract. You can search by code, exchange, or name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {
                        "type": "string",
                        "description": "The futures code (e.g., 'CU2412.SHF' for copper futures)."
                    },
                    "exchange": {
                        "type": "string",
                        "description": "Exchange code: SHF (Shanghai Futures Exchange), DCE (Dalian Commodity Exchange), CZE (Zhengzhou Commodity Exchange), INE (Shanghai International Energy Exchange)."
                    },
                    "name": {
                        "type": "string",
                        "description": "The futures name."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_futures_daily_price",
            "description": "Get historical daily price data for a futures contract within a date range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ts_code": {
                        "type": "string",
                        "description": "The futures code (e.g., 'CU2412.SHF')."
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYYMMDD format. Defaults to 30 days ago."
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYYMMDD format. Defaults to today."
                    },
                    "exchange": {
                        "type": "string",
                        "description": "Exchange code (SHF, DCE, CZE, INE)."
                    }
                },
                "required": ["ts_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_macro_gdp",
            "description": "Get GDP (Gross Domestic Product) macroeconomic data for China. Supports annual and quarterly data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "description": "Specific period in YYYY format (e.g., '2024') or YYYYQ format (e.g., '2024Q1'). For quarterly, only the year part is used."
                    },
                    "start_period": {
                        "type": "string",
                        "description": "Start period in YYYY format (e.g., '2022') or YYYYQ format (e.g., '2022Q1')."
                    },
                    "end_period": {
                        "type": "string",
                        "description": "End period in YYYY format (e.g., '2024') or YYYYQ format (e.g., '2024Q4')."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_macro_cpi",
            "description": "Get CPI (Consumer Price Index) macroeconomic data for China.",
            "parameters": {
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "description": "Specific period in YYYYMM format (e.g., '202412')."
                    },
                    "start_period": {
                        "type": "string",
                        "description": "Start period in YYYYMM format."
                    },
                    "end_period": {
                        "type": "string",
                        "description": "End period in YYYYMM format."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_macro_m2",
            "description": "Get M2 money supply macroeconomic data for China.",
            "parameters": {
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "description": "Specific period in YYYYMM format (e.g., '202412')."
                    },
                    "start_period": {
                        "type": "string",
                        "description": "Start period in YYYYMM format."
                    },
                    "end_period": {
                        "type": "string",
                        "description": "End period in YYYYMM format."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_macro_interest_rate",
            "description": "Get interest rate data (SHIBOR, deposit rate, loan rate) for China.",
            "parameters": {
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "description": "Specific period in YYYYMM format."
                    },
                    "start_period": {
                        "type": "string",
                        "description": "Start period in YYYYMM format."
                    },
                    "end_period": {
                        "type": "string",
                        "description": "End period in YYYYMM format."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_global_index_comparison",
            "description": "Compare global market indices (e.g., Shanghai Composite, Shenzhen Component, Hang Seng Index, S&P 500).",
            "parameters": {
                "type": "object",
                "properties": {
                    "indices": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of index codes to compare (e.g., ['000001.SH', '399001.SZ', 'HSI.HI', 'SPX.GI']). If not provided, defaults to major indices."
                    }
                },
                "required": []
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
                    "net_profit_min": {"type": "number", "description": "Minimum Net Profit (TTM) (in 10k/Wan CNY). Estimated from MV/PE."},
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
            "name": "get_long_tail_stocks",
            "description": "Discover long-tail, neglected stocks or hidden champions. Screen for low turnover, small/mid cap, good value, and optionally consolidation or volume spikes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_mv": {"type": "number", "description": "Min Market Value in Yi (100M). Default 10."},
                    "max_mv": {"type": "number", "description": "Max Market Value in Yi (100M). Default 200 (Mid Cap)."},
                    "max_pe": {"type": "number", "description": "Max PE Ratio (TTM). Default 40."},
                    "max_pb": {"type": "number", "description": "Max PB Ratio. Default 5."},
                    "max_turnover": {"type": "number", "description": "Max Turnover Rate (%). Default 3.0 (Neglected)."},
                    "check_consolidation": {"type": "boolean", "description": "If true, checks for long-term low volatility (consolidation)."},
                    "check_volume_spike": {"type": "boolean", "description": "If true, checks for recent abnormal volume spike."},
                    "limit": {"type": "integer", "description": "Max results. Default 20."}
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
                        "enum": [
                            "ma_cross",
                            "macd",
                            "rsi",
                            "kdj",
                            "boll_reversion",
                            "boll_breakout",
                            "momentum_roc",
                            "donchian_breakout",
                            "turtle",
                            "adx_macd",
                            "triple_ma",
                            "ema_sma_bias",
                            "cci",
                            "williams_r",
                            "stochastic",
                            "rsi_ma200",
                            "volume_breakout",
                            "obv_cross",
                            "vwap_deviation",
                            "ma_cross_atr_stop",
                            "vol_target_ma_cross",
                            "kelly_ma_cross",
                            "cross_section_momentum"
                        ],
                        "description": "Built-in strategies: MA/MACD/RSI/KDJ/BOLL/ROC; donchian_breakout; turtle; adx_macd; triple_ma; ema_sma_bias; cci; williams_r; stochastic; rsi_ma200; volume_breakout; obv_cross; vwap_deviation; ma_cross_atr_stop; vol_target_ma_cross; kelly_ma_cross; cross_section_momentum (multi-asset only, error on single-symbol)."
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
                        "description": "Strategy-specific params, e.g. donchian_breakout {channel_period}; turtle {entry_period, exit_period, atr_stop_mult, atr_period}; adx_macd {adx_period, min_adx, fast_period, slow_period, signal_period}; triple_ma {short_window, mid_window, long_window}; ema_sma_bias {ema_span, sma_window, bias_threshold}; cci/williams_r/stochastic periods; rsi_ma200 {ma_window, window, lower, upper}; volume_breakout {breakout_period, vol_ma_period, volume_mult, exit_period}; obv_cross {obv_ma_period}; vwap_deviation {period, deviation}; ma_cross_atr_stop {atr_stop_mult, atr_period, short_window, long_window}; vol_target_ma_cross {risk_budget_pct, max_fraction, atr_period}; kelly_ma_cross {equity_fraction, short_window, long_window}."
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
        if not arguments.strip():
            arguments = {}
        else:
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                # Try to fix common JSON issues or handle python dict string
                try:
                    import ast
                    # Fallback for single quotes or python-style dicts
                    val = ast.literal_eval(arguments)
                    if isinstance(val, dict):
                        arguments = val
                    else:
                        return "Error: Invalid JSON arguments (not a dict)."
                except:
                    return "Error: Invalid JSON arguments."
    
    if arguments is None:
        arguments = {}

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
    elif tool_name == "get_hk_stock_basic":
        return get_hk_stock_basic(**arguments)
    elif tool_name == "get_hk_daily_price":
        return get_hk_daily_price(**arguments)
    elif tool_name == "get_us_stock_basic":
        return get_us_stock_basic(**arguments)
    elif tool_name == "get_us_daily_price":
        return get_us_daily_price(**arguments)
    elif tool_name == "get_hk_realtime_price":
        return get_hk_realtime_price(**arguments)
    elif tool_name == "get_us_realtime_price":
        return get_us_realtime_price(**arguments)
    elif tool_name == "get_etf_basic":
        return get_etf_basic(**arguments)
    elif tool_name == "get_etf_daily_price":
        return get_etf_daily_price(**arguments)
    elif tool_name == "get_cb_basic":
        return get_cb_basic(**arguments)
    elif tool_name == "get_cb_daily_price":
        return get_cb_daily_price(**arguments)
    elif tool_name == "get_futures_basic":
        return get_futures_basic(**arguments)
    elif tool_name == "get_futures_daily_price":
        return get_futures_daily_price(**arguments)
    elif tool_name == "get_macro_gdp":
        return get_macro_gdp(**arguments)
    elif tool_name == "get_macro_cpi":
        return get_macro_cpi(**arguments)
    elif tool_name == "get_macro_m2":
        return get_macro_m2(**arguments)
    elif tool_name == "get_macro_interest_rate":
        return get_macro_interest_rate(**arguments)
    elif tool_name == "get_global_index_comparison":
        return get_global_index_comparison(**arguments)
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
    elif tool_name == "get_long_tail_stocks":
        return get_long_tail_stocks(**arguments)
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
