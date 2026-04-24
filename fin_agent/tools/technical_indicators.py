import pandas as pd
import numpy as np
import tushare as ts
from fin_agent.config import Config
from datetime import datetime, timedelta
import json

def get_pro():
    ts.set_token(Config.TUSHARE_TOKEN)
    return ts.pro_api()

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate MACD, Signal, and Hist.
    """
    # EMA12
    ema12 = df['close'].ewm(span=fast_period, adjust=False).mean()
    # EMA26
    ema26 = df['close'].ewm(span=slow_period, adjust=False).mean()
    # DIF
    dif = ema12 - ema26
    # DEA
    dea = dif.ewm(span=signal_period, adjust=False).mean()
    # MACD Hist
    macd_hist = (dif - dea) * 2
    
    df['dif'] = dif
    df['dea'] = dea
    df['macd'] = macd_hist
    return df

def calculate_rsi(df, period=14):
    """
    Calculate RSI.
    """
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def calculate_kdj(df, k_period=9, d_period=3, j_period=3):
    """
    Calculate KDJ.
    """
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
    
    k_values = []
    d_values = []
    
    k = 50
    d = 50
    
    for r in rsv:
        if pd.isna(r):
            k_values.append(np.nan)
            d_values.append(np.nan)
            continue
        k = (2/3) * k + (1/3) * r
        d = (2/3) * d + (1/3) * k
        k_values.append(k)
        d_values.append(d)
        
    df['k'] = k_values
    df['d'] = d_values
    df['j'] = 3 * df['k'] - 2 * df['d']
    return df

def calculate_boll(df, period=20, std_dev=2):
    """
    Calculate Bollinger Bands.
    """
    df['boll_mid'] = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    df['boll_upper'] = df['boll_mid'] + (std * std_dev)
    df['boll_lower'] = df['boll_mid'] - (std * std_dev)
    return df


def calculate_atr(df, period=14):
    """Average True Range (Wilder 风格平滑)."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    return df


def calculate_adx_di(df, period=14):
    """ADX 与 +DI / -DI（日线近似）。"""
    high = df["high"]
    low = df["low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / (atr + 1e-12))
    minus_di = 100 * (minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / (atr + 1e-12))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    df["adx"] = dx.ewm(alpha=1.0 / period, adjust=False).mean()
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    return df


def calculate_cci(df, period=20):
    """Commodity Channel Index."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    sma_tp = tp.rolling(period).mean()
    md = tp.rolling(period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))),
        raw=True,
    )
    df["cci"] = (tp - sma_tp) / (0.015 * md + 1e-12)
    return df


def calculate_williams_r(df, period=14):
    """Williams %R，范围约 [-100, 0]。"""
    highest = df["high"].rolling(period).max()
    lowest = df["low"].rolling(period).min()
    df["willr"] = -100.0 * (highest - df["close"]) / (highest - lowest + 1e-12)
    return df


def calculate_stochastic(df, k_period=14, d_period=3):
    """随机指标 %K / %D（列名 st_k, st_d，避免与 KDJ 的 k、d 冲突）。"""
    lowest_low = df["low"].rolling(k_period).min()
    highest_high = df["high"].rolling(k_period).max()
    df["st_k"] = 100.0 * (df["close"] - lowest_low) / (highest_high - lowest_low + 1e-12)
    df["st_d"] = df["st_k"].rolling(d_period).mean()
    return df


def calculate_obv(df):
    """能量潮 OBV。"""
    direction = np.sign(df["close"].diff().fillna(0.0))
    vol = df["vol"].fillna(0.0)
    df["obv"] = (direction * vol).cumsum()
    return df


def calculate_vwap_ma(df, period=20):
    """日频 VWAP 近似：N 日 (典型价×成交量) / 成交量 的滚动和。"""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    v = df["vol"].fillna(0.0)
    num = (tp * v).rolling(period).sum()
    den = v.rolling(period).sum()
    df["vwap_ma"] = num / (den + 1e-12)
    return df


def detect_patterns(df):
    """
    Detect technical patterns from a dataframe containing indicators.
    Returns a dictionary of detected patterns for the latest date.
    """
    if df.empty or len(df) < 2:
        return {}
        
    # Ensure data is sorted by date ascending for calculation
    # (Though get_technical_indicators uses ascending for calculation, it returns descending or sorts it again inside)
    # Let's ensure we are working with the end of the series (latest data)
    
    # Check sort order. If the first date is newer than the last, it's descending.
    if df.iloc[0]['trade_date'] > df.iloc[-1]['trade_date']:
        # Descending, so row 0 is latest, row 1 is yesterday
        curr = df.iloc[0]
        prev = df.iloc[1]
    else:
        # Ascending, so last row is latest
        curr = df.iloc[-1]
        prev = df.iloc[-2]

    patterns = []
    
    # --- MACD Patterns ---
    # Golden Cross: Previous DIF < DEA and Current DIF > DEA
    if prev['dif'] < prev['dea'] and curr['dif'] > curr['dea']:
        patterns.append("MACD_Golden_Cross (MACD金叉)")
    # Dead Cross: Previous DIF > DEA and Current DIF < DEA
    if prev['dif'] > prev['dea'] and curr['dif'] < curr['dea']:
        patterns.append("MACD_Dead_Cross (MACD死叉)")
        
    # --- KDJ Patterns ---
    # Golden Cross: K crosses above D
    if prev['k'] < prev['d'] and curr['k'] > curr['d']:
        patterns.append("KDJ_Golden_Cross (KDJ金叉)")
    # Dead Cross: K crosses below D
    if prev['k'] > prev['d'] and curr['k'] < curr['d']:
        patterns.append("KDJ_Dead_Cross (KDJ死叉)")
    # Overbought/Oversold
    if curr['k'] > 80 or curr['d'] > 80:
         patterns.append("KDJ_Overbought (KDJ超买)")
    if curr['k'] < 20 or curr['d'] < 20:
         patterns.append("KDJ_Oversold (KDJ超卖)")

    # --- RSI Patterns ---
    if curr['rsi'] > 70:
        patterns.append("RSI_Overbought (RSI超买)")
    if curr['rsi'] < 30:
        patterns.append("RSI_Oversold (RSI超卖)")
        
    # --- Bollinger Bands ---
    if curr['close'] > curr['boll_upper']:
        patterns.append("BOLL_Upper_Break (突破布林上轨)")
    if curr['close'] < curr['boll_lower']:
        patterns.append("BOLL_Lower_Break (跌破布林下轨)")
        
    return {
        "trade_date": curr['trade_date'],
        "patterns": patterns,
        "signals": {
            "macd": "bullish" if curr['dif'] > curr['dea'] else "bearish",
            "rsi": curr['rsi'],
            "kdj": "bullish" if curr['k'] > curr['d'] else "bearish"
        }
    }

def get_technical_indicators(ts_code, start_date=None, end_date=None):
    """
    Get technical indicators (MACD, RSI, KDJ, BOLL) for a stock.
    Returns the last 5 records by default to save token usage, 
    but calculates based on a longer history to ensure accuracy.
    """
    # Fetch enough history for accurate calculation (at least 60-90 days)
    if not end_date:
        end_date = datetime.now().strftime('%Y%m%d')
    
    # Logic: Fetch ~200 days of data to calculate indicators properly
    calc_start_date = (datetime.strptime(end_date, '%Y%m%d') - timedelta(days=365)).strftime('%Y%m%d')
    
    try:
        pro = get_pro()
        df = pro.daily(ts_code=ts_code, start_date=calc_start_date, end_date=end_date)
        
        if df.empty:
            return f"No daily data found for {ts_code} to calculate indicators."
        
        # Sort ascending for calculation
        df = df.sort_values('trade_date', ascending=True).reset_index(drop=True)
        
        # Calculate Indicators
        df = calculate_macd(df)
        df = calculate_rsi(df)
        df = calculate_kdj(df)
        df = calculate_boll(df)
        
        # Format columns
        # Keep trade_date, close, and indicators
        cols = ['trade_date', 'close', 'dif', 'dea', 'macd', 'rsi', 'k', 'd', 'j', 'boll_upper', 'boll_mid', 'boll_lower']
        result_df = df[cols].copy()
        
        # Round values
        for col in cols:
            if col != 'trade_date':
                result_df[col] = result_df[col].round(3)
        
        # Sort descending again for display (newest first)
        result_df = result_df.sort_values('trade_date', ascending=False)
        
        # Filter by requested start_date if provided
        if start_date:
             result_df = result_df[result_df['trade_date'] >= start_date]
        else:
             # Default return last 10 days to avoid token limit overflow
             result_df = result_df.head(10)
             
        return result_df.to_json(orient='records', force_ascii=False)
        
    except Exception as e:
        return f"Error calculating technical indicators: {str(e)}"

def get_technical_patterns(ts_code):
    """
    Identify technical patterns (Golden Cross, Overbought/Oversold, etc.) for a stock based on latest data.
    """
    end_date = datetime.now().strftime('%Y%m%d')
    calc_start_date = (datetime.strptime(end_date, '%Y%m%d') - timedelta(days=365)).strftime('%Y%m%d')
    
    try:
        pro = get_pro()
        df = pro.daily(ts_code=ts_code, start_date=calc_start_date, end_date=end_date)
        
        if df.empty:
            return f"No daily data found for {ts_code}."
            
        # Sort ascending for calculation
        df = df.sort_values('trade_date', ascending=True).reset_index(drop=True)
        
        # Calculate Indicators
        df = calculate_macd(df)
        df = calculate_rsi(df)
        df = calculate_kdj(df)
        df = calculate_boll(df)
        
        # Detect patterns
        result = detect_patterns(df)
        
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return f"Error identifying technical patterns: {str(e)}"
