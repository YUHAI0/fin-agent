import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import json
from fin_agent.config import Config
from fin_agent.tools.technical_indicators import calculate_macd, calculate_rsi, calculate_kdj, calculate_boll

class BacktestEngine:
    def __init__(self, initial_capital=100000, commission=0.0003):
        self.initial_capital = initial_capital
        self.commission = commission
        self.cash = initial_capital
        self.position = 0
        self.history = [] # Trade history
        self.portfolio_values = [] # Daily portfolio values

    def _fetch_data(self, ts_code, start_date, end_date):
        """Fetch daily data using Tushare"""
        try:
            ts.set_token(Config.TUSHARE_TOKEN)
            pro = ts.pro_api()
            
            # Fetch a bit more data before start_date for indicator warm-up
            warmup_start = (datetime.strptime(start_date, '%Y%m%d') - timedelta(days=60)).strftime('%Y%m%d')
            
            df = pro.daily(ts_code=ts_code, start_date=warmup_start, end_date=end_date)
            if df.empty:
                raise ValueError(f"No data found for {ts_code}")
                
            # Sort ascending
            df = df.sort_values('trade_date', ascending=True).reset_index(drop=True)
            return df
        except Exception as e:
            raise e

    def _calculate_indicators(self, df, strategy_config):
        """Calculate indicators needed for the strategy"""
        strategy_type = strategy_config.get('type', 'ma_cross')
        
        if strategy_type == 'ma_cross':
            short_window = int(strategy_config.get('short_window', 5))
            long_window = int(strategy_config.get('long_window', 20))
            
            df['short_ma'] = df['close'].rolling(window=short_window).mean()
            df['long_ma'] = df['close'].rolling(window=long_window).mean()
            
        elif strategy_type == 'macd':
            df = calculate_macd(df)
            
        elif strategy_type == 'rsi':
            window = int(strategy_config.get('window', 14))
            df = calculate_rsi(df, period=window)
            
        return df

    def _generate_signal(self, row, prev_row, strategy_config):
        """Generate Buy (1), Sell (-1), or Hold (0) signal"""
        strategy_type = strategy_config.get('type', 'ma_cross')
        
        if prev_row is None:
            return 0
            
        if strategy_type == 'ma_cross':
            # Golden Cross
            if prev_row['short_ma'] <= prev_row['long_ma'] and row['short_ma'] > row['long_ma']:
                return 1
            # Dead Cross
            elif prev_row['short_ma'] >= prev_row['long_ma'] and row['short_ma'] < row['long_ma']:
                return -1
                
        elif strategy_type == 'macd':
            # MACD Golden Cross (DIF crosses above DEA)
            if prev_row['dif'] <= prev_row['dea'] and row['dif'] > row['dea']:
                return 1
            # MACD Dead Cross
            elif prev_row['dif'] >= prev_row['dea'] and row['dif'] < row['dea']:
                return -1
                
        elif strategy_type == 'rsi':
            lower = int(strategy_config.get('lower', 30))
            upper = int(strategy_config.get('upper', 70))
            
            # Oversold -> Buy
            if prev_row['rsi'] >= lower and row['rsi'] < lower:
                return 1
            # Overbought -> Sell
            elif prev_row['rsi'] <= upper and row['rsi'] > upper:
                return -1
                
        return 0

    def run(self, ts_code, start_date, end_date, strategy_config):
        # 1. Prepare Data
        df = self._fetch_data(ts_code, start_date, end_date)
        df = self._calculate_indicators(df, strategy_config)
        
        # Filter data to match requested start_date (after indicator calculation)
        # But we need prev_row, so we keep one extra row before start_date if possible
        mask = df['trade_date'] >= start_date
        # Find index of first row matching mask
        if not mask.any():
            return {"error": "No data in requested date range"}
            
        start_idx = mask.idxmax()
        if start_idx > 0:
            start_idx -= 1 # Keep one prior row for signal calc
            
        # Iterate
        prev_row = None
        
        for index, row in df.iloc[start_idx:].iterrows():
            # Skip rows before actual start_date for TRADING, but use for signal
            is_trading_period = row['trade_date'] >= start_date
            
            current_price = row['close']
            
            # Record Portfolio Value (Daily close)
            if is_trading_period:
                total_value = self.cash + (self.position * current_price)
                self.portfolio_values.append({
                    'trade_date': row['trade_date'],
                    'value': total_value
                })
            
            # Generate Signal
            signal = self._generate_signal(row, prev_row, strategy_config)
            
            # Execute Trade
            if is_trading_period and signal != 0:
                # Buy
                if signal == 1 and self.cash > 0:
                    # Buy max shares (lots of 100)
                    max_shares = int(self.cash / (current_price * (1 + self.commission)) / 100) * 100
                    if max_shares > 0:
                        cost = max_shares * current_price
                        comm = cost * self.commission
                        self.cash -= (cost + comm)
                        self.position += max_shares
                        self.history.append({
                            'date': row['trade_date'],
                            'action': 'BUY',
                            'price': current_price,
                            'shares': max_shares,
                            'commission': comm
                        })
                # Sell
                elif signal == -1 and self.position > 0:
                    # Sell all
                    revenue = self.position * current_price
                    comm = revenue * self.commission
                    self.cash += (revenue - comm)
                    
                    self.history.append({
                        'date': row['trade_date'],
                        'action': 'SELL',
                        'price': current_price,
                        'shares': self.position,
                        'commission': comm
                    })
                    self.position = 0
            
            prev_row = row
            
        # Final Calculation
        final_value = self.cash + (self.position * df.iloc[-1]['close'])
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Calculate Max Drawdown
        pv_df = pd.DataFrame(self.portfolio_values)
        if not pv_df.empty:
            pv_df['cummax'] = pv_df['value'].cummax()
            pv_df['drawdown'] = (pv_df['cummax'] - pv_df['value']) / pv_df['cummax']
            max_drawdown = pv_df['drawdown'].max()
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
            "strategy": strategy_config.get('type'),
            "initial_capital": self.initial_capital,
            "final_value": round(final_value, 2),
            "total_return_pct": round(total_return * 100, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "trades_count": len(self.history),
            "trades": self.history[-5:],  # Return last 5 trades for brevity
            "equity_curve": equity_curve,
        }

def run_backtest(ts_code, strategy="ma_cross", start_date=None, end_date=None, params=None):
    """
    Wrapper for Tool usage.
    params: JSON string or dict of strategy parameters.
    """
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y%m%d')
        
    # Parse params
    strategy_config = {"type": strategy}
    if params:
        if isinstance(params, str):
            try:
                params_dict = json.loads(params)
                strategy_config.update(params_dict)
            except:
                pass
        elif isinstance(params, dict):
            strategy_config.update(params)
            
    engine = BacktestEngine()
    try:
        result = engine.run(ts_code, start_date, end_date, strategy_config)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error running backtest: {str(e)}"

