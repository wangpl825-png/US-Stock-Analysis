import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class CrashProbabilitySimulator:
    def __init__(self):
        # 基準大盤
        self.market_ticker = 'SPY'
        # 使用 10年期 (^TNX), 13週短債 (^IRX) 計算利差, 以及 VIX (^VIX)
        self.macro_tickers = ['^VIX', '^TNX', '^IRX']

    def fetch_data(self):
        end_date = datetime.today()
        start_date = end_date - timedelta(days=365)
        # 靜默下載避免干擾 UI
        data = yf.download([self.market_ticker] + self.macro_tickers, start=start_date, end=end_date, progress=False)['Adj Close']
        return data.dropna()

    def calculate_left_tail_risk(self, data, days_ahead=21, simulations=5000, crash_threshold=-0.15):
        """
        利用大盤歷史波動，跑 Monte Carlo 計算單月跌幅超過 15% 的極端左尾機率
        """
        spy_returns = data[self.market_ticker].pct_change().dropna()
        mu = spy_returns.mean()
        sigma = spy_returns.std()

        Z = np.random.normal(0, 1, (days_ahead, simulations))
        daily_paths = np.exp((mu - 0.5 * sigma**2) + sigma * Z)
        
        cumulative_returns = np.prod(daily_paths, axis=0) - 1
        crash_prob = np.mean(cumulative_returns < crash_threshold) * 100
        return crash_prob

    def evaluate_macro_penalty(self, data):
        """
        總經與情緒指標惡化懲罰 (Vital Signs Check)
        """
        latest = data.iloc[-1]
        
        vix_current = latest['^VIX']
        vix_mean = data['^VIX'].mean()
        
        yield_10y = latest['^TNX']
        yield_short = latest['^IRX']
        yield_spread = yield_10y - yield_short # 負值代表殖利率倒掛
        
        penalty = 0
        messages = []
        
        # 1. 恐慌指數評估
        if vix_current > 30:
            penalty += 30
            messages.append(f"🚨 **VIX 突破 30 高危線** ({vix_current:.1f})：市場流動性風險劇增，機構正在大舉買入賣權避險。")
        elif vix_current > vix_mean * 1.3:
            penalty += 15
            messages.append(f"⚠️ **VIX 處於相對高位** ({vix_current:.1f})：市場情緒恐慌，波動率擴大。")
            
        # 2. 殖利率倒掛評估 (衰退前兆)
        if yield_spread < 0:
            penalty += 25
            messages.append(f"🚨 **美債殖利率嚴重倒掛** (利差 {yield_spread:.2f}%)：長短期資金成本錯置，為歷史上最準確的經濟衰退與崩盤前兆之一。")
        elif yield_spread < 0.5:
            penalty += 10
            messages.append(f"👀 **美債利差收窄** (利差 {yield_spread:.2f}%)：經濟動能出現疲態，需密切關注聯準會動向。")
            
        return penalty, messages

    def get_crash_probability(self):
        data = self.fetch_data()
        left_tail_prob = self.calculate_left_tail_risk(data)
        macro_penalty, messages = self.evaluate_macro_penalty(data)
        
        # 綜合崩盤機率 (純數學左尾風險 + 總經面懲罰)
        final_prob = min(left_tail_prob + macro_penalty, 100.0)
        
        return final_prob, left_tail_prob, macro_penalty, messages
