import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class PortfolioRiskModel:
    def __init__(self, tickers: list, weights: list, initial_investment: float):
        """
        初始化投資組合
        :param tickers: 股票代碼列表 e.g., ['AAPL', 'MSFT', 'TLT']
        :param weights: 對應權重 e.g., [0.4, 0.4, 0.2]
        :param initial_investment: 總投資金額 e.g., 10000
        """
        assert len(tickers) == len(weights), "股票代碼與權重數量必須一致"
        assert abs(sum(weights) - 1.0) < 1e-6, "權重總和必須為 1"
        
        self.tickers = tickers
        self.weights = np.array(weights)
        self.initial_investment = initial_investment
        self.historical_data = self._fetch_historical_data()
        self.portfolio_returns = self._calculate_portfolio_returns()

    def _fetch_historical_data(self, days=252):
        """獲取過去 252 個交易日的歷史收盤價，並加入 SPY 作為 Benchmark"""
        end_date = datetime.today()
        start_date = end_date - timedelta(days=days + 100) # 多抓幾天確保有 252 個交易日
        
        # 加上 SPY 作為計算 Beta 的基準
        all_tickers = self.tickers + ['SPY']
        data = yfinance.download(all_tickers, start=start_date, end=end_date)['Adj Close']
        
        # 取最近 252 個交易日
        return data.tail(252).dropna()

    def _calculate_portfolio_returns(self):
        """計算投資組合的每日歷史報酬率"""
        # 計算每檔股票的日報酬率
        daily_returns = self.historical_data.pct_change().dropna()
        
        # 提取投資組合部分的報酬率 (排除 SPY)
        portfolio_assets_returns = daily_returns[self.tickers]
        
        # 根據權重計算投資組合的總體日報酬率
        portfolio_daily_returns = portfolio_assets_returns.dot(self.weights)
        return pd.DataFrame({
            'Portfolio': portfolio_daily_returns,
            'SPY': daily_returns['SPY']
        })

    def run_monte_carlo(self, days_ahead=21, simulations=10000):
        """
        子系統 A: 蒙地卡羅常態預測
        使用幾何布朗運動 (GBM) 進行預測
        """
        returns = self.portfolio_returns['Portfolio']
        mu = returns.mean()
        sigma = returns.std()
        
        # 建立模擬矩陣 (10000次模擬, 每次21天)
        # Z 為標準常態分佈隨機變數
        Z = np.random.normal(0, 1, (days_ahead, simulations))
        
        # GBM 每日漂移率與隨機衝擊
        daily_drift = mu - (0.5 * sigma ** 2)
        daily_shock = sigma * Z
        
        # 計算每日價格變化率
        price_paths = np.exp(daily_drift + daily_shock)
        
        # 建立價格路徑矩陣，並將初始值設為 initial_investment
        price_matrix = np.zeros_like(price_paths)
        price_matrix[0] = self.initial_investment * price_paths[0]
        
        for t in range(1, days_ahead):
            price_matrix[t] = price_matrix[t-1] * price_paths[t]
            
        # 獲取最終第 21 天的所有模擬結果
        final_values = price_matrix[-1]
        
        # 計算分位數
        percentiles = {
            "P99 (極度樂觀)": np.percentile(final_values, 99),
            "P90 (樂觀)": np.percentile(final_values, 90),
            "P50 (中位數)": np.percentile(final_values, 50),
            "P10 (悲觀)": np.percentile(final_values, 10),
            "P01 (極度悲觀)": np.percentile(final_values, 1)
        }
        
        # 尋找開始出現虧損的百分位 (Break-even Percentile)
        # 排序最終價值，找出第一個大於等於初始投資額的位置
        sorted_values = np.sort(final_values)
        loss_probability = (sorted_values < self.initial_investment).mean() * 100
        
        return percentiles, loss_probability, price_matrix

    def run_stress_test(self):
        """
        子系統 B: 歷史壓力測試 (Beta Based)
        """
        returns = self.portfolio_returns
        
        # 計算共變異數矩陣來求 Beta = Cov(Port, SPY) / Var(SPY)
        cov_matrix = np.cov(returns['Portfolio'], returns['SPY'])
        portfolio_beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        
        # 定義過往重大崩盤事件中 SPY 的最大回撤率 (Max Drawdown)
        historical_crashes = {
            "1973-1974 石油危機": -0.48,
            "2000-2003 網路泡沫": -0.49,
            "2007-2009 金融海嘯": -0.55,
            "2020 COVID-19 股災": -0.34
        }
        
        stress_results = {}
        for event, market_drop in historical_crashes.items():
            # 預估投資組合跌幅 = 大盤跌幅 * Beta
            expected_drop = market_drop * portfolio_beta
            expected_loss_value = self.initial_investment * expected_drop
            stress_results[event] = {
                "預估跌幅": f"{expected_drop*100:.2f}%",
                "預估資金損失": f"${expected_loss_value:,.2f}"
            }
            
        return portfolio_beta, stress_results

# --- 測試區塊 (直接執行此檔案可看結果) ---
if __name__ == "__main__":
    print("初始化投資組合 (60% 蘋果, 40% 微軟), 總資金 $10,000...")
    model = PortfolioRiskModel(tickers=['AAPL', 'MSFT'], weights=[0.6, 0.4], initial_investment=10000)
    
    print("\n--- 執行常態 Monte Carlo 模擬 (10,000次, 21天) ---")
    percentiles, loss_prob, _ = model.run_monte_carlo()
    for k, v in percentiles.items():
        print(f"{k}: ${v:,.2f}")
    print(f"⚠️ 模型預測一個月後虧損機率為: {loss_prob:.2f}% (約從 P{100-loss_prob:.0f} 開始出現虧損)")
    
    print("\n--- 執行歷史壓力測試 (Stress Test) ---")
    beta, stress_res = model.run_stress_test()
    print(f"投資組合 Beta 值: {beta:.2f}")
    for event, res in stress_res.items():
        print(f"發生 {event} 級別崩盤: 預期跌幅 {res['預估跌幅']}, 預估損失 {res['預估資金損失']}")
