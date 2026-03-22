import google.generativeai as genai
import yfinance as yf
import streamlit as st

class MarketAnalyzer:
    def __init__(self):
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except KeyError:
            raise ValueError("找不到 GEMINI_API_KEY，請確認 Streamlit Secrets 設定。")
            
        genai.configure(api_key=api_key)
        
        # --- 🚀 終極解法：動態偵測可用的模型 ---
        available_models = []
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
        except Exception as e:
            raise ValueError(f"無法獲取模型列表，請確認 API Key 權限。錯誤: {e}")
            
        if not available_models:
            raise ValueError("您的 API Key 目前沒有配置任何支援文字生成的模型。")
            
        # 優先尋找 1.5 flash，找不到就用列表裡的第一個可用模型
        target_model_name = available_models[0]
        for name in available_models:
            if 'gemini-1.5-flash' in name:
                target_model_name = name
                break
            elif 'gemini-1.0-pro' in name:
                target_model_name = name
                
        self.model = genai.GenerativeModel(target_model_name)
        self.used_model = target_model_name # 記錄下來給 UI 顯示

    def get_market_news(self, ticker="SPY"):
        """透過 yfinance 免費抓取最新大盤新聞標題"""
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            if news:
                return [item['title'] for item in news[:5]]
            return ["目前無重大新聞"]
        except:
            return ["新聞抓取失敗，請稍後再試"]

    def generate_hedging_strategy(self, crash_prob, portfolio_beta):
        """生成 AI 避險策略"""
        news_titles = self.get_market_news()
        news_text = "\n".join(f"- {title}" for title in news_titles)

        prompt = f"""
        你是一位資深的華爾街量化風險分析師。
        請根據以下最新的美股市場數據與新聞，給出一段簡短、專業且具建設性的「投資組合避險建議」。

        【當前市場狀態】
        - 系統性崩盤預測機率: {crash_prob:.1f}%
        - 客戶投資組合 Beta 值: {portfolio_beta:.2f}

        【最新大盤新聞標題】
        {news_text}

        請用繁體中文回答，包含：
        1. 📊 市場情緒總結 (一句話)。
        2. 🛡️ 具體避險操作建議 (針對此 Beta 值與崩盤機率，給出兩點具體的資產配置建議)。
        """
        try:
            response = self.model.generate_content(prompt)
            # 在策略開頭標示我們到底成功呼叫了哪個模型，方便確認
            return f"*(本策略由 `{self.used_model}` 生成)*\n\n" + response.text
        except Exception as e:
            return f"❌ AI 呼叫失敗！系統回報的真實錯誤訊息為：\n\n`{str(e)}`"
