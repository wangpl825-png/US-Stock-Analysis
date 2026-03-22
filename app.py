import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from modules.risk_model import PortfolioRiskModel
from modules.crash_simulator import CrashProbabilitySimulator
from modules.llm_analyzer import MarketAnalyzer
import os

# --- 1. 頁面與 UI 設定 ---
st.set_page_config(page_title="量化投資風險分析", layout="wide", initial_sidebar_state="expanded")
st.title("📈 美股投資組合風險與壓力測試儀表板")
st.markdown("透過蒙地卡羅模擬 (Monte Carlo) 與歷史壓力測試，評估您的資產配置風險。")

# --- 2. 側邊欄：使用者輸入區 ---
st.sidebar.header("⚙️ 投資策略設定")

# 預設輸入範例
default_tickers = "AAPL, MSFT, TLT"
default_weights = "0.4, 0.4, 0.2"

tickers_input = st.sidebar.text_input("股票代碼 (請以逗號分隔)", value=default_tickers)
weights_input = st.sidebar.text_input("對應權重 (請以逗號分隔，總和需為1)", value=default_weights)
initial_investment = st.sidebar.number_input("總投資金額 (USD)", min_value=1000, value=10000, step=1000)
days_ahead = st.sidebar.slider("預測天數 (交易日)", min_value=5, max_value=60, value=21)

analyze_button = st.sidebar.button("🚀 開始風險運算")

# --- 3. 主畫面：運算與視覺化 ---
if analyze_button:
    # 處理輸入字串
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    try:
        weights = [float(w.strip()) for w in weights_input.split(',')]
    except ValueError:
        st.error("⚠️ 權重格式錯誤，請確保輸入的是數字並以逗號分隔。")
        st.stop()

    # 基礎防呆檢查
    if len(tickers) != len(weights):
        st.error(f"⚠️ 股票數量 ({len(tickers)}) 與權重數量 ({len(weights)}) 不一致！")
        st.stop()
    if abs(sum(weights) - 1.0) > 1e-6:
        st.error(f"⚠️ 權重總和必須為 1，目前總和為 {sum(weights):.2f}")
        st.stop()

    with st.spinner('⏳ 正在抓取歷史數據並執行 10,000 次蒙地卡羅模擬...'):
        try:
            # 初始化模型
            model = PortfolioRiskModel(tickers=tickers, weights=weights, initial_investment=initial_investment)
            
            # 執行模塊三 A: 常態預測
            percentiles, loss_prob, price_matrix = model.run_monte_carlo(days_ahead=days_ahead)
            
            # 執行模塊三 B: 壓力測試
            beta, stress_res = model.run_stress_test()

            st.success("✅ 運算完成！")

            # --- 建立頁籤以分類資訊 (適合手機瀏覽) ---
            tab1, tab2, tab3 = st.tabs(["🎲 蒙地卡羅預測", "⛈️ 壓力測試", "🚨 崩盤預警系統"])

            with tab1:
                st.subheader(f"未來 {days_ahead} 個交易日後之資產分佈預測")
                
                # 指標卡片
                col1, col2, col3 = st.columns(3)
                col1.metric("P50 (中位數預期)", f"${percentiles['P50 (中位數)']:,.0f}")
                col2.metric("P01 (極端左尾風險)", f"${percentiles['P01 (極度悲觀)']:,.0f}")
                col3.metric("預期虧損機率", f"{loss_prob:.1f}%", delta_color="inverse")

                # Plotly 視覺化：最終資金分佈直方圖
                final_values = price_matrix[-1]
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=final_values, 
                    nbinsx=100, 
                    marker_color='#636EFA', 
                    opacity=0.75,
                    name="模擬結果分佈"
                ))
                
                # 加入初始資金的垂直線作為損益兩平點
                fig_hist.add_vline(x=initial_investment, line_dash="dash", line_color="red", 
                                   annotation_text="投入本金", annotation_position="top right")
                
                fig_hist.update_layout(
                    title_text="10,000 次模擬之最終資產價值分佈",
                    xaxis_title="資產價值 (USD)",
                    yaxis_title="發生次數",
                    bargap=0.05,
                    template="plotly_white"
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with tab2:
                st.subheader("假設明天發生歷史級別股災...")
                st.write(f"您當前投資組合的 **Beta 值估算為: {beta:.2f}** (相對於 SPY)")
                
                # 使用 DataFrame 呈現壓力測試結果，較為美觀
                stress_df = pd.DataFrame(stress_res).T
                stress_df.columns = ["預估跌幅", "預估資金損失"]
                st.dataframe(stress_df, use_container_width=True)
                
                if beta > 1.2:
                    st.warning("⚠️ 您的投資組合波動性顯著高於大盤，在系統性風險發生時將承受較大回撤，建議增加低相關性資產（如美債）進行避險。")
                elif beta < 0.8:
                    st.info("💡 您的投資組合具備一定的抗跌特性。")

            with tab3:
                st.subheader("未來一個月系統性崩盤機率評估")
                st.markdown("綜合大盤左尾風險 (Left-tail Risk) 與總體經濟指標 (Macro Indicators) 之量化預測。")
                
                with st.spinner('正在分析總體經濟指標與市場波動率...'):
                    crash_sim = CrashProbabilitySimulator()
                    final_prob, left_tail, penalty, messages = crash_sim.get_crash_probability()
                    
                    # 依據機率給予不同的顏色與警告層級
                    if final_prob < 15:
                        status_color = "normal"
                        risk_level = "🟢 低風險 (Low Risk)"
                    elif final_prob < 40:
                        status_color = "off"
                        risk_level = "🟡 中度警戒 (Elevated Risk)"
                    else:
                        status_color = "inverse"
                        risk_level = "🔴 極度危險 (Critical Risk)"

                    # 顯示大指標
                    st.metric(label=f"當前市場狀態: {risk_level}", value=f"{final_prob:.1f}%", delta=f"總經懲罰權重: +{penalty}%", delta_color=status_color)
                    st.progress(final_prob / 100.0)
                    
                    st.divider()
                    st.markdown("### 📊 系統診斷報告 (Diagnostic Report)")
                    st.write(f"- **純數學極端左尾機率**: {left_tail:.2f}% (大盤單月暴跌 15% 以上之自然機率)")
                    
                    if messages:
                        st.markdown("#### ⚠️ 觸發之總經警報：")
                        for msg in messages:
                            st.warning(msg)
                    else:
                        st.success("✅ 目前總體經濟與情緒指標未出現明顯異常，各項 Vital Signs 穩定。")

                    # --- AI 避險建議區塊 ---
                    st.divider()
                    st.markdown("### 🤖 AI 量化顧問避險建議")
                    
                    with st.spinner('AI 正在閱讀最新市場新聞並計算專屬策略...'):
                        try:
                            if "GEMINI_API_KEY" in st.secrets:
                                os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
                            
                            analyzer = MarketAnalyzer()
                            strategy = analyzer.generate_hedging_strategy(final_prob, beta)
                            st.info(strategy)
                        except Exception as e:
                            st.error(f"無法啟動 AI 顧問，請確認 API Key。錯誤訊息: {e}")

        except Exception as e:
            st.error(f"❌ 運算過程中發生錯誤: {e}")
            st.info("請檢查股票代碼是否正確（如美股代碼），或稍後再試。")
