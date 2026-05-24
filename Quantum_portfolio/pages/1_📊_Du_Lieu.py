
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from components.sidebar import render_sidebar
from utils import get_stock_data, calculate_returns

st.set_page_config(page_title="Khám Phá Dữ Liệu | Quantum Portfolio", page_icon="📊", layout="wide")

# Custom CSS for cards
st.markdown("""
<style>
    .metric-card {
        background-color: rgba(99, 102, 241, 0.05);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #4f46e5;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Khám Phá Dữ Liệu (Data Explorer)")
st.markdown("Phân tích dữ liệu lịch sử giá cổ phiếu S&P500 và các đặc trưng thống kê làm đầu vào cho bài toán tối ưu.")

# Render shared sidebar
selected_tickers, start_date, end_date, risk_aversion = render_sidebar()

if not selected_tickers:
    st.warning("Vui lòng chọn ít nhất một cổ phiếu từ thanh bên.")
    st.stop()

with st.spinner("Đang tải dữ liệu từ Yahoo Finance..."):
    prices_df = get_stock_data(selected_tickers, start_date, end_date)

if prices_df.empty:
    st.error("Không thể tải dữ liệu. Vui lòng kiểm tra lại kết nối hoặc danh sách cổ phiếu.")
    st.stop()

daily_returns, mean_returns, cov_matrix = calculate_returns(prices_df)

# Quick stats row
st.markdown("### 📈 Tổng quan")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"<div class='metric-card'><div class='metric-title'>Số Lượng Cổ Phiếu</div><div class='metric-value'>{len(selected_tickers)}</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><div class='metric-title'>Ngày Bắt Đầu</div><div class='metric-value'>{start_date}</div></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><div class='metric-title'>Ngày Kết Thúc</div><div class='metric-value'>{end_date}</div></div>", unsafe_allow_html=True)
with col4:
    trading_days = len(prices_df)
    st.markdown(f"<div class='metric-card'><div class='metric-title'>Số Phiên Giao Dịch</div><div class='metric-value'>{trading_days}</div></div>", unsafe_allow_html=True)

# Tabs for detailed data views
tab1, tab2, tab3 = st.tabs(["Biểu Đồ Giá", "Thống Kê Lợi Nhuận", "Ma Trận Tương Quan"])

with tab1:
    st.subheader("Tăng Trưởng Tích Lũy")
    # Calculate cumulative returns for normalized comparison
    normalized_prices = prices_df / prices_df.iloc[0] * 100
    
    fig = px.line(
        normalized_prices, 
        x=normalized_prices.index, 
        y=normalized_prices.columns,
        title="Biểu đồ giá chuẩn hóa (Cơ sở 100 tại ngày bắt đầu)",
        labels={"value": "Giá trị chuẩn hóa", "variable": "Cổ phiếu", "Date": "Ngày"}
    )
    fig.update_layout(hovermode="x unified", legend_title_text='Mã CP', template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Thống Kê Cơ Bản")
    
    # Calculate volatility (annualized)
    volatility = np.sqrt(np.diag(cov_matrix))
    
    stats_df = pd.DataFrame({
        "Lợi nhuận kỳ vọng (%)": (mean_returns * 100).round(2),
        "Độ biến động/Rủi ro (%)": (volatility * 100).round(2),
        "Sharpe Ratio": (mean_returns / volatility).round(2)
    })
    
    st.dataframe(
        stats_df.style.background_gradient(cmap='viridis', subset=["Lợi nhuận kỳ vọng (%)"])
                      .background_gradient(cmap='Reds', subset=["Độ biến động/Rủi ro (%)"])
                      .background_gradient(cmap='Blues', subset=["Sharpe Ratio"]),
        use_container_width=True
    )
    
    c1, c2 = st.columns(2)
    with c1:
        fig_ret = px.bar(
            x=stats_df.index, 
            y=stats_df["Lợi nhuận kỳ vọng (%)"],
            title="Lợi Nhuận Kỳ Vọng Từng Cổ Phiếu",
            labels={"x": "Cổ phiếu", "y": "Lợi nhuận (%)"},
            color=stats_df["Lợi nhuận kỳ vọng (%)"],
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_ret, use_container_width=True)
    with c2:
        fig_vol = px.bar(
            x=stats_df.index, 
            y=stats_df["Độ biến động/Rủi ro (%)"],
            title="Mức Độ Rủi Ro (Volatility)",
            labels={"x": "Cổ phiếu", "y": "Độ biến động (%)"},
            color=stats_df["Độ biến động/Rủi ro (%)"],
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig_vol, use_container_width=True)

with tab3:
    st.subheader("Ma Trận Hiệp Phương Sai & Tương Quan")
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("**Ma trận Tương quan (Correlation)**")
        corr_matrix = daily_returns.corr()
        fig_corr = px.imshow(
            corr_matrix.round(2), 
            text_auto=True, 
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with c2:
        st.markdown("**Hiệp phương sai (Covariance)**")
        st.dataframe(cov_matrix.style.background_gradient(cmap='coolwarm'), use_container_width=True)
        st.info("💡 Các cổ phiếu có hệ số tương quan thấp (màu xanh đậm) khi kết hợp với nhau sẽ giúp giảm thiểu rủi ro chung của danh mục.")
