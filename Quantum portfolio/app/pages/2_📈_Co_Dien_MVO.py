import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from app.components.sidebar import render_sidebar
from utils import get_stock_data, calculate_returns, optimize_classical_mvo, generate_efficient_frontier

st.set_page_config(page_title="Tối Ưu Cổ Điển | Quantum Portfolio", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .metric-card {
        background-color: rgba(99, 102, 241, 0.05);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 1rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #4f46e5;
        margin: 10px 0;
    }
    .metric-desc {
        font-size: 0.85rem;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Tối Ưu Cổ Điển (Classical MVO)")
st.markdown("Áp dụng mô hình **Markowitz Mean-Variance Optimization** để tìm danh mục tối ưu, được giải quyết bằng thư viện CVXPY.")

selected_tickers, start_date, end_date, risk_aversion = render_sidebar()

if not selected_tickers:
    st.warning("Vui lòng chọn ít nhất một cổ phiếu từ thanh bên.")
    st.stop()

with st.spinner("Đang tính toán tối ưu hóa cổ điển..."):
    prices_df = get_stock_data(selected_tickers, start_date, end_date)
    
if prices_df.empty:
    st.error("Không thể tải dữ liệu.")
    st.stop()

daily_returns, mean_returns, cov_matrix = calculate_returns(prices_df)

# Run MVO for the specific risk aversion
weights = optimize_classical_mvo(mean_returns, cov_matrix, risk_aversion)

# Calculate metrics for the optimal portfolio
expected_return = np.sum(mean_returns * weights)
expected_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
sharpe_ratio = expected_return / expected_volatility

# KPI Cards
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>Tỷ Suất Sharpe</div>
        <div class='metric-value'>{sharpe_ratio:.4f}</div>
        <div class='metric-desc'>Lợi nhuận điều chỉnh rủi ro</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>Lợi Nhuận Kỳ Vọng</div>
        <div class='metric-value'>{(expected_return * 100):.2f}%</div>
        <div class='metric-desc'>Lợi suất ước tính hàng năm</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>Độ Biến Động (Rủi Ro)</div>
        <div class='metric-value'>{(expected_volatility * 100):.2f}%</div>
        <div class='metric-desc'>Độ lệch chuẩn hàng năm</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
c1, c2 = st.columns([1, 1])

with c1:
    st.subheader("Phân Bổ Danh Mục (Optimal Weights)")
    
    # Filter out very small weights (less than 1%) for a cleaner pie chart
    weight_df = pd.DataFrame({"Cổ phiếu": selected_tickers, "Tỷ trọng": weights})
    weight_df["Tỷ trọng (%)"] = weight_df["Tỷ trọng"] * 100
    display_df = weight_df[weight_df["Tỷ trọng (%)"] > 0.5].sort_values(by="Tỷ trọng", ascending=False)
    
    fig_pie = px.pie(
        display_df, 
        values="Tỷ trọng", 
        names="Cổ phiếu",
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.Purp
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)
    
    st.dataframe(
        weight_df.sort_values(by="Tỷ trọng", ascending=False).style.format({"Tỷ trọng": "{:.4f}", "Tỷ trọng (%)": "{:.2f}%"}),
        use_container_width=True
    )

with c2:
    st.subheader("Đường Biên Hiệu Quả (Efficient Frontier)")
    
    with st.spinner("Đang vẽ đường biên hiệu quả..."):
        # Generate frontier
        frontier_vols, frontier_rets, opt_vol, opt_ret = generate_efficient_frontier(
            mean_returns, cov_matrix, risk_aversion
        )
        
        fig_ef = go.Figure()
        
        # Plot the frontier curve
        fig_ef.add_trace(go.Scatter(
            x=frontier_vols, 
            y=frontier_rets, 
            mode='lines',
            line=dict(color='indigo', width=3),
            name='Đường biên hiệu quả'
        ))
        
        # Highlight current optimal portfolio
        fig_ef.add_trace(go.Scatter(
            x=[expected_volatility], 
            y=[expected_return],
            mode='markers+text',
            marker=dict(color='red', size=15, symbol='star'),
            text=['Danh mục hiện tại'],
            textposition='top left',
            name=f'Tối ưu (γ={risk_aversion})'
        ))
        
        # Plot individual assets
        asset_vols = np.sqrt(np.diag(cov_matrix))
        asset_rets = mean_returns.values
        fig_ef.add_trace(go.Scatter(
            x=asset_vols,
            y=asset_rets,
            mode='markers+text',
            marker=dict(color='blue', size=8),
            text=selected_tickers,
            textposition='bottom center',
            name='Cổ phiếu đơn lẻ'
        ))
        
        fig_ef.update_layout(
            title="Đường Biên Hiệu Quả Markowitz",
            xaxis_title="Độ biến động (Rủi ro)",
            yaxis_title="Lợi nhuận kỳ vọng",
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig_ef, use_container_width=True)
