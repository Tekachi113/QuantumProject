import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from app.components.sidebar import render_sidebar
from utils import get_stock_data, calculate_returns, solve_qaoa

st.set_page_config(page_title="Tối Ưu Lượng Tử | Quantum Portfolio", page_icon="⚛️", layout="wide")

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
        color: #db2777;
        margin: 10px 0;
    }
    .metric-desc {
        font-size: 0.85rem;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚛️ Tối Ưu Lượng Tử (QAOA)")
st.markdown("Sử dụng **Quantum Approximate Optimization Algorithm (QAOA)** để tìm danh mục tối ưu trên mô phỏng máy tính lượng tử.")

selected_tickers, start_date, end_date, risk_aversion = render_sidebar()

if not selected_tickers:
    st.warning("Vui lòng chọn ít nhất một cổ phiếu từ thanh bên.")
    st.stop()

# Budget slider
budget = st.slider(
    "Số lượng cổ phiếu muốn chọn (Budget B):",
    min_value=1,
    max_value=len(selected_tickers),
    value=max(1, len(selected_tickers) // 2),
    help="QAOA sẽ bị phạt (penalty) nếu chọn số lượng cổ phiếu khác với B."
)

p_layers = st.selectbox(
    "Số lớp QAOA (p):", 
    options=[1, 2, 3], 
    index=0,
    help="p càng lớn thì độ chính xác càng cao, nhưng mạch lượng tử càng sâu và dễ gặp lỗi (noise) trên phần cứng thật."
)

if st.button("🚀 Chạy Mô Phỏng Lượng Tử QAOA", type="primary"):
    with st.spinner("Đang xây dựng Hamiltonian và chạy mô phỏng QAOA... (Có thể mất vài giây)"):
        prices_df = get_stock_data(selected_tickers, start_date, end_date)
        
        if prices_df.empty:
            st.error("Không thể tải dữ liệu.")
            st.stop()
            
        daily_returns, mean_returns, cov_matrix = calculate_returns(prices_df)
        
        # Run QAOA
        qaoa_results, raw_probs = solve_qaoa(
            mean_returns, cov_matrix, risk_aversion, budget=budget, p=p_layers
        )
        
        st.success("Mô phỏng hoàn tất!")
        
        # Get the top state (most probable)
        best_state = qaoa_results[0]
        
        # Determine Sharpe ratio
        ret = best_state['return']
        risk = best_state['risk']
        sharpe_ratio = (ret - 0.02) / risk if risk > 0 else 0
        
        # Display KPIs
        st.subheader("📊 Kết Quả Tối Ưu Nhất")
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
                <div class='metric-value'>{(ret * 100):.2f}%</div>
                <div class='metric-desc'>Dựa trên trọng số đều (Equal Weight)</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Độ Biến Động (Rủi Ro)</div>
                <div class='metric-value'>{(risk * 100):.2f}%</div>
                <div class='metric-desc'>Độ lệch chuẩn hàng năm</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Danh Mục Chọn (Bitstring)")
            
            # Format the selected stocks
            selected_indices = [i for i, bit in enumerate(best_state['binary_array']) if bit == 1]
            selected_stock_names = [selected_tickers[i] for i in selected_indices]
            
            st.info(f"**Chuỗi bit tối ưu:** `{best_state['state_str']}`\n\n**Xác suất đo được:** `{best_state['prob']*100:.2f}%`")
            
            if len(selected_stock_names) > 0:
                st.markdown("**Các cổ phiếu được chọn:**")
                for stock in selected_stock_names:
                    st.markdown(f"- ✅ **{stock}**")
                
                if len(selected_stock_names) != budget:
                    st.warning(f"⚠️ Chú ý: Số lượng chọn ({len(selected_stock_names)}) khác với Budget yêu cầu ({budget}).")
            else:
                st.warning("QAOA không chọn cổ phiếu nào cho trạng thái này (Chuỗi bit toàn 0).")
                
        with c2:
            st.subheader("Phân Phối Xác Suất QAOA")
            
            # Prepare data for bar chart
            plot_df = pd.DataFrame([
                {
                    "Trạng thái": res['state_str'],
                    "Xác suất (%)": res['prob'] * 100,
                    "Số CP": res['num_selected']
                } 
                for res in qaoa_results
            ])
            
            # Highlight states that match the budget constraint
            plot_df['Thỏa mãn Budget'] = plot_df['Số CP'] == budget
            
            fig = px.bar(
                plot_df, 
                x="Trạng thái", 
                y="Xác suất (%)",
                color="Thỏa mãn Budget",
                color_discrete_map={True: "#db2777", False: "#94a3b8"},
                title=f"Top {len(plot_df)} trạng thái có xác suất cao nhất",
                labels={"Trạng thái": "Chuỗi bit", "Xác suất (%)": "Xác suất (%)"}
            )
            fig.update_layout(xaxis_type='category', template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        with st.expander("Toán Học Đằng Sau QAOA"):
            st.markdown(r"""
            Thuật toán QAOA sử dụng hai Hamiltonian: Cost Hamiltonian ($H_C$) và Mixer Hamiltonian ($H_B$).
            
            Bài toán Markowitz được chuyển thành **QUBO** (Quadratic Unconstrained Binary Optimization) bằng cách thêm hình phạt cho ràng buộc số lượng tài sản:
            $$ C(x) = \frac{\gamma}{2} x^T \Sigma x - \mu^T x + \lambda \left(\sum x_i - B\right)^2 $$
            
            Sau đó QUBO được chuyển thành **Ising Hamiltonian** bằng biến đổi $x_i = \frac{1 - Z_i}{2}$, tạo ra $H_C$.
            
            Trong mạch lượng tử, xen kẽ 2 toán tử tiến hóa thời gian:
            - **Cost Layer:** $U_C(\gamma) = e^{-i \gamma H_C}$ (Mã hóa bài toán, sử dụng cổng $R_Z$ và $R_{ZZ}$)
            - **Mixer Layer:** $U_B(\beta) = e^{-i \beta H_B}$ (Khám phá không gian trạng thái, sử dụng cổng $R_X$)
            """)
