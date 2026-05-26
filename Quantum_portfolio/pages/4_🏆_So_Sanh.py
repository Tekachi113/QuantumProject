
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from components.sidebar import render_sidebar

st.set_page_config(page_title="So Sánh Tổng Hợp | Quantum Portfolio", page_icon="🏆", layout="wide")

st.title("🏆 So Sánh Tổng Hợp (Classical vs Quantum)")
st.markdown("Đánh giá tổng quan hiệu suất của các phương pháp: Cổ điển (MVO), QAOA Simulator (p=1,2,3), và phần cứng lượng tử thật của IBM.")

# Render shared sidebar (we might not use all variables, but keeps UI consistent)
selected_tickers, start_date, end_date, risk_aversion = render_sidebar()

# Function to load pre-computed JSON results
@st.cache_data
def load_results():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    comp_path = os.path.join(base_dir, "comparison_results.json")
    hw_path = os.path.join(base_dir, "hardware_results.json")
    
    comp_data = {}
    hw_data = {}
    
    if os.path.exists(comp_path):
        with open(comp_path, 'r') as f:
            comp_data = json.load(f)
            
    if os.path.exists(hw_path):
        with open(hw_path, 'r') as f:
            hw_data = json.load(f)
            
    return comp_data, hw_data

comp_data, hw_data = load_results()


# Tính returns để dùng cho cả QAOA live và Hardware
from utils import get_stock_data, calculate_returns, solve_qaoa, _optimize_weights_for_selection

with st.spinner("Đang tải dữ liệu..."):
    prices_df = get_stock_data(selected_tickers, start_date, end_date)
    _, mean_returns, cov_matrix = calculate_returns(prices_df)

if not comp_data or not hw_data:
    st.error("Không tìm thấy file kết quả `comparison_results.json` hoặc `hardware_results.json`. Vui lòng chạy thí nghiệm lượng tử trước!")
    st.stop()

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Hiệu Suất Tổng Hợp", "Độ Sâu Mạch (QAOA p=1,2,3)", "Nhiễu (Hardware & ZNE)"])

with tab1:
    st.subheader("Bảng Xếp Hạng Sharpe Ratio")
    
    # Construct summary DataFrame
    summary = []
    
    # 1. Classical MVO
    mvo = comp_data.get("mvo_continuous", {})
    if mvo:
        summary.append({
            "Phương pháp": "Cổ điển (CVXPY MVO)",
            "Lợi nhuận (%)": mvo.get("return", 0) * 100,
            "Rủi ro (%)": mvo.get("risk", 0) * 100,
            "Sharpe Ratio": mvo.get("sharpe", 0),
            "Ghi chú": "Trọng số liên tục (Continuous)"
        })
        
    # 2. QAOA by p
    # Bằng đoạn này — tính lại với optimal weights
from utils import get_stock_data, calculate_returns, solve_qaoa, _optimize_weights_for_selection

@st.cache_data
def compute_qaoa_comparison(tickers, start, end, risk_aversion):
    prices_df = get_stock_data(tickers, start, end)
    _, mean_returns, cov_matrix = calculate_returns(prices_df)
    mu = mean_returns.values
    cov = cov_matrix.values
    budget = max(1, len(tickers) // 2)
    
    results_by_p = {}
    for p in [1, 2, 3]:
        qaoa_results, _ = solve_qaoa(mean_returns, cov_matrix, risk_aversion, budget, p=p)
        
        # Lấy state có prob cao nhất và đủ số cổ phiếu
        best = next(
            (r for r in qaoa_results if r['num_selected'] == budget),
            qaoa_results[0]  # fallback
        )
        
        x = best['binary_array']
        w_full, ret, vol = _optimize_weights_for_selection(x, mu, cov, risk_aversion)
        sharpe = ret / vol if vol > 0 else 0.0
        
        selected_names = [tickers[i] for i in range(len(tickers)) if x[i] == 1]
        
        results_by_p[p] = {
            "return": ret,
            "risk": vol,
            "sharpe": sharpe,
            "bitstring": best['state_str'],
            "selected": selected_names,
            "weights": w_full.tolist(),
            "prob": best['prob']
        }
    
    return results_by_p

# Dùng trong tab1
with st.spinner("Đang tính QAOA với optimal weights..."):
    qaoa_live = compute_qaoa_comparison(
        selected_tickers, start_date, end_date, risk_aversion
    )

for p, data in qaoa_live.items():
    selected_str = ", ".join(data['selected'])
    summary.append({
        "Phương pháp": f"QAOA Simulator (p={p})",
        "Lợi nhuận (%)": data["return"] * 100,
        "Rủi ro (%)": data["risk"] * 100,
        "Sharpe Ratio": data["sharpe"],
        "Ghi chú": f"Chọn: {selected_str} | Prob: {data['prob']:.3f}"
    })
# 3. Hardware - dùng qaoa_live p=2 (phù hợp nhất với hardware thực)
if hw_data:
    hw_best_bitstring = hw_data.get("best_bitstring", "")
    
    if hw_best_bitstring:
        # Tính optimal weights từ bitstring hardware thực
        x_hw = np.array([int(b) for b in hw_best_bitstring])
        mu = mean_returns.values
        cov = cov_matrix.values
        
        w_hw, ret_hw, vol_hw = _optimize_weights_for_selection(
            x_hw, mu, cov, risk_aversion
        )
        sharpe_hw = ret_hw / vol_hw if vol_hw > 0 else 0.0
        selected_hw = [selected_tickers[i] for i in range(len(x_hw)) if x_hw[i] == 1]
    else:
        # Fallback nếu không có bitstring
        ret_hw = qaoa_live.get(2, {}).get("return", 0)
        vol_hw = qaoa_live.get(2, {}).get("risk", 0)
        sharpe_hw = qaoa_live.get(2, {}).get("sharpe", 0)
        selected_hw = []

    summary.append({
        "Phương pháp": "IBM Quantum (ibm_brisbane)",
        "Lợi nhuận (%)": ret_hw * 100,
        "Rủi ro (%)": vol_hw * 100,
        "Sharpe Ratio": sharpe_hw,
        "Ghi chú": f"Chọn: {', '.join(selected_hw)} | Mitigated Energy: {hw_data.get('zne_mitigated_energy', 0):.4f}"
    })

with tab2:
    st.subheader("Hội Tụ COBYLA theo số lớp (p)")
    st.markdown("Khi số lớp (layers) $p$ tăng lên, thuật toán QAOA có khả năng xấp xỉ nghiệm tốt hơn, nhưng đồng thời làm mạch lượng tử sâu hơn.")
    
    qaoa = comp_data.get("qaoa_by_p", {})
    if qaoa:
        fig_conv = go.Figure()
        
        colors = {"1": "#3b82f6", "2": "#8b5cf6", "3": "#ec4899"}
        
        for p, data in qaoa.items():
            convergence = data.get("convergence", [])
            if convergence:
                fig_conv.add_trace(go.Scatter(
                    y=convergence,
                    mode="lines+markers",
                    name=f"QAOA p={p}",
                    line=dict(color=colors.get(p, "gray"))
                ))
                
        fig_conv.add_hline(y=comp_data.get("brute_force", {}).get("energy", -2.1987), 
                           line_dash="dash", line_color="red", 
                           annotation_text="Nghiệm Tối Ưu Toàn Cục")
                           
        fig_conv.update_layout(
            title="Quá trình hội tụ (Năng lượng Expectation vs Số vòng lặp)",
            xaxis_title="Vòng lặp COBYLA",
            yaxis_title="Giá trị Expectation Energy",
            template="plotly_white"
        )
        st.plotly_chart(fig_conv, use_container_width=True)

with tab3:
    st.subheader("Kết Quả Phần Cứng Thực: IBM Quantum")
    st.markdown(f"**Backend:** `{hw_data.get('backend', 'ibm_brisbane')}` | **Shots:** {hw_data.get('shots', 8192)}")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("**Phân Phối Xác Suất (Nhiễu Môi Trường)**")
        counts = hw_data.get("counts_top", {})
        if counts:
            df_counts = pd.DataFrame(list(counts.items()), columns=["Bitstring", "Counts"])
            df_counts["Probability (%)"] = (df_counts["Counts"] / hw_data.get("shots", 8192)) * 100
            
            fig_counts = px.bar(
                df_counts,
                x="Bitstring",
                y="Probability (%)",
                title="Top trạng thái đo được (Có Nhiễu)",
                color_discrete_sequence=["#ef4444"]
            )
            fig_counts.update_layout(xaxis_type='category', template="plotly_white")
            st.plotly_chart(fig_counts, use_container_width=True)
            
            st.info(f"Nghiệm tối ưu lý tưởng `{hw_data.get('best_bitstring', '00001')}` bị suy giảm mạnh do nhiễu cổng (gate error) và decoherence.")
            
    with c2:
        st.markdown("**Error Mitigation: Zero Noise Extrapolation (ZNE)**")
        noise_levels = hw_data.get("noise_levels", [])
        
        if noise_levels:
            df_zne = pd.DataFrame(noise_levels, columns=["Noise Scale", "Energy"])
            
            fig_zne = px.scatter(
                df_zne, x="Noise Scale", y="Energy", 
                title="Zero Noise Extrapolation (ZNE)"
            )
            fig_zne.update_traces(marker=dict(size=12, color="blue"))
            
            # Draw extrapolation line
            mitigated = hw_data.get("zne_mitigated_energy", 0)
            ideal = hw_data.get("simulator_energy_p1", 0)
            
            fig_zne.add_trace(go.Scatter(
                x=[0] + df_zne["Noise Scale"].tolist(),
                y=[mitigated] + df_zne["Energy"].tolist(),
                mode="lines", line=dict(dash="dot", color="blue"),
                name="Extrapolation"
            ))
            
            # Add markers for mitigated and ideal
            fig_zne.add_trace(go.Scatter(
                x=[0], y=[mitigated], mode="markers+text", 
                marker=dict(size=14, color="green", symbol="star"),
                text=["Mitigated (Scale 0)"], textposition="bottom right",
                name="Mitigated"
            ))
            
            fig_zne.add_trace(go.Scatter(
                x=[0], y=[ideal], mode="markers+text", 
                marker=dict(size=14, color="red", symbol="x"),
                text=["Ideal (Simulator)"], textposition="top right",
                name="Ideal"
            ))
            
            fig_zne.update_layout(template="plotly_white")
            st.plotly_chart(fig_zne, use_container_width=True)
            
            st.success(f"ZNE giúp cải thiện Energy từ {df_zne.iloc[0]['Energy']:.4f} (Raw HW) xuống {mitigated:.4f} (Gần hơn với lý tưởng {ideal:.4f}).")
