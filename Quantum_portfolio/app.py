import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quantum Portfolio Optimizer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global premium CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stMarkdown {
        font-family: 'Outfit', sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #4338ca 40%, #6d28d9 75%, #db2777 100%);
        padding: 2.5rem 3rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.12);
        text-align: center;
    }
    .main-header h1 { font-weight: 800; font-size: 3.2rem !important; margin: 0; letter-spacing: -1.5px; text-shadow: 0 4px 14px rgba(0,0,0,0.35); }
    .main-header p  { font-weight: 300; font-size: 1.25rem !important; opacity: 0.9; margin-top: 0.75rem; letter-spacing: 0.3px; }
    .quantum-badge  { background: linear-gradient(90deg, #6d28d9 0%, #db2777 100%); color: white; padding: 0.3rem 0.8rem; border-radius: 9999px; font-size: 0.82rem; font-weight: 600; display: inline-block; margin-bottom: 0.75rem; }
    .nav-card {
        background: rgba(99, 102, 241, 0.05);
        border: 1px solid rgba(99, 102, 241, 0.18);
        border-radius: 16px;
        padding: 1.8rem 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
        cursor: pointer;
    }
    .nav-card:hover {
        transform: translateY(-6px);
        background: rgba(99, 102, 241, 0.1);
        border-color: rgba(99, 102, 241, 0.55);
        box-shadow: 0 12px 28px rgba(99, 102, 241, 0.22);
    }
    .nav-icon   { font-size: 2.8rem; margin-bottom: 0.5rem; }
    .nav-title  { font-size: 1.15rem; font-weight: 700; color: #6366f1; margin-bottom: 0.4rem; }
    .nav-desc   { font-size: 0.88rem; color: #94a3b8; line-height: 1.5; }
    .stat-row   { display: flex; gap: 1rem; margin: 1.5rem 0; }
    .stat-item  { flex: 1; background: rgba(99,102,241,0.06); border-radius: 12px; padding: 1.2rem; text-align: center; border: 1px solid rgba(99,102,241,0.12); }
    .stat-val   { font-size: 1.9rem; font-weight: 800; color: #7c3aed; }
    .stat-lbl   { font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.3rem; }
    .custom-card { background: rgba(99, 102, 241, 0.05); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border: 1px solid rgba(99, 102, 241, 0.15); border-radius: 16px; padding: 1.5rem; text-align: center; transition: all 0.3s ease; margin-bottom: 1rem; }
    .custom-card:hover { transform: translateY(-5px); background: rgba(99, 102, 241, 0.08); border-color: rgba(99, 102, 241, 0.5); box-shadow: 0 10px 20px rgba(99, 102, 241, 0.2); }
    .card-title { font-size: 0.85rem !important; text-transform: uppercase; letter-spacing: 1.5px; color: #4f46e5; margin-bottom: 0.5rem; font-weight: 600; }
    .card-value { font-size: 2.2rem !important; font-weight: 800; color: #7c3aed; margin: 0; }
    .card-desc  { font-size: 0.8rem !important; color: #475569; margin-top: 0.5rem; }
    .math-info-box { background: rgba(99, 102, 241, 0.07); border-left: 4px solid #6366f1; padding: 1.2rem; border-radius: 8px; margin: 1.5rem 0; color: #cbd5e1; }
    .table-container { border-radius: 12px; overflow: hidden; border: 1px solid rgba(255, 255, 255, 0.1); }
</style>
""", unsafe_allow_html=True)


# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="quantum-badge">Quantum Computing × Quantitative Finance</div>
    <h1>⚛️ QUANTUM PORTFOLIO</h1>
    <p>Tối ưu hóa danh mục đầu tư S&P500 bằng thuật toán Cổ điển (MVO) & Lượng tử (QAOA)</p>
    <p style="font-size:0.9rem; opacity:0.7; margin-top:0.4rem;">Nhóm 3 thành viên · IBM Quantum · Qiskit · CVXPY · Streamlit</p>
</div>
""", unsafe_allow_html=True)


# ── Quick stats row ───────────────────────────────────────────────────────────
st.markdown("""
<div class="stat-row">
    <div class="stat-item"><div class="stat-val">10</div><div class="stat-lbl">Cổ phiếu S&amp;P500</div></div>
    <div class="stat-item"><div class="stat-val">3y</div><div class="stat-lbl">Dữ liệu lịch sử</div></div>
    <div class="stat-item"><div class="stat-val">QAOA</div><div class="stat-lbl">Thuật toán lượng tử</div></div>
    <div class="stat-item"><div class="stat-val">127</div><div class="stat-lbl">Qubits (ibm_brisbane)</div></div>
    <div class="stat-item"><div class="stat-val">0.914</div><div class="stat-lbl">Approximation Ratio</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Navigation cards ──────────────────────────────────────────────────────────
st.markdown("### 🧭 Điều hướng — Chọn trang để khám phá")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-icon">📊</div>
        <div class="nav-title">Khám Phá Dữ Liệu</div>
        <div class="nav-desc">Thống kê, ma trận tương quan, tăng trưởng tích lũy của 10 cổ phiếu S&amp;P500</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("👉 Khám Phá Dữ Liệu", use_container_width=True, key="btn_data"):
        st.switch_page("pages/1_📊_Du_Lieu.py")

with col2:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-icon">📈</div>
        <div class="nav-title">Tối Ưu Cổ Điển</div>
        <div class="nav-desc">Mean-Variance Optimization (Markowitz MVO) với Efficient Frontier và phân bổ vốn tối ưu</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("👉 Tối Ưu Cổ Điển", use_container_width=True, key="btn_mvo"):
        st.switch_page("pages/2_📈_Co_Dien_MVO.py")

with col3:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-icon">⚛️</div>
        <div class="nav-title">Tối Ưu Lượng Tử</div>
        <div class="nav-desc">QAOA (Quantum Approximate Optimization Algorithm) — mô phỏng Qiskit + kết quả IBM Quantum</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("👉 Tối Ưu Lượng Tử", use_container_width=True, key="btn_qaoa"):
        st.switch_page("pages/3_⚛️_Luong_Tu_QAOA.py")

with col4:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-icon">🏆</div>
        <div class="nav-title">So Sánh Tổng Hợp</div>
        <div class="nav-desc">QAOA p=1,2,3 vs Cổ điển MVO vs IBM Hardware — phân tích ZNE và noise</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("👉 So Sánh Tổng Hợp", use_container_width=True, key="btn_compare"):
        st.switch_page("pages/4_🏆_So_Sanh.py")

st.markdown("<br>", unsafe_allow_html=True)
st.info("👈 **Sử dụng menu bên trái** để điều hướng giữa các trang phân tích.")

st.markdown("---")

# ── Project overview ──────────────────────────────────────────────────────────
st.markdown("## 📋 Giới thiệu Dự án")

c1, c2 = st.columns([3, 2])
with c1:
    st.markdown("""
    Bài toán **tối ưu hóa danh mục đầu tư** — chọn tỷ trọng phân bổ vốn vào các cổ phiếu sao cho
    lợi nhuận kỳ vọng cao nhất với rủi ro thấp nhất — là một trong những bài toán cốt lõi của
    **tài chính định lượng hiện đại**.

    Kể từ khi Harry Markowitz giới thiệu lý thuyết **Mean-Variance Optimization (MVO)** vào năm 1952,
    bài toán này được giải bằng lập trình bậc hai cổ điển. Tuy nhiên, khi số lượng cổ phiếu tăng,
    không gian tổ hợp bùng nổ theo hàm mũ — **2ⁿ** với n cổ phiếu — khiến các solver cổ điển
    gặp khó khăn với ràng buộc nhị phân.

    **Thuật toán QAOA** (Quantum Approximate Optimization Algorithm) tận dụng superposition và
    entanglement của máy tính lượng tử để giải bài toán dưới dạng **QUBO** (Quadratic Unconstrained
    Binary Optimization), mở ra tiềm năng lợi thế tính toán đáng kể.
    """)

with c2:
    st.markdown("#### 🛠️ Stack Kỹ Thuật")
    st.markdown("""
    | Layer | Công cụ |
    |-------|---------|
    | Dữ liệu | `yfinance` (Yahoo Finance) |
    | Cổ điển | `CVXPY` (Markowitz MVO) |
    | Lượng tử | `Qiskit` + `AerSimulator` |
    | Hardware | `IBM Quantum` (ibm_brisbane) |
    | Trực quan | `Plotly`, `Matplotlib` |
    | Web App | `Streamlit` |
    """)

st.markdown("---")

# ── Method overview ───────────────────────────────────────────────────────────
st.markdown("## ⚙️ Pipeline Kỹ Thuật")

st.markdown("""
```
Dữ liệu S&P500 (yfinance)
        ↓
Tính μ, Σ (lợi nhuận kỳ vọng & ma trận hiệp phương sai)
        ↓
┌───────────────────────┬──────────────────────────────┐
│  Classical Solver     │     Quantum Solver           │
│  CVXPY (MVO)          │     Qiskit QAOA              │
│  - GMV Portfolio      │     Markowitz → QUBO → Ising │
│  - Max Sharpe         │     → QAOA Circuit           │
│  - Efficient Frontier │     → AerSimulator / IBM HW  │
└───────────────────────┴──────────────────────────────┘
        ↓
So sánh: Sharpe Ratio · Return · Volatility · Time
```
""")

# ── Team ─────────────────────────────────────────────────────────────────────
st.markdown("## 👥 Phân Công Nhóm")
t1, t2, t3 = st.columns(3)
with t1:
    st.markdown("""
    **Thành viên 1**
    - Thu thập & xử lý dữ liệu
    - `src/data/` pipeline
    - Báo cáo: Phần I, II, IV
    """)
with t2:
    st.markdown("""
    **Thành viên 2**
    - Lập trình lượng tử QAOA
    - IBM Quantum hardware run
    - Báo cáo: Phần III, V, VI
    """)
with t3:
    st.markdown("""
    **Thành viên 3**
    - Streamlit web application
    - Classical MVO solver
    - Báo cáo: Abstract, VII
    """)
