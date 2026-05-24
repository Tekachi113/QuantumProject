"""
app/components/sidebar.py
--------------------------
Shared sidebar component for all pages of the Quantum Portfolio multi-page app.
Provides: ticker selection, date range, risk aversion slider, and returns shared state.
"""

import datetime
import streamlit as st

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "JPM", "V", "DIS", "NFLX"]
EXTENDED_TICKERS = DEFAULT_TICKERS + ["GOOG", "META", "MS", "GS", "WMT", "KO", "PEP", "PG", "COST", "AMD",
                                       "JNJ", "UNH", "COST", "HD", "MCD"]

def render_sidebar():
    """Renders the shared sidebar and returns (selected_tickers, start_date, end_date, risk_aversion)."""
    st.sidebar.markdown("<div style='margin-bottom: 0px;'><span style='font-size: 50px;'>⚛️</span></div>", unsafe_allow_html=True)
    st.sidebar.markdown("<h2 style='font-weight:800; margin-top:0;'>Quantum Portfolio</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("---")

    # 1. Stock selection
    st.sidebar.subheader("📌 1. Chọn Cổ Phiếu")
    selected_tickers = st.sidebar.multiselect(
        "Danh sách cổ phiếu đầu tư:",
        options=EXTENDED_TICKERS,
        default=["AAPL", "MSFT", "NVDA", "JPM", "AMZN"]
    )

    max_assets = 8
    if len(selected_tickers) > max_assets:
        st.sidebar.warning(
            f"⚠️ Nhằm đảm bảo mô phỏng lượng tử QAOA chạy mượt mà, "
            f"vui lòng chọn tối đa {max_assets} cổ phiếu (Hiện tại: {len(selected_tickers)})."
        )

    # 2. Date range
    st.sidebar.subheader("📅 2. Chọn Thời Gian")
    today = datetime.date.today()
    five_years_ago = today - datetime.timedelta(days=5 * 365)
    start_date = st.sidebar.date_input("Từ ngày:", five_years_ago)
    end_date = st.sidebar.date_input("Đến ngày:", today)

    if start_date >= end_date:
        st.sidebar.error("Lỗi: Ngày bắt đầu phải nhỏ hơn ngày kết thúc!")

    # 3. Risk aversion
    st.sidebar.subheader("⚡ 3. Chấp Nhận Rủi Ro")
    risk_aversion = st.sidebar.slider(
        "Hệ số e ngại rủi ro (γ):",
        min_value=0.1, max_value=10.0, value=2.0, step=0.1,
        help="γ càng cao → ưu tiên rủi ro thấp. γ thấp → ưu tiên lợi nhuận cao."
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **Hệ thống sử dụng**:\n"
        "- **yfinance** để tải dữ liệu lịch sử.\n"
        "- **CVXPY** cho tối ưu hóa cổ điển.\n"
        "- **Qiskit Statevector** để mô phỏng mạch QAOA.\n"
        "- **IBM Quantum** cho kết quả phần cứng thật."
    )

    return selected_tickers, start_date, end_date, risk_aversion
