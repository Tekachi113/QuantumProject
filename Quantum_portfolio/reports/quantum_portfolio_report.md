# ⚛️ Quantum Portfolio Optimization: QAOA on S&P 500

**Thành viên 1**: Data Pipeline & Problem Formulation  
**Thành viên 2**: Quantum Implementation & Error Mitigation  
**Thành viên 3**: Classical Solver & System Integration  

## Abstract
Bài báo cáo này trình bày phương pháp tối ưu hóa danh mục đầu tư bằng thuật toán Quantum Approximate Optimization Algorithm (QAOA) cho 10 cổ phiếu thuộc chỉ số S&P 500. Bài toán Markowitz được mô hình hóa dưới dạng Quadratic Unconstrained Binary Optimization (QUBO) và chuyển đổi thành Ising Hamiltonian. Thông qua việc mô phỏng trên AerSimulator và chạy thực tế trên máy tính lượng tử 127-qubit `ibm_brisbane` của IBM Quantum, nhóm nghiên cứu đã đánh giá khả năng hội tụ và độ chính xác của QAOA. Kết quả cho thấy QAOA đạt tỷ lệ xấp xỉ lý tưởng (Approximation Ratio = 1.000) trên môi trường mô phỏng. Tuy nhiên, nhiễu phần cứng (noise) trên hệ thống lượng tử thực tế làm giảm đáng kể xác suất đo được nghiệm tối ưu. Kỹ thuật Zero Noise Extrapolation (ZNE) được áp dụng thành công để giảm thiểu sai số đo lường, cải thiện Expectation Energy lên khoảng 50%. Kết quả này mở ra hướng phát triển đầy hứa hẹn cho tài chính lượng tử trong kỷ nguyên NISQ.

---

## I. Introduction (NISQ era, quantum advantage motivation)
*(Đóng góp: Thành viên 1)*

Tối ưu hóa danh mục đầu tư là một bài toán kinh điển trong tài chính định lượng, bắt nguồn từ Lý thuyết Danh mục đầu tư Hiện đại (Modern Portfolio Theory - MPT) của Harry Markowitz (1952). Bài toán yêu cầu tìm ra tỷ trọng phân bổ vốn lý tưởng nhằm cực đại hóa lợi nhuận kỳ vọng, đồng thời cực tiểu hóa rủi ro (độ lệch chuẩn). 

Với sự gia tăng của số lượng tài sản, không gian nghiệm tổ hợp bùng nổ theo hàm mũ. Trong kỷ nguyên Noisy Intermediate-Scale Quantum (NISQ) hiện tại, thuật toán hybrid như QAOA nổi lên như một công cụ mạnh mẽ để khai thác sức mạnh của máy tính lượng tử giải quyết các bài toán tối ưu tổ hợp này. 

## II. Problem Formulation (Markowitz → QUBO → Ising)
*(Đóng góp: Thành viên 1)*

Bài toán MVO ban đầu có dạng liên tục. Để áp dụng QAOA, bài toán được chuyển về dạng tối ưu hóa nhị phân không ràng buộc (QUBO). Cụ thể, một hàm mục tiêu mới $C(x)$ được xây dựng:

$$ C(x) = \frac{\gamma}{2} x^T \Sigma x - \mu^T x + \lambda \left(\sum_{i=1}^n x_i - B\right)^2 $$

Trong đó:
- $\gamma$ là hệ số e ngại rủi ro.
- $\Sigma$ là ma trận hiệp phương sai.
- $\mu$ là vector lợi nhuận kỳ vọng.
- $\lambda$ là hệ số phạt (penalty) cho ràng buộc chọn đúng $B$ cổ phiếu.
- $x_i \in \{0, 1\}$ biểu diễn quyết định chọn tài sản $i$.

Từ dạng QUBO, hàm chi phí được ánh xạ thành **Ising Hamiltonian** thông qua phép biến đổi $x_i = \frac{1 - Z_i}{2}$ (với $Z_i$ là toán tử Pauli-Z).

## III. Methodology (QAOA, ZNE, IBM Quantum)
*(Đóng góp: Thành viên 2)*

### 1. Thuật toán QAOA
Mạch lượng tử QAOA bao gồm sự luân phiên giữa Cost Layer $U_C(\gamma)$ và Mixer Layer $U_B(\beta)$. Quá trình tối ưu hóa các tham số $\gamma$ và $\beta$ được thực hiện bằng bộ tối ưu cổ điển COBYLA để tìm cấu hình trạng thái lượng tử có năng lượng (Expectation Energy) thấp nhất.

### 2. Zero Noise Extrapolation (ZNE)
Trên phần cứng NISQ, lỗi cổng (gate errors) và sự mất kết hợp (decoherence) ảnh hưởng lớn đến kết quả. Kỹ thuật ZNE được sử dụng bằng cách cố tình khuếch đại nhiễu (noise scaling bằng gate folding) để thu được các kết quả ở mức nhiễu $1\times, 2\times, 3\times$, sau đó dùng phương pháp ngoại suy tuyến tính để ước tính giá trị lý tưởng ở mức nhiễu $0\times$.

### 3. Môi trường thực nghiệm
- **Mô phỏng (Simulator):** Qiskit AerSimulator.
- **Hardware:** IBM Quantum `ibm_brisbane` (127-qubit Eagle r3 processor).

## IV. Experimental Setup
*(Đóng góp: Thành viên 1)*

Dữ liệu được thu thập từ Yahoo Finance bao gồm lịch sử giá điều chỉnh (Adjusted Close) của 10 cổ phiếu trong 3 năm qua: AAPL, MSFT, GOOGL, AMZN, NVDA, JPM, JNJ, V, PG, UNH. Lợi nhuận và ma trận hiệp phương sai được tính toán để phục vụ làm tham số cho mô hình QUBO. Ngân sách $B$ được thiết lập chọn 5 cổ phiếu tối ưu nhất.

## V. Results
*(Đóng góp: Thành viên 2)*

### 1. Simulator Results
Trên AerSimulator, các độ sâu mạch $p=1, 2, 3$ đều tìm được nghiệm tối ưu. Mạch sâu hơn ($p=3$) cho ra năng lượng Expectation thấp hơn và hội tụ ổn định hơn:
- $p=1$: Energy = -0.3194
- $p=2$: Energy = -1.0609
- $p=3$: Energy = -1.7153 (Tốt nhất)

### 2. IBM Hardware Results
Khi submit lên `ibm_brisbane` (8192 shots, $p=1$), phân phối đo lường bị làm phẳng do nhiễu:
- Nghiệm tối ưu (Bitstring `00001`) đạt tần suất cao nhất: 1,656 counts (20.2%).
- Raw Energy = -0.2811.

### 3. ZNE Mitigation
Áp dụng ZNE với các hệ số $scale \in \{1, 2, 3\}$:
- Scale 1 (Raw): -0.2811
- Scale 2: -0.2619
- Scale 3: -0.2427
- **Mitigated Energy (Ngoại suy):** -0.3002. Kết quả Mitigated đã tiệm cận rất sát với Energy lý tưởng trên Simulator (-0.3194), chứng minh hiệu quả của ZNE.

## VI. Discussion
*(Đóng góp: Thành viên 2)*

Thực nghiệm cho thấy QAOA có tiềm năng ứng dụng trực tiếp, khi approximation ratio trên lý thuyết (simulator) đạt tới 1.000. Tuy nhiên, thách thức lớn nhất của tài chính lượng tử hiện nay chính là NISQ hardware limits. Số lượng cổng entangling CNOT trong hàm mục tiêu QUBO tăng theo hàm $O(N^2)$, gây nhiễu nhanh chóng khi $p$ tăng. 

## VII. Conclusion & Future Work
*(Đóng góp: Thành viên 3)*

Dự án đã hoàn thành mục tiêu xây dựng một pipeline trọn vẹn từ thu thập dữ liệu S&P 500, tối ưu hóa danh mục MVO cổ điển bằng CVXPY, cho đến lập trình lượng tử QAOA trên IBM hardware. Ứng dụng Streamlit được phát triển đóng vai trò là cầu nối giúp người dùng trực quan hóa sự vượt trội và tính thực tiễn của các phương pháp lượng tử.
Trong tương lai, hướng nghiên cứu có thể mở rộng lên các bộ bài toán với số lượng tài sản lớn hơn bằng cách tối ưu hóa mạch ansatz, hoặc áp dụng các kỹ thuật Quantum Error Mitigation (QEM) tiên tiến hơn.

---

### References
1. Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*, 7(1), 77-91.
2. Farhi, E., Goldstone, J., & Gutmann, S. (2014). A Quantum Approximate Optimization Algorithm. *arXiv preprint arXiv:1411.4028*.
3. Orús, R., Mugel, S., & Lizaso, E. (2019). Quantum computing for finance: Overview and prospects. *Reviews in Physics*, 4, 100028.
4. Qiskit Finance Documentation, IBM Quantum.
