

import argparse
import json
import time
import warnings
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  Qiskit imports
# ─────────────────────────────────────────────
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

# IBM Runtime (hardware) — optional
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as RuntimeSampler
    RUNTIME_AVAILABLE = True
except ImportError:
    RUNTIME_AVAILABLE = False


# ══════════════════════════════════════════════
#  PHẦN 1: XÂY DỰNG BÀI TOÁN PORTFOLIO → QUBO
# ══════════════════════════════════════════════

def build_portfolio_qubo(mu: np.ndarray, sigma: np.ndarray,
                          alpha: float = 0.5,
                          beta:  float = 1.0,
                          gamma: float = 2.0) -> np.ndarray:
    """
    Chuyển bài toán Markowitz sang dạng QUBO.

    minimize  x^T Q x  =  alpha*(risk) - beta*(return) + gamma*(budget_penalty)

    Tham số:
        mu    : vector lợi nhuận kỳ vọng (n,)
        sigma : ma trận hiệp phương sai (n, n)
        alpha : trọng số rủi ro
        beta  : trọng số lợi nhuận
        gamma : hệ số phạt ràng buộc ngân sách (sum(x)=1)

    Trả về:
        Q : ma trận QUBO (n, n)
    """
    n = len(mu)
    Q = alpha * sigma - beta * np.diag(mu)

    # Ràng buộc ngân sách: gamma * (sum(x_i) - 1)^2
    # = gamma * [sum_i x_i^2 + 2*sum_{i<j} x_i*x_j - 2*sum_i x_i + 1]
    # Đóng góp vào Q:
    for i in range(n):
        Q[i, i] += gamma * (1 - 2)   # hạng tuyến tính
        for j in range(n):
            if i != j:
                Q[i, j] += gamma      # hạng bậc hai

    return Q


def qubo_to_ising(Q: np.ndarray):
    """
    Chuyển ma trận QUBO Q sang Ising Hamiltonian.

    x_i = (1 - z_i) / 2  →  thay thế vào Q và rút gọn:
    H = offset + sum_i h_i * Z_i + sum_{i<j} J_ij * Z_i Z_j

    Trả về:
        linear     : dict {qubit_index: h_i}
        quadratic  : dict {(i,j): J_ij}  (i < j)
        offset     : hằng số năng lượng
    """
    n = Q.shape[0]
    linear, quadratic = {}, {}
    offset = 0.0

    for i in range(n):
        for j in range(n):
            if i == j:
                offset    += Q[i, i] / 4
                linear[i]  = linear.get(i, 0.0) - Q[i, i] / 2
            elif i < j:
                qij          = Q[i, j] + Q[j, i]
                offset      += qij / 4
                linear[i]    = linear.get(i, 0.0) - qij / 4
                linear[j]    = linear.get(j, 0.0) - qij / 4
                quadratic[(i, j)] = quadratic.get((i, j), 0.0) + qij / 4

    return linear, quadratic, offset


def build_pauli_hamiltonian(n: int, linear: dict, quadratic: dict) -> SparsePauliOp:
    """
    Xây dựng Pauli Hamiltonian từ các hệ số Ising.

    Quy ước Qiskit: qubit 0 là LSB (rightmost character).
    """
    pauli_list = []

    for i, coeff in linear.items():
        paulis = ['I'] * n
        paulis[i] = 'Z'
        pauli_list.append((''.join(reversed(paulis)), coeff))

    for (i, j), coeff in quadratic.items():
        paulis = ['I'] * n
        paulis[i] = 'Z'
        paulis[j] = 'Z'
        pauli_list.append((''.join(reversed(paulis)), coeff))

    return SparsePauliOp.from_list(pauli_list)


# ══════════════════════════════════════════════
#  PHẦN 2: XÂY MẠCH QAOA
# ══════════════════════════════════════════════

def build_qaoa_circuit(n: int, p: int,
                        Q: np.ndarray,
                        gamma: np.ndarray,
                        beta:  np.ndarray) -> QuantumCircuit:
    """
    Xây mạch QAOA p-layer cho bài toán QUBO.

    Cấu trúc mạch:
      |+>^n  →  [Cost(γ_1) → Mixer(β_1)]^p  →  Measure

    Cost layer:
      - Cổng RZZ(2γ Q_ij) cho mỗi cặp (i,j) có Q_ij ≠ 0
      - Cổng RZ(2γ Q_ii/2) cho mỗi qubit i

    Mixer layer:
      - Cổng RX(2β) cho mỗi qubit (standard transverse-field mixer)

    Tham số:
        n     : số qubits (= số cổ phiếu)
        p     : số layers QAOA
        Q     : ma trận QUBO (n, n)
        gamma : tham số cost (p,)
        beta  : tham số mixer (p,)

    Trả về:
        qc : QuantumCircuit đã có lệnh đo
    """
    qc = QuantumCircuit(n)

    # Bước khởi tạo: trạng thái chồng chất đều
    qc.h(range(n))

    for layer in range(p):
        # ── Cost layer ──────────────────────────
        # ZZ interactions (off-diagonal)
        for i in range(n):
            for j in range(i + 1, n):
                qij = Q[i, j] + Q[j, i]
                if abs(qij) > 1e-10:
                    angle = 2 * gamma[layer] * qij / 4
                    qc.cx(i, j)
                    qc.rz(angle, j)
                    qc.cx(i, j)

        # Single Z rotations (diagonal)
        for i in range(n):
            if abs(Q[i, i]) > 1e-10:
                qc.rz(2 * gamma[layer] * Q[i, i] / 2, i)

        # ── Mixer layer ─────────────────────────
        for i in range(n):
            qc.rx(2 * beta[layer], i)

    qc.measure_all()
    return qc


def circuit_depth_stats(qc: QuantumCircuit) -> dict:
    """Trả về thống kê độ sâu mạch."""
    ops = qc.count_ops()
    return {
        "depth": qc.depth(),
        "n_qubits": qc.num_qubits,
        "gate_counts": dict(ops),
        "total_gates": sum(ops.values()),
        "cx_count": ops.get("cx", 0),
    }


# ══════════════════════════════════════════════
#  PHẦN 3: CHẠY QAOA 
# ══════════════════════════════════════════════

def evaluate_circuit(params: np.ndarray, n: int, p: int,
                      Q: np.ndarray, backend, shots: int) -> float:
    """
    Hàm mục tiêu cho COBYLA optimizer.
    Chạy mạch và tính giá trị kỳ vọng <E> = Σ P(x) * x^T Q x
    """
    gamma = params[:p]
    beta  = params[p:]

    qc = build_qaoa_circuit(n, p, Q, gamma, beta)
    t_qc = transpile(qc, backend, optimization_level=1)
    job = backend.run(t_qc, shots=shots)
    counts = job.result().get_counts()

    total_energy = 0.0
    total_shots = sum(counts.values())
    for bitstring, count in counts.items():
        x = np.array([int(b) for b in reversed(bitstring[:n])])
        energy = float(x @ Q @ x)
        total_energy += energy * count / total_shots

    return total_energy


def run_qaoa_simulator(Q: np.ndarray, p: int = 1,
                        shots: int = 4096,
                        max_iter: int = 150,
                        seed: int = 42) -> dict:
    """
    Chạy QAOA trên Qiskit AerSimulator với COBYLA optimizer.

    Trả về dict kết quả đầy đủ bao gồm:
    - Tham số tối ưu γ*, β*
    - Phân phối xác suất các nghiệm
    - Lịch sử hội tụ
    - Thống kê mạch
    """
    np.random.seed(seed)
    n = Q.shape[0]
    backend = AerSimulator(method="statevector", seed_simulator=seed)

    # Khởi tạo tham số ngẫu nhiên
    params_init = np.concatenate([
        np.random.uniform(0, np.pi, p),      # gamma
        np.random.uniform(0, np.pi / 2, p),  # beta
    ])

    convergence_history = []

    def objective_with_log(params):
        val = evaluate_circuit(params, n, p, Q, backend, shots)
        convergence_history.append(val)
        return val

    t_start = time.time()
    result = minimize(
        objective_with_log,
        params_init,
        method="COBYLA",
        options={"maxiter": max_iter, "rhobeg": 0.5, "disp": False},
    )
    t_end = time.time()

    # Chạy mạch cuối cùng với nhiều shots hơn để lấy phân phối
    gamma_opt = result.x[:p]
    beta_opt  = result.x[p:]
    qc_final  = build_qaoa_circuit(n, p, Q, gamma_opt, beta_opt)
    t_qc      = transpile(qc_final, backend, optimization_level=1)
    job_final = backend.run(t_qc, shots=8192)
    counts_final = job_final.result().get_counts()

    # Tìm nghiệm tốt nhất (QUBO energy thấp nhất)
    best_energy = np.inf
    best_x      = None
    for bitstring, _ in counts_final.items():
        x = np.array([int(b) for b in reversed(bitstring[:n])])
        e = float(x @ Q @ x)
        if e < best_energy:
            best_energy = e
            best_x = x.copy()

    # Thống kê mạch
    stats = circuit_depth_stats(qc_final)

    return {
        "p": p,
        "gamma_opt": gamma_opt.tolist(),
        "beta_opt": beta_opt.tolist(),
        "final_expectation": float(result.fun),
        "best_x": best_x.tolist(),
        "best_energy": best_energy,
        "convergence": convergence_history,
        "n_evaluations": len(convergence_history),
        "time_seconds": t_end - t_start,
        "circuit_stats": stats,
        "counts": {k[:n]: v for k, v in counts_final.items()},
    }


# ══════════════════════════════════════════════
#  PHẦN 4: SUBMIT LÊN IBM QUANTUM HARDWARE
# ══════════════════════════════════════════════

def submit_to_ibm_quantum(Q: np.ndarray, p: int = 1,
                           shots: int = 8192,
                           ibm_token: str = None,
                           backend_name: str = "ibm_brisbane") -> dict:
    """
    Submit job QAOA lên IBM Quantum hardware thật.

    Quy trình:
      1. Kết nối IBM Quantum với API token
      2. Chọn backend (ibm_brisbane: 127 qubits, Eagle r3)
      3. Transpile mạch với optimization_level=3
      4. Submit job và chờ kết quả
      5. Thu thập counts từ hardware

    Lưu ý:
      - Queue time có thể từ vài phút đến vài giờ
      - Lưu job_id để truy vấn kết quả sau (xem bên dưới)
      - Kết quả thật sẽ có noise do decoherence và gate error

    Sử dụng:
      results = submit_to_ibm_quantum(Q, p=1, ibm_token="YOUR_TOKEN_HERE")
    """
    if not RUNTIME_AVAILABLE:
        raise ImportError("Cần cài: pip install qiskit-ibm-runtime")
    if ibm_token is None:
        raise ValueError("Cần cung cấp IBM Quantum API token")

    n = Q.shape[0]

    # ── Kết nối IBM Quantum ──────────────────
    service = QiskitRuntimeService(channel="ibm_quantum", token=ibm_token)
    backend = service.backend(backend_name)

    print(f"Backend: {backend.name}")
    print(f"  Số qubits: {backend.num_qubits}")
    print(f"  Pending jobs: {backend.status().pending_jobs}")

    # ── Tham số tối ưu từ simulator ─────────
    # (chạy simulator trước, dùng kết quả làm warm start cho hardware)
    sim_result = run_qaoa_simulator(Q, p=p, shots=4096)
    gamma_opt  = np.array(sim_result["gamma_opt"])
    beta_opt   = np.array(sim_result["beta_opt"])

    # ── Transpile cho hardware ───────────────
    qc = build_qaoa_circuit(n, p, Q, gamma_opt, beta_opt)
    qc_transpiled = transpile(qc, backend=backend, optimization_level=3,
                               seed_transpiler=42)

    print(f"Độ sâu mạch sau transpile: {qc_transpiled.depth()}")
    print(f"Số cổng 2-qubit (CNOT): {qc_transpiled.count_ops().get('cx', 0)}")

    # ── Submit job ───────────────────────────
    sampler = RuntimeSampler(backend)
    job = sampler.run([qc_transpiled], shots=shots)
    job_id = job.job_id()
    print(f"\nJob submitted! Job ID: {job_id}")
    print("Lưu job_id để truy vấn sau: service.job(job_id).result()")

    # ── Chờ kết quả ─────────────────────────
    print("Chờ kết quả từ hardware...")
    t_start = time.time()
    result = job.result()
    t_end = time.time()

    pub_result = result[0]
    counts = pub_result.data.meas.get_counts()

    # Tìm nghiệm tốt nhất
    best_energy = np.inf
    best_x = None
    for bitstring, _ in counts.items():
        x = np.array([int(b) for b in reversed(bitstring[:n])])
        e = float(x @ Q @ x)
        if e < best_energy:
            best_energy = e
            best_x = x.copy()

    return {
        "job_id": job_id,
        "backend": backend_name,
        "p": p,
        "shots": shots,
        "gamma_opt": gamma_opt.tolist(),
        "beta_opt": beta_opt.tolist(),
        "best_x": best_x.tolist(),
        "best_energy": best_energy,
        "counts": counts,
        "time_seconds": t_end - t_start,
        "circuit_depth_transpiled": qc_transpiled.depth(),
    }


def retrieve_job_result(job_id: str, ibm_token: str) -> dict:
    """
    Truy vấn kết quả job đã submit lên IBM Quantum theo job_id.
    Hữu ích khi job chạy async và cần lấy kết quả sau.
    """
    service = QiskitRuntimeService(channel="ibm_quantum", token=ibm_token)
    job = service.job(job_id)
    status = job.status()
    print(f"Job {job_id} status: {status}")

    if status.name == "DONE":
        result = job.result()
        pub_result = result[0]
        counts = pub_result.data.meas.get_counts()
        return {"status": "DONE", "counts": counts}
    else:
        return {"status": status.name, "counts": None}


# ══════════════════════════════════════════════
#  PHẦN 5: ERROR MITIGATION
# ══════════════════════════════════════════════

def apply_zero_noise_extrapolation(Q: np.ndarray, p: int = 1,
                                    noise_scales: list = None,
                                    shots: int = 4096,
                                    seed: int = 42) -> dict:
    """
    Zero Noise Extrapolation (ZNE) — Error Mitigation.

    Ý tưởng:
      - Chạy mạch tại nhiều mức noise khác nhau (scale = 1, 2, 3)
      - Fit đường ngoại suy về mức zero noise
      - Giá trị tại scale=0 là kết quả đã được mitigate

    Kỹ thuật tăng noise:
      - Gate folding: thay mỗi cổng G bằng G·G†·G (scale=3)
      - Pulse stretching (chỉ trên hardware thật)

    Trong simulator này, noise được mô phỏng bằng cách
    thêm depolarizing noise với xác suất tỷ lệ với scale.
    """
    if noise_scales is None:
        noise_scales = [1, 2, 3]

    n = Q.shape[0]
    results_by_scale = {}

    from qiskit_aer.noise import NoiseModel, depolarizing_error

    # Tham số tối ưu từ noiseless simulator
    base_result  = run_qaoa_simulator(Q, p=p, shots=shots, seed=seed)
    gamma_opt    = np.array(base_result["gamma_opt"])
    beta_opt     = np.array(base_result["beta_opt"])
    ideal_energy = base_result["final_expectation"]

    energies_at_scale = []

    for scale in noise_scales:
        # Tạo noise model với depolarizing error
        noise_model = NoiseModel()
        p1q = min(0.002 * scale, 0.1)   # 1-qubit gate error
        p2q = min(0.010 * scale, 0.2)   # 2-qubit gate error

        error_1q = depolarizing_error(p1q, 1)
        error_2q = depolarizing_error(p2q, 2)
        noise_model.add_all_qubit_quantum_error(error_1q, ['rz', 'rx', 'h'])
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

        backend_noisy = AerSimulator(
            method="density_matrix",
            noise_model=noise_model,
            seed_simulator=seed,
        )

        qc = build_qaoa_circuit(n, p, Q, gamma_opt, beta_opt)
        t_qc = transpile(qc, backend_noisy, optimization_level=1)
        job = backend_noisy.run(t_qc, shots=shots)
        counts = job.result().get_counts()

        # Tính expectation value
        total_e = 0.0
        total_s = sum(counts.values())
        for bitstring, count in counts.items():
            x = np.array([int(b) for b in reversed(bitstring[:n])])
            total_e += float(x @ Q @ x) * count / total_s

        results_by_scale[scale] = total_e
        energies_at_scale.append(total_e)
        print(f"  ZNE scale={scale}: <E> = {total_e:.4f}")

    # Ngoại suy tuyến tính về scale=0
    coeffs = np.polyfit(noise_scales, energies_at_scale, 1)
    mitigated_energy = float(np.polyval(coeffs, 0))

    print(f"  ZNE mitigated (scale→0): <E> = {mitigated_energy:.4f}")
    print(f"  Ideal simulator:         <E> = {ideal_energy:.4f}")
    print(f"  Cải thiện: {abs(mitigated_energy - energies_at_scale[0]):.4f}")

    return {
        "noise_scales": noise_scales,
        "energies_at_scale": energies_at_scale,
        "mitigated_energy": mitigated_energy,
        "ideal_energy": ideal_energy,
        "extrapolation_coeffs": coeffs.tolist(),
        "gamma_opt": gamma_opt.tolist(),
        "beta_opt": beta_opt.tolist(),
    }


# ══════════════════════════════════════════════
#  PHẦN 6: PHÂN TÍCH KẾT QUẢ VÀ VISUALIZATION
# ══════════════════════════════════════════════

def analyze_portfolio_result(x: np.ndarray, mu: np.ndarray,
                              sigma: np.ndarray, stocks: list) -> dict:
    """
    Tính các chỉ số tài chính cho danh mục được chọn.
    Giả sử phân bổ đều (equal weight) giữa các cổ phiếu được chọn.
    """
    selected_idx = np.where(x == 1)[0]
    if len(selected_idx) == 0:
        return {"error": "Không có cổ phiếu nào được chọn"}

    selected_stocks = [stocks[i] for i in selected_idx]
    w = np.ones(len(selected_idx)) / len(selected_idx)

    ret  = float(mu[selected_idx] @ w)
    risk = float(np.sqrt(w @ sigma[np.ix_(selected_idx, selected_idx)] @ w))
    sharpe = ret / risk if risk > 0 else 0.0

    return {
        "selected_stocks": selected_stocks,
        "weights": dict(zip(selected_stocks, w.tolist())),
        "expected_return": ret,
        "volatility": risk,
        "sharpe_ratio": sharpe,
    }


def print_results_table(qaoa_results: dict, mu: np.ndarray,
                         sigma: np.ndarray, stocks: list):
    """In bảng so sánh kết quả QAOA các layer."""
    print("\n" + "="*65)
    print("  BẢNG KẾT QUẢ QAOA — SO SÁNH CÁC LAYER")
    print("="*65)
    print(f"{'p':>4} | {'Danh mục':>15} | {'Return':>8} | {'Risk':>8} | "
          f"{'Sharpe':>8} | {'QUBO E':>9} | {'Time(s)':>7}")
    print("-"*65)

    for p, res in sorted(qaoa_results.items()):
        x = np.array(res["best_x"])
        m = analyze_portfolio_result(x, mu, sigma, stocks)
        print(f"{p:>4} | {','.join(m['selected_stocks']):>15} | "
              f"{m['expected_return']:>7.2%} | {m['volatility']:>7.2%} | "
              f"{m['sharpe_ratio']:>8.3f} | {res['best_energy']:>9.4f} | "
              f"{res['time_seconds']:>7.2f}")
    print("="*65)


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="QAOA Portfolio Optimization — Thành viên 2"
    )
    parser.add_argument("--mode", choices=["simulator", "hardware"],
                        default="simulator", help="Chế độ chạy")
    parser.add_argument("--p", type=int, default=3,
                        help="Số QAOA layers (default: 3)")
    parser.add_argument("--shots", type=int, default=4096,
                        help="Số shots (default: 4096)")
    parser.add_argument("--ibm-token", type=str, default=None,
                        help="IBM Quantum API token (chỉ cần khi --mode hardware)")
    parser.add_argument("--zne", action="store_true",
                        help="Áp dụng Zero Noise Extrapolation")
    args = parser.parse_args()

    # ── Dữ liệu cổ phiếu (từ Thành viên 3 / yfinance) ──────────
    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    mu = np.array([0.18, 0.22, 0.15, 0.20, 0.30])
    vols = np.array([0.25, 0.22, 0.20, 0.28, 0.45])
    corr = np.array([
        [1.00, 0.75, 0.70, 0.65, 0.45],
        [0.75, 1.00, 0.72, 0.68, 0.42],
        [0.70, 0.72, 1.00, 0.60, 0.38],
        [0.65, 0.68, 0.60, 1.00, 0.40],
        [0.45, 0.42, 0.38, 0.40, 1.00],
    ])
    sigma = np.outer(vols, vols) * corr
    n = len(stocks)

    # ── Xây QUBO ────────────────────────────────────────────────
    print("\n[1/4] Xây dựng ma trận QUBO...")
    Q = build_portfolio_qubo(mu, sigma, alpha=0.5, beta=1.0, gamma=2.0)
    print(f"  QUBO matrix shape: {Q.shape}")
    print(f"  Cổ phiếu: {stocks}")

    # ── Chạy QAOA ───────────────────────────────────────────────
    qaoa_results = {}

    if args.mode == "simulator":
        print(f"\n[2/4] Chạy QAOA (p=1→{args.p}) trên AerSimulator...")
        for p in range(1, args.p + 1):
            print(f"\n  Đang chạy QAOA p={p}...")
            res = run_qaoa_simulator(Q, p=p, shots=args.shots)
            qaoa_results[p] = res
            selected = [stocks[i] for i in range(n) if res["best_x"][i] == 1]
            print(f"  → Best: {selected} | Energy: {res['best_energy']:.4f} | "
                  f"Time: {res['time_seconds']:.2f}s | "
                  f"Evals: {res['n_evaluations']}")

    else:  # hardware
        if args.ibm_token is None:
            raise ValueError("Cần --ibm-token khi dùng --mode hardware")
        print(f"\n[2/4] Submit QAOA (p={args.p}) lên IBM Quantum...")
        hw_result = submit_to_ibm_quantum(Q, p=args.p, shots=args.shots,
                                           ibm_token=args.ibm_token)
        qaoa_results[args.p] = hw_result
        with open("hardware_job_result.json", "w") as f:
            json.dump(hw_result, f, indent=2)
        print(f"  Job ID: {hw_result['job_id']}")
        print(f"  Kết quả đã lưu vào hardware_job_result.json")

    # ── Error Mitigation ────────────────────────────────────────
    if args.zne:
        print(f"\n[3/4] Áp dụng Zero Noise Extrapolation (ZNE)...")
        zne_result = apply_zero_noise_extrapolation(Q, p=min(args.p, 2),
                                                     noise_scales=[1, 2, 3],
                                                     shots=args.shots)
        with open("zne_result.json", "w") as f:
            json.dump(zne_result, f, indent=2)
        print("  ZNE kết quả đã lưu vào zne_result.json")
    else:
        print("\n[3/4] ZNE skipped. Dùng --zne để áp dụng error mitigation.")

    # ── In kết quả ──────────────────────────────────────────────
    print("\n[4/4] Tổng hợp kết quả...")
    print_results_table(qaoa_results, mu, sigma, stocks)

    # Lưu kết quả
    output = {
        "stocks": stocks,
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "Q": Q.tolist(),
        "qaoa_results": {
            str(p): {k: v for k, v in r.items() if k != "counts"}
            for p, r in qaoa_results.items()
        },
    }
    with open("qaoa_portfolio_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n  Kết quả đã lưu vào qaoa_portfolio_results.json")


if __name__ == "__main__":
    main()
