"""
src/quantum/optimizer.py
------------------------
Vòng lặp tối ưu hóa hybrid classical-quantum cho QAOA.
Chạy hoàn toàn local trên AerSimulator — không cần IBM account.

Luồng:
  1. Khởi tạo tham số θ = [γ₁, β₁, ..., γₚ, βₚ]
  2. Bind θ vào mạch QAOA → chạy trên AerSimulator
  3. Đo kết quả → phân phối bitstring → tính ⟨H_C⟩
  4. Classical optimizer (COBYLA/SPSA) cập nhật θ
  5. Lặp đến khi hội tụ
  6. Lấy bitstring tốt nhất → decode thành portfolio weights

Classical optimizers:
  - COBYLA : gradient-free, ổn định, khuyến nghị cho shot-based
  - SPSA   : stochastic gradient, tốt hơn khi có nhiều tham số (p ≥ 4)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Dataclass kết quả
# ══════════════════════════════════════════════════════════════════

@dataclass
class OptimizationResult:
    """
    Kết quả từ vòng lặp QAOA.

    Attributes
    ----------
    best_bitstring : np.ndarray
        Bitstring tốt nhất {0,1}^n.
    best_objective : float
        Giá trị QUBO tại bitstring tốt nhất.
    optimal_params : np.ndarray
        Tham số θ* = [γ*, β*] tối ưu.
    weights : pd.Series
        Trọng số danh mục decode từ bitstring.
    n_iterations : int
        Số lần evaluate hàm mục tiêu.
    convergence_history : list[float]
        Lịch sử ⟨H_C⟩ qua các vòng lặp.
    feasible : bool
        True nếu bitstring thỏa ràng buộc budget.
    elapsed_seconds : float
        Thời gian chạy.
    simulator_mode : str
        Mode simulator đã dùng: 'statevector'/'ideal'/'noisy'.
    shots : int
        Số lần đo mỗi circuit (0 nếu statevector).
    """

    best_bitstring: np.ndarray
    best_objective: float
    optimal_params: np.ndarray
    weights: pd.Series
    n_iterations: int
    convergence_history: list = field(default_factory=list)
    feasible: bool = False
    elapsed_seconds: float = 0.0
    simulator_mode: str = "ideal"
    shots: int = 1024

    def __str__(self) -> str:
        lines = [
            "=" * 52,
            "QAOA OPTIMIZATION RESULT",
            "=" * 52,
            f"Simulator      : {self.simulator_mode}",
            f"Shots          : {self.shots if self.simulator_mode != 'statevector' else 'N/A (exact)'}",
            f"Iterations     : {self.n_iterations}",
            f"Elapsed        : {self.elapsed_seconds:.1f}s",
            f"Best objective : {self.best_objective:.6f}",
            f"Feasible       : {'✓' if self.feasible else '✗'}",
            f"Bitstring      : {''.join(map(str, self.best_bitstring.astype(int)))}",
            "",
            "Portfolio weights:",
        ]
        for ticker, w in self.weights[self.weights > 1e-6].sort_values(ascending=False).items():
            bar = "█" * int(w * 30)
            lines.append(f"  {ticker:<8} {w:.2%}  {bar}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# Helpers: expectation và sampling
# ══════════════════════════════════════════════════════════════════

def counts_to_expectation(counts: dict, Q: np.ndarray, shots: int) -> float:
    """
    Tính ⟨H_C⟩ = Σ_x P(x) · xᵀQx từ kết quả đo.

    Parameters
    ----------
    counts : dict
        {bitstring: count} từ Qiskit result.get_counts().
        Qiskit convention: qubit 0 ở cuối bitstring → cần reverse.
    Q : np.ndarray
        Ma trận QUBO.
    shots : int
        Tổng số lần đo.

    Returns
    -------
    float
        Expectation value.
    """
    expectation = 0.0
    for bitstring, count in counts.items():
        x = np.array([int(b) for b in reversed(bitstring)], dtype=float)
        expectation += (count / shots) * float(x @ Q @ x)
    return expectation


def statevector_to_expectation(statevector: np.ndarray, Q: np.ndarray) -> float:
    """
    Tính ⟨H_C⟩ chính xác từ statevector (không cần shot sampling).

    Parameters
    ----------
    statevector : np.ndarray
        Vector trạng thái 2^n chiều.
    Q : np.ndarray
        Ma trận QUBO.

    Returns
    -------
    float
        ⟨ψ|H_C|ψ⟩ chính xác.
    """
    n = Q.shape[0]
    expectation = 0.0
    for idx in range(2 ** n):
        prob = abs(statevector[idx]) ** 2
        if prob < 1e-12:
            continue
        x = np.array([(idx >> i) & 1 for i in range(n)], dtype=float)
        expectation += prob * float(x @ Q @ x)
    return expectation


def sample_best_bitstring(
    counts: dict,
    Q: np.ndarray,
    budget: int,
) -> tuple[np.ndarray, float]:
    """
    Chọn bitstring tốt nhất từ counts: ưu tiên feasible, rồi minimize objective.

    Parameters
    ----------
    counts : dict
        {bitstring: count}.
    Q : np.ndarray
        Ma trận QUBO.
    budget : int
        Số tài sản cần chọn.

    Returns
    -------
    tuple[np.ndarray, float]
        (x_best, objective_value)
    """
    best_x, best_val = None, float("inf")
    fallback_x, fallback_val = None, float("inf")

    for bitstring in counts:
        x = np.array([int(b) for b in reversed(bitstring)], dtype=float)
        val = float(x @ Q @ x)
        if int(x.sum()) == budget:
            if val < best_val:
                best_val, best_x = val, x.copy()
        else:
            if val < fallback_val:
                fallback_val, fallback_x = val, x.copy()

    if best_x is not None:
        return best_x, best_val

    logger.warning("Không tìm được feasible bitstring. Dùng fallback.")
    return fallback_x, fallback_val


def statevector_best_bitstring(
    statevector: np.ndarray,
    Q: np.ndarray,
    budget: int,
    n: int,
) -> tuple[np.ndarray, float]:
    """
    Tìm bitstring tốt nhất từ statevector (exact probability).

    Chọn feasible state có amplitude lớn nhất.

    Parameters
    ----------
    statevector : np.ndarray
    Q : np.ndarray
    budget : int
    n : int
        Số qubit.

    Returns
    -------
    tuple[np.ndarray, float]
    """
    best_x, best_val = None, float("inf")
    best_prob = -1.0

    for idx in range(2 ** n):
        x = np.array([(idx >> i) & 1 for i in range(n)], dtype=float)
        if int(x.sum()) != budget:
            continue
        prob = abs(statevector[idx]) ** 2
        val = float(x @ Q @ x)
        # Ưu tiên: feasible với prob cao nhất
        # (nếu muốn min objective thay vì max prob thì đổi thành val < best_val)
        if prob > best_prob:
            best_prob, best_val, best_x = prob, val, x.copy()

    if best_x is None:
        # Fallback: state có prob cao nhất bất kể feasibility
        probs = np.abs(statevector) ** 2
        idx = int(np.argmax(probs))
        best_x = np.array([(idx >> i) & 1 for i in range(n)], dtype=float)
        best_val = float(best_x @ Q @ best_x)

    return best_x, best_val


# ══════════════════════════════════════════════════════════════════
# Classical optimizers (thuần Python, không cần qiskit_algorithms)
# ══════════════════════════════════════════════════════════════════

class COBYLAOptimizer:
    """
    COBYLA — Constrained Optimization BY Linear Approximation.
    Gradient-free, ổn định với hàm mục tiêu nhiễu (shot noise).
    Khuyến nghị cho QAOA với depth p ≤ 3.
    """

    def __init__(self, maxiter: int = 200, rhobeg: float = 0.5):
        self.maxiter = maxiter
        self.rhobeg = rhobeg

    def minimize(
        self, fun: Callable, x0: np.ndarray
    ) -> tuple[np.ndarray, float, int, list]:
        from scipy.optimize import minimize

        history = []
        n_evals = [0]

        def tracked(x):
            val = fun(x)
            n_evals[0] += 1
            history.append(float(val))
            return val

        result = minimize(
            tracked, x0, method="COBYLA",
            options={"maxiter": self.maxiter, "rhobeg": self.rhobeg},
        )
        return result.x, float(result.fun), n_evals[0], history


class SPSAOptimizer:
    """
    SPSA — Simultaneous Perturbation Stochastic Approximation.
    Hiệu quả khi nhiều tham số (depth p ≥ 4) hoặc noise cao.
    """

    def __init__(
        self,
        maxiter: int = 200,
        learning_rate: float = 0.1,
        perturbation: float = 0.1,
        seed: int = 42,
    ):
        self.maxiter = maxiter
        self.a = learning_rate
        self.c = perturbation
        self.rng = np.random.default_rng(seed)

    def minimize(
        self, fun: Callable, x0: np.ndarray
    ) -> tuple[np.ndarray, float, int, list]:
        x = x0.copy()
        n = len(x)
        n_evals = 0
        history = []

        for k in range(1, self.maxiter + 1):
            ak = self.a / (k + 1) ** 0.602
            ck = self.c / (k + 1) ** 0.101
            delta = np.where(self.rng.random(n) > 0.5, 1.0, -1.0)

            f_plus  = fun(x + ck * delta); n_evals += 1
            f_minus = fun(x - ck * delta); n_evals += 1

            grad = (f_plus - f_minus) / (2 * ck * delta)
            x -= ak * grad

            val = fun(x); n_evals += 1
            history.append(float(val))

        return x, float(fun(x)), n_evals, history


# ══════════════════════════════════════════════════════════════════
# Helper: chạy statevector đúng cách với Qiskit Aer
# ══════════════════════════════════════════════════════════════════

def _run_statevector(qc, simulator) -> np.ndarray:
    """
    Lấy statevector từ circuit — thử 3 cách, fallback an toàn.

    Các phiên bản Qiskit/Aer khác nhau có API khác nhau:
      - Aer ≥ 0.12 : cần save_statevector() instruction
      - Aer < 0.12  : get_statevector() trực tiếp từ result
      - Fallback    : Statevector.from_instruction() (qiskit-terra, không cần Aer)

    Parameters
    ----------
    qc : QuantumCircuit
        Circuit KHÔNG có measurement gates.
    simulator : AerSimulator
        Backend với method='statevector'.

    Returns
    -------
    np.ndarray
        Statevector phức (2^n,).
    """
    from qiskit import transpile

    def _to_array(sv_obj) -> np.ndarray:
        """Chuyển Statevector object → numpy array."""
        if hasattr(sv_obj, "data"):
            return np.array(sv_obj.data)
        if hasattr(sv_obj, "_data"):
            return np.array(sv_obj._data)
        return np.array(sv_obj)

    # ── Cách 1: save_statevector() instruction (Aer ≥ 0.12) ──────
    try:
        qc_sv = qc.copy()
        qc_sv.save_statevector()
        qc_t = transpile(qc_sv, simulator)
        result = simulator.run(qc_t).result()
        if result.status == "ERROR" or not result.success:
            raise RuntimeError(result.status)
        sv_obj = result.get_statevector(qc_t)
        return _to_array(sv_obj)
    except Exception as e1:
        logger.debug(f"save_statevector method failed: {e1}")

    # ── Cách 2: get_statevector() trực tiếp (Aer < 0.12) ─────────
    try:
        qc_t = transpile(qc, simulator)
        result = simulator.run(qc_t).result()
        sv_obj = result.get_statevector()
        return _to_array(sv_obj)
    except Exception as e2:
        logger.debug(f"direct get_statevector failed: {e2}")

    # ── Cách 3: Statevector.from_instruction() (qiskit-terra) ─────
    try:
        from qiskit.quantum_info import Statevector
        sv_obj = Statevector.from_instruction(qc)
        return _to_array(sv_obj)
    except Exception as e3:
        logger.debug(f"Statevector.from_instruction failed: {e3}")

    raise RuntimeError(
        "Không thể lấy statevector bằng bất kỳ phương pháp nào. "
        "Thử dùng --sim-mode ideal thay vì statevector."
    )


# ══════════════════════════════════════════════════════════════════
# Main optimizer: optimize_qaoa
# ══════════════════════════════════════════════════════════════════

def optimize_qaoa(
    qubo,
    simulator_mode: str = "ideal",
    depth: int = 2,
    optimizer_name: str = "COBYLA",
    max_iterations: int = 200,
    shots: int = 1024,
    seed: int = 42,
    noise_params: dict | None = None,
) -> OptimizationResult:
    """
    Chạy vòng lặp tối ưu hóa QAOA hybrid trên AerSimulator.

    Parameters
    ----------
    qubo : QUBOProblem
        Bài toán QUBO đã formulate.
    simulator_mode : str
        'statevector' | 'ideal' | 'noisy'
        - statevector : exact, không shot noise, nhanh, tốt để debug
        - ideal       : shot-based, không gate noise
        - noisy       : shot-based + IBM-like noise model
    depth : int
        Số layer QAOA (p).
    optimizer_name : str
        'COBYLA' hoặc 'SPSA'.
    max_iterations : int
        Số vòng lặp tối đa.
    shots : int
        Số lần đo mỗi circuit (bỏ qua khi mode='statevector').
    seed : int
        Random seed.
    noise_params : dict, optional
        Override noise parameters khi mode='noisy'.

    Returns
    -------
    OptimizationResult
    """
    from qiskit import transpile

    from .backend import get_simulator
    from .circuit import (
        bind_parameters,
        build_qaoa_circuit,
        get_circuit_config,
    )

    start_time = time.time()
    Q = qubo.Q
    n = qubo.n_assets
    use_statevector = (simulator_mode == "statevector")

    logger.info(
        f"QAOA: n={n} qubits, depth={depth}, mode={simulator_mode}, "
        f"optimizer={optimizer_name}, maxiter={max_iterations}"
        + (f", shots={shots}" if not use_statevector else ", exact statevector")
    )

    # ── Lấy simulator ─────────────────────────────────────────────
    simulator = get_simulator(mode=simulator_mode, noise_params=noise_params)

    # ── Khởi tạo tham số ──────────────────────────────────────────
    config = get_circuit_config(n, depth=depth, seed=seed)
    x0 = config.initial_params

    # ── Build mạch template ───────────────────────────────────────
    # statevector mode: dùng mạch không có measurement
    # shot-based mode: dùng mạch có measurement
    qc_template = build_qaoa_circuit(Q, depth=depth)
    if use_statevector:
        qc_template_sv = qc_template.remove_final_measurements(inplace=False)
    convergence_history = []

    # ── Hàm mục tiêu ──────────────────────────────────────────────
    def objective(params: np.ndarray) -> float:
        if use_statevector:
            qc_bound = bind_parameters(qc_template_sv, params, depth)
            sv = _run_statevector(qc_bound, simulator)
            exp_val = statevector_to_expectation(sv, Q)
        else:
            qc_bound = bind_parameters(qc_template, params, depth)
            qc_t = transpile(qc_bound, simulator)
            job = simulator.run(qc_t, shots=shots)
            counts = job.result().get_counts()
            exp_val = counts_to_expectation(counts, Q, shots)

        convergence_history.append(float(exp_val))
        return exp_val

    # ── Chọn và chạy optimizer ────────────────────────────────────
    if optimizer_name.upper() == "SPSA":
        opt = SPSAOptimizer(maxiter=max_iterations, seed=seed)
    else:
        opt = COBYLAOptimizer(maxiter=max_iterations)

    optimal_params, _, n_evals, history = opt.minimize(objective, x0)
    convergence_history = history

    # ── Final measurement với tham số tối ưu ─────────────────────
    final_shots = shots * 4

    if use_statevector:
        qc_final = bind_parameters(qc_template_sv, optimal_params, depth)
        sv = _run_statevector(qc_final, simulator)
        best_x, best_val = statevector_best_bitstring(sv, Q, qubo.budget, n)
    else:
        qc_final = bind_parameters(qc_template, optimal_params, depth)
        qc_t = transpile(qc_final, simulator)
        job = simulator.run(qc_t, shots=final_shots)
        final_counts = job.result().get_counts()
        best_x, best_val = sample_best_bitstring(final_counts, Q, qubo.budget)

    elapsed = time.time() - start_time
    weights = qubo.decode_weights(best_x)
    feasible = qubo.is_feasible(best_x)

    logger.info(
        f"Hoàn thành: {elapsed:.1f}s, {n_evals} evals, "
        f"obj={best_val:.6f}, feasible={feasible}"
    )

    return OptimizationResult(
        best_bitstring=best_x,
        best_objective=best_val,
        optimal_params=optimal_params,
        weights=weights,
        n_iterations=n_evals,
        convergence_history=convergence_history,
        feasible=feasible,
        elapsed_seconds=elapsed,
        simulator_mode=simulator_mode,
        shots=shots if not use_statevector else 0,
    )


def run_all_modes(
    qubo,
    depth: int = 2,
    max_iterations: int = 100,
    shots: int = 1024,
    seed: int = 42,
) -> dict[str, OptimizationResult]:
    """
    Chạy QAOA trên cả 3 simulator modes và so sánh kết quả.

    Hữu ích để đánh giá ảnh hưởng của noise lên kết quả.

    Parameters
    ----------
    qubo : QUBOProblem
    depth : int
    max_iterations : int
    shots : int
    seed : int

    Returns
    -------
    dict[str, OptimizationResult]
        {'statevector': ..., 'ideal': ..., 'noisy': ...}
    """
    results = {}
    for mode in ("statevector", "ideal", "noisy"):
        logger.info(f"\n--- Mode: {mode} ---")
        results[mode] = optimize_qaoa(
            qubo=qubo,
            simulator_mode=mode,
            depth=depth,
            max_iterations=max_iterations,
            shots=shots,
            seed=seed,
        )
    return results


def print_mode_comparison(results: dict[str, "OptimizationResult"]) -> None:
    """In bảng so sánh kết quả giữa các simulator modes."""
    print("\n" + "=" * 65)
    print("SO SÁNH KẾT QUẢ GIỮA CÁC SIMULATOR MODES")
    print("=" * 65)
    print(f"{'Mode':<15} {'Objective':>12} {'Feasible':>10} {'Time(s)':>9} {'Bitstring'}")
    print("-" * 65)
    for mode, r in results.items():
        print(
            f"{mode:<15} {r.best_objective:>12.6f} "
            f"{'✓' if r.feasible else '✗':>10} "
            f"{r.elapsed_seconds:>9.1f} "
            f"{''.join(map(str, r.best_bitstring.astype(int)))}"
        )
    print("=" * 65)