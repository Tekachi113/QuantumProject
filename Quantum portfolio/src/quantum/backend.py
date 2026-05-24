"""
src/quantum/backend.py
----------------------
Quản lý simulator backend cho QAOA.

Hỗ trợ 3 chế độ giả lập (không cần IBM account):
  1. statevector  — tính chính xác bằng statevector, không có shot noise
                    → tốt nhất để debug và verify circuit
  2. ideal        — AerSimulator shot-based, không có gate noise
                    → nhanh, sát thực tế hơn statevector
  3. noisy        — AerSimulator + noise model giả lập hardware IBM
                    → sát kết quả hardware thật nhất có thể trên local

Không có kết nối cloud, không cần token.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (
        NoiseModel,
        depolarizing_error,
        thermal_relaxation_error,
    )
    AER_AVAILABLE = True
except ImportError:
    AER_AVAILABLE = False
    logger.warning("qiskit-aer chưa được cài. Chạy: pip install qiskit-aer")


# ── Tham số noise giả lập hardware IBM (Brisbane-like) ────────────────────────
_IBM_NOISE_PARAMS = {
    # Gate error rates (depolarizing)
    "single_qubit_error": 1e-3,   # ~0.1% lỗi mỗi single-qubit gate
    "two_qubit_error":    1e-2,   # ~1%   lỗi mỗi two-qubit gate (CX, CZ)
    "readout_error":      1e-2,   # ~1%   lỗi đo

    # Thermal relaxation
    "t1_us": 100.0,   # T1 = 100 μs (amplitude damping)
    "t2_us":  80.0,   # T2 = 80  μs (phase damping)

    # Gate times (μs)
    "gate_time_1q_us": 0.05,   # 50 ns
    "gate_time_2q_us": 0.30,   # 300 ns
}


def _build_noise_model(params: dict | None = None) -> "NoiseModel":
    """
    Xây dựng NoiseModel giả lập hardware IBM.

    Bao gồm:
    - Depolarizing error cho single-qubit và two-qubit gates
    - Thermal relaxation (T1/T2) theo gate time
    - Readout error

    Parameters
    ----------
    params : dict, optional
        Override tham số noise. Mặc định dùng _IBM_NOISE_PARAMS.

    Returns
    -------
    NoiseModel
    """
    if not AER_AVAILABLE:
        raise ImportError("qiskit-aer chưa được cài.")

    p = {**_IBM_NOISE_PARAMS, **(params or {})}
    noise_model = NoiseModel()

    # ── Single-qubit gate errors ────────────────────────────────
    err_1q_depol = depolarizing_error(p["single_qubit_error"], 1)

    t1_ns = p["t1_us"] * 1000
    t2_ns = p["t2_us"] * 1000
    gate_1q_ns = p["gate_time_1q_us"] * 1000
    err_1q_relax = thermal_relaxation_error(t1_ns, t2_ns, gate_1q_ns)

    err_1q = err_1q_depol.compose(err_1q_relax)
    noise_model.add_all_qubit_quantum_error(err_1q, ["u1", "u2", "u3", "rx", "ry", "rz", "h", "x"])

    # ── Two-qubit gate errors ───────────────────────────────────
    err_2q_depol = depolarizing_error(p["two_qubit_error"], 2)

    gate_2q_ns = p["gate_time_2q_us"] * 1000
    err_2q_relax_0 = thermal_relaxation_error(t1_ns, t2_ns, gate_2q_ns)
    err_2q_relax_1 = thermal_relaxation_error(t1_ns, t2_ns, gate_2q_ns)
    err_2q_relax = err_2q_relax_0.expand(err_2q_relax_1)

    err_2q = err_2q_depol.compose(err_2q_relax)
    noise_model.add_all_qubit_quantum_error(err_2q, ["cx", "cz", "rzz", "ecr"])

    # ── Readout error ───────────────────────────────────────────
    ro_err = p["readout_error"]
    noise_model.add_all_qubit_readout_error([[1 - ro_err, ro_err],
                                             [ro_err,     1 - ro_err]])

    logger.info(
        f"Noise model: 1q_err={p['single_qubit_error']:.0e}, "
        f"2q_err={p['two_qubit_error']:.0e}, "
        f"readout={p['readout_error']:.0e}, "
        f"T1={p['t1_us']}μs, T2={p['t2_us']}μs"
    )
    return noise_model


def get_statevector_simulator() -> "AerSimulator":
    """
    AerSimulator dùng statevector method.

    Tính chính xác biên độ xác suất — không có shot noise.
    Dùng để debug và verify circuit trước khi chạy shot-based.

    Returns
    -------
    AerSimulator (method=statevector)
    """
    if not AER_AVAILABLE:
        raise ImportError("qiskit-aer chưa được cài.")
    sim = AerSimulator(method="statevector")
    logger.info("Backend: AerSimulator (statevector, exact)")
    return sim


def get_ideal_simulator() -> "AerSimulator":
    """
    AerSimulator shot-based, không có gate noise hay readout error.

    Phù hợp để chạy thực nghiệm với số shots tùy chỉnh
    mà không cần lo về noise.

    Returns
    -------
    AerSimulator (method=automatic)
    """
    if not AER_AVAILABLE:
        raise ImportError("qiskit-aer chưa được cài.")
    sim = AerSimulator()
    logger.info("Backend: AerSimulator (ideal, shot-based)")
    return sim


def get_noisy_simulator(noise_params: dict | None = None) -> "AerSimulator":
    """
    AerSimulator với noise model giả lập hardware IBM.

    Bao gồm depolarizing error, thermal relaxation (T1/T2),
    và readout error — không cần kết nối cloud.

    Parameters
    ----------
    noise_params : dict, optional
        Override tham số noise mặc định. Các key hợp lệ:
        single_qubit_error, two_qubit_error, readout_error,
        t1_us, t2_us, gate_time_1q_us, gate_time_2q_us.

    Returns
    -------
    AerSimulator với NoiseModel
    """
    if not AER_AVAILABLE:
        raise ImportError("qiskit-aer chưa được cài.")
    noise_model = _build_noise_model(noise_params)
    sim = AerSimulator(noise_model=noise_model)
    logger.info("Backend: AerSimulator (noisy, IBM-like noise model)")
    return sim


# Mapping tên → hàm tạo backend
_SIMULATOR_REGISTRY = {
    "statevector": get_statevector_simulator,
    "ideal":       get_ideal_simulator,
    "noisy":       get_noisy_simulator,
}


def get_simulator(mode: str = "ideal", noise_params: dict | None = None):
    """
    Factory function — chọn simulator theo mode.

    Parameters
    ----------
    mode : str
        Một trong: 'statevector', 'ideal', 'noisy'.
        - statevector : tính chính xác, không shot noise
        - ideal       : shot-based, không gate noise
        - noisy       : shot-based + IBM-like noise model
    noise_params : dict, optional
        Chỉ dùng khi mode='noisy'. Override noise parameters.

    Returns
    -------
    AerSimulator

    Raises
    ------
    ValueError
        Nếu mode không hợp lệ.
    """
    mode = mode.lower()
    if mode not in _SIMULATOR_REGISTRY:
        raise ValueError(
            f"mode='{mode}' không hợp lệ. "
            f"Chọn một trong: {list(_SIMULATOR_REGISTRY.keys())}"
        )
    if mode == "noisy":
        return get_noisy_simulator(noise_params)
    return _SIMULATOR_REGISTRY[mode]()


def print_simulator_info(simulator, mode: str) -> None:
    """In thông tin simulator ra console."""
    descriptions = {
        "statevector": "Statevector (exact, không shot noise)",
        "ideal":       "Ideal shot-based (không gate noise)",
        "noisy":       "Noisy shot-based (IBM-like noise model)",
    }
    print(f"\nSimulator: AerSimulator")
    print(f"  Mode    : {descriptions.get(mode, mode)}")
    print(f"  Local   : ✓ (không cần IBM account)")


def compare_modes_info() -> str:
    """Trả về bảng so sánh các simulator mode."""
    return """
┌─────────────────┬──────────────┬────────────┬──────────────────────────────┐
│ Mode            │ Shot noise   │ Gate noise │ Dùng khi                     │
├─────────────────┼──────────────┼────────────┼──────────────────────────────┤
│ statevector     │ Không        │ Không      │ Debug, verify circuit        │
│ ideal           │ Có (shots)   │ Không      │ Thực nghiệm sạch             │
│ noisy           │ Có (shots)   │ Có (IBM)   │ Ước lượng kết quả thực tế    │
└─────────────────┴──────────────┴────────────┴──────────────────────────────┘
"""