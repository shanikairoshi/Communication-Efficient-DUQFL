# tele/noise.py
from common.imports import *

def _build_noise_model(p1: float, p2: float) -> NoiseModel:
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ['rx','rz','sx','x','h'])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ['cx'])
    return nm

def make_backend(noise: Literal['none','low','med','high']='none', shots: int=1024) -> AerSimulator:
    if noise == 'none':
        return AerSimulator(shots=shots)
    elif noise == 'low':
        nm = _build_noise_model(0.001, 0.005)
    elif noise == 'med':
        nm = _build_noise_model(0.005, 0.015)
    elif noise == 'high':
        nm = _build_noise_model(0.01,  0.02)
    else:
        raise ValueError(f"unknown noise preset {noise!r}")
    return AerSimulator(noise_model=nm, shots=shots)
