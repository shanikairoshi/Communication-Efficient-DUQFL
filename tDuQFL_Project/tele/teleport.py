# tele/teleport.py
from common.imports import *
from tele.noise import make_backend

def _teleport_circuit(theta: float) -> QuantumCircuit:
    qc = QuantumCircuit(3, name='teleport')
    c0 = ClassicalRegister(1, 'c0')
    c1 = ClassicalRegister(1, 'c1')
    qc.add_register(c0); qc.add_register(c1)
    qc.rx(theta, 0); qc.h(1); qc.cx(1, 2)
    qc.cx(0, 1); qc.h(0)
    qc.measure(0, c0[0]); qc.measure(1, c1[0])
    qc.z(2).c_if(c0, 1);  qc.x(2).c_if(c1, 1)
    return qc

def teleport_parameter(val: float, backend: Optional[AerSimulator]=None,
                       noise: Literal['none','low','med','high']='none', shots: int=1024,
                       show_histogram: bool=False) -> float:
    if backend is None:
        backend = make_backend(noise=noise, shots=shots)
    qc = _teleport_circuit(val)
    tqc = transpile(qc, backend)
    result = backend.run(tqc).result()
    counts = result.get_counts()
    if show_histogram:
        plot_histogram(counts); plt.show()
    return float(val)

# (Your parameterized version with decode/shrinkage)
def build_param_teleport_circuit() -> Tuple[QuantumCircuit, Parameter]:
    theta = Parameter('theta')
    qc = QuantumCircuit(3, name='teleport')
    c0 = ClassicalRegister(1, 'c0')
    c1 = ClassicalRegister(1, 'c1')
    c2 = ClassicalRegister(1, 'c2')
    qc.add_register(c0); qc.add_register(c1); qc.add_register(c2)
    qc.rx(theta, 0)
    qc.h(1); qc.cx(1, 2)
    qc.cx(0, 1); qc.h(0)
    qc.measure(0, c0[0]); qc.measure(1, c1[0])
    qc.z(2).c_if(c0, 1); qc.x(2).c_if(c1, 1)
    qc.measure(2, c2[0])
    return qc, theta

def _p1_from_counts_lastbit(counts: Dict[str,int]) -> float:
    n0 = n1 = 0
    for bs, c in counts.items():
        b = bs.replace(' ','')[-1]
        if b == '0': n0 += c
        elif b == '1': n1 += c
    tot = n0 + n1
    return 0.0 if tot == 0 else n1 / tot

def _decode_angle_from_p1(p1: float) -> float:
    p1 = float(np.clip(p1, 0.0, 1.0))
    return float(2.0 * np.arcsin(np.sqrt(p1)))

def _fidelity_proxy(theta_in: float, theta_out: float) -> float:
    return float(np.cos(0.5 * (theta_out - theta_in))**2)

def _shrinkage_from_fidelity(fid: np.ndarray, lo: float=0.70, hi: float=0.95) -> float:
    Fav = float(np.mean(np.clip(fid, 0.0, 1.0)))
    if Fav >= hi: return 1.0
    if Fav <= lo: return 0.0
    return (Fav - lo) / (hi - lo)

def teleport_parameters(params: np.ndarray, *, mode: Literal['demo','perturb']='demo',
                        noise: Literal['none','low','med','high']='none', shots: int=1024,
                        backend: Optional[AerSimulator]=None, show_histogram: bool=False,
                        apply_shrinkage: bool=True):
    theta_in = np.asarray(params, dtype=float).reshape(-1)
    if backend is None:
        backend = make_backend(noise=noise, shots=shots)

    qc, theta_sym = build_param_teleport_circuit()
    tqc = transpile(qc, backend)
    circuits = [tqc.assign_parameters({theta_sym: t}, inplace=False) for t in theta_in]
    result = backend.run(circuits).result()

    theta_out, delta, fid = [], [], []
    for i, t_in in enumerate(theta_in):
        counts = result.get_counts(i)
        if show_histogram: plot_histogram(counts); plt.show()
        p1 = _p1_from_counts_lastbit(counts)
        t_eff = _decode_angle_from_p1(p1)
        theta_out.append(t_eff)
        delta.append(t_eff - t_in)
        fid.append(_fidelity_proxy(t_in, t_eff))

    theta_out = np.asarray(theta_out, dtype=np.float32).reshape(params.shape)
    delta     = np.asarray(delta,     dtype=np.float32).reshape(params.shape)
    fid       = np.asarray(fid,       dtype=np.float32).reshape(params.shape)

    if mode == 'demo':
        return np.asarray(params, dtype=np.float32), {
            'theta_out': theta_out, 'delta_tel': delta, 'fidelity': fid,
            'beta': 0.0, 'noise': noise, 'shots': shots
        }
    beta = _shrinkage_from_fidelity(fid) if apply_shrinkage else 1.0
    params_out = np.asarray(params, dtype=np.float32) + beta * delta
    return params_out, {
        'theta_out': theta_out, 'delta_tel': delta, 'fidelity': fid,
        'beta': beta, 'noise': noise, 'shots': shots
    }
