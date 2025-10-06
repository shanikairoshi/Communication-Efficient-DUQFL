# ml/optimizers.py
from common.imports import *
from qiskit_algorithms.optimizers import SPSA
from configs.base_config import momentum as MOMENTUM
import importlib, ml.optimizers


# ------------------- global trackers (for plots/logging) -------------------
objective_func_vals: list[float] = []
learning_rates:      list[float] = []
perturbations:       list[float] = []
gradient_moving_avg: float = 0.0
_current_round:      int = 1
MOMENTUM = 0.95

# callback instrumentation
callback_calls = 0

# Trust-region caps (keep modest)
LR_MIN,   LR_MAX   = 1e-3, 2.50
PERT_MIN, PERT_MAX = 1e-3, 2.50

# ---------- PER-FIT NORMALIZATION ----------
# This scale should approximate 1 / (#callback calls during one fit).
# We will set it from the training loop using the optimizer's maxiter.
UPDATE_SCALE = 1.0

def set_update_scale(v: float) -> None:
    global UPDATE_SCALE
    UPDATE_SCALE = float(max(1e-6, v))

# ---------- DEEP-UNFOLD CONTROLLER ----------
def deep_unfolding_learning_rate_adjustment(parameters,
                                            obj_func_eval,
                                            gradients=None,
                                            round_number: int = 0):
    """
    Append new LR/PERT based on a momentum-smoothed gradient proxy.
    Each callback contributes a small nudge, scaled by UPDATE_SCALE so that
    the sum over callbacks during one fit is of reasonable magnitude.
    """
    global gradient_moving_avg
    objective_func_vals.append(float(obj_func_eval))

    if gradients is not None:
        try:
            g = float(np.mean(gradients))
        except Exception:
            try:
                g = float(gradients)
            except Exception:
                g = 0.0
        gradient_moving_avg = MOMENTUM * gradient_moving_avg + (1.0 - MOMENTUM) * g
        delta_lr           = (0.05 * gradient_moving_avg) * UPDATE_SCALE
        delta_perturbation = (0.10 * gradient_moving_avg) * UPDATE_SCALE
    else:
        delta_lr = 0.0
        delta_perturbation = 0.0

    base_lr   = float(learning_rates[-1]) if learning_rates else 0.15
    base_pert = float(perturbations[-1])  if perturbations  else 0.15

    new_lr   = float(np.clip(base_lr   + delta_lr,           LR_MIN,   LR_MAX))
    new_pert = float(np.clip(base_pert + delta_perturbation, PERT_MIN, PERT_MAX))

    learning_rates.append(new_lr)
    perturbations.append(new_pert)
    return new_lr, new_pert


# ------------------- SPSA callback (fires each SPSA iteration) -------------
def spsa_callback(nfev, parameters, obj_func_eval, stepsize, accept):
    """
    Qiskit passes (nfev, x, fx, stepsize, accept). We treat stepsize as a
    proxy for gradients (same as your notebook) and log/update trackers.
    """
    gradients = stepsize
    deep_unfolding_learning_rate_adjustment(parameters, obj_func_eval, gradients)


# ------------------- Learnable SPSA wrapper --------------------------------
class LearnableLRPerturbationSPSA(SPSA):
    """
    SPSA with learnable LR and perturbation. We do NOT override internal
    stepping (Qiskit calls minimize(...)); instead, we expose a method
    to apply the latest tracked LR/PERT to this live optimizer instance.
    """
    def __init__(self, *, initial_lr: float, initial_perturbation: float,
                 lr_alpha: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        # Use the attribute names Qiskit actually reads:
        self.learning_rate = float(initial_lr)
        self.perturbation  = float(initial_perturbation)
        self.lr_alpha      = float(lr_alpha)

    def apply_updates_from_trackers(self):
        """
        Call this AFTER model.fit(...) at the end of each deep-unfolding
        iteration to push the newly learned LR/PERT to the optimizer.
        """
        if learning_rates:
            self.learning_rate = float(learning_rates[-1])
        if perturbations:
            self.perturbation = float(perturbations[-1])

    def reset(self, *, initial_learning_rate: float, initial_perturbation: float):
        """
        Optional: reinitialize for a fresh round/client as in your notebook.
        """
        global gradient_moving_avg
        self.learning_rate = float(initial_learning_rate)
        self.perturbation  = float(initial_perturbation)
        gradient_moving_avg = 0.0
        learning_rates.clear()
        perturbations.clear()
        objective_func_vals.clear()


# ------------------- factory -----------------------------------------------
def make_spsa(*, maxiter: int,
              initial_learning_rate: float,
              initial_perturbation: float,
              lr_alpha: float = 0.01) -> LearnableLRPerturbationSPSA:
    """
    Create a learnable SPSA wired with our callback that updates trackers.
    """
    return LearnableLRPerturbationSPSA(
        maxiter=maxiter,
        learning_rate=initial_learning_rate,
        perturbation=initial_perturbation,
        callback=spsa_callback,
        initial_lr=initial_learning_rate,
        initial_perturbation=initial_perturbation,
        lr_alpha=lr_alpha,
    )
