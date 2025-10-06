# training/callbacks.py
from common.imports import *

# Global trackers (same semantics as your script)
objective_func_vals: list = []
learning_rates: list = []
perturbations: list = []
momentum: float = 0.95
gradient_moving_avg = 0.0
current_round = 1

def deep_unfolding_learning_rate_adjustment(parameters, obj_func_eval, gradients=None,
                                            round_number=0,
                                            initial_learning_rate=0.15,
                                            initial_perturbation=0.15):
    """
    Updates lists: objective_func_vals, learning_rates, perturbations.
    Tolerates gradients=None (some backends).
    """
    global gradient_moving_avg, learning_rates, perturbations, current_round, objective_func_vals
    objective_func_vals.append(obj_func_eval)

    if gradients is not None:
        gradient_moving_avg = momentum * gradient_moving_avg + (1 - momentum) * gradients
        delta_lr = 0.05 * gradient_moving_avg
        delta_pert = 0.10 * gradient_moving_avg
    else:
        delta_lr = 0.0
        delta_pert = 0.0

    if learning_rates:
        new_lr = max(0.001, learning_rates[-1] + delta_lr)
        new_pert = max(0.001, perturbations[-1] + delta_pert)
    else:
        new_lr = initial_learning_rate
        new_pert = initial_perturbation

    learning_rates.append(new_lr)
    perturbations.append(new_pert)
    current_round += 1

def spsa_callback(nfev, parameters, obj_func_eval, stepsize, accept):
    """
    Version-robust: stepsize may be None.
    """
    gradients = stepsize if stepsize is not None else 0.0
    deep_unfolding_learning_rate_adjustment(parameters, obj_func_eval, gradients)

def reset_state():
    """
    Clear per-client lists between clients if desired.
    """
    global objective_func_vals, learning_rates, perturbations
    objective_func_vals = []
    learning_rates = []
    perturbations = []

def reset_callback_graph(initial_learning_rate=0.15, initial_perturbation=0.15):
    """
    Reinitialize graph-wise state (if you want a fresh start between rounds).
    Only call if you INTEND to reset learnability state.
    """
    global gradient_moving_avg, learning_rates, perturbations
    gradient_moving_avg = 0.0
    learning_rates = [initial_learning_rate]
    perturbations = [initial_perturbation]
