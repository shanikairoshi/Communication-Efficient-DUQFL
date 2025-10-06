# viz/plots.py
from common.imports import *

def plot_accuracy_curve(rounds, acc, label="Global accuracy"):
    plt.figure(figsize=(6.5,4)); plt.plot(rounds, acc, marker="o", lw=2)
    plt.xlabel("Federated round"); plt.ylabel("Accuracy"); plt.title(label)
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

def plot_val_loss(rounds, losses, label="Central validation loss"):
    plt.figure(figsize=(6.5,4)); plt.plot(rounds, losses, marker="o", lw=2)
    plt.xlabel("Federated round"); plt.ylabel("Loss"); plt.title(label)
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

def plot_time_per_round(rounds, times):
    plt.figure(figsize=(6.5,4)); plt.bar(rounds, times)
    plt.xlabel("Federated round"); plt.ylabel("Seconds"); plt.title("Wall-clock time per round")
    plt.grid(axis='y', alpha=0.3); plt.tight_layout(); plt.show()

def plot_fidelity_vs_delta_acc(fidelity_means, delta_acc, label="ΔAccuracy vs mean fidelity"):
    mask = np.isfinite(fidelity_means) & np.isfinite(delta_acc)
    f, d = np.asarray(fidelity_means)[mask], np.asarray(delta_acc)[mask]
    plt.figure(figsize=(6.0,4.5)); plt.scatter(f, d, s=30)
    plt.xlabel("Mean teleportation fidelity (per round)"); plt.ylabel("Δ Accuracy (round-to-round)")
    plt.title(label); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

def plot_beta_hist(betas):
    b = np.asarray(betas); b = b[np.isfinite(b)]
    if b.size == 0: print("No β values to plot."); return
    plt.figure(figsize=(6.0,4.0)); plt.hist(b, bins=10)
    plt.xlabel("β (fidelity-weighted shrinkage)"); plt.ylabel("Count"); plt.title("Distribution of β across rounds")
    plt.grid(axis='y', alpha=0.3); plt.tight_layout(); plt.show()

def plot_client_fairness_last_round(client_accs_last):
    arr = np.asarray(client_accs_last, dtype=float)
    if arr.size == 0: print("No client accuracies to plot."); return
    plt.figure(figsize=(6.0,4.0)); plt.boxplot(arr, vert=True, labels=["final"])
    plt.ylabel("Client accuracy"); plt.title("Client fairness at final round")
    plt.grid(axis='y', alpha=0.3); plt.tight_layout(); plt.show()
