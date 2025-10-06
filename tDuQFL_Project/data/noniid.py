# data/noniid.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Reuse your Client class
from fl.client import Client

def _label_indices_binary(dataset: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    y = np.array([int(d["label"]) for d in dataset], dtype=int)
    return np.where(y == 0)[0], np.where(y == 1)[0]

def _cyclic_take(pool: np.ndarray, ptr: int, k: int) -> Tuple[np.ndarray, int]:
    if k <= 0:
        return np.empty(0, dtype=int), ptr
    n = len(pool)
    if n == 0:
        raise ValueError("Empty pool: dataset has no samples for one label.")
    end = ptr + k
    if end <= n:
        sel = pool[ptr:end]
        ptr = end % n
    else:
        r = end % n
        sel = np.concatenate([pool[ptr:], pool[:r]])
        ptr = r
    return sel, ptr

def make_non_iid_clients(
    train_data: List[Dict[str, Any]],
    test_data:  List[Dict[str, Any]],
    *,
    num_clients: int,
    num_epochs: int,
    samples_per_epoch: int,
    non_iid_ratio: float = 0.8,         # dominance of the preferred label per client/epoch
    quantity_variation: float = 0.5,    # ±50% around samples_per_epoch
    seed: Optional[int] = None,
    plot: bool = True
) -> List[Client]:
    """
    Build non-IID clients with label skew + quantity variation.
    Even client IDs prefer label 0; odd client IDs prefer label 1.
    Cyclic (wrap-around) sampling avoids depletion; deterministic with 'seed'.
    """
    assert 0.0 <= non_iid_ratio <= 1.0
    assert quantity_variation >= 0.0
    assert num_clients > 0 and num_epochs > 0 and samples_per_epoch > 0

    rng = np.random.default_rng(seed)
    idx0, idx1 = _label_indices_binary(train_data)
    rng.shuffle(idx0); rng.shuffle(idx1)
    p0 = p1 = 0

    label0_counts = np.zeros(num_clients, dtype=int)
    label1_counts = np.zeros(num_clients, dtype=int)

    clients: List[Client] = []

    for cid in range(num_clients):
        client_epochs: List[List[Dict[str, Any]]] = []

        for _ in range(num_epochs):
            scale = 1.0 + rng.uniform(-quantity_variation, quantity_variation)
            m = max(1, int(round(samples_per_epoch * scale)))

            if cid % 2 == 0:
                n0 = int(round(m * non_iid_ratio)); n1 = m - n0
            else:
                n1 = int(round(m * non_iid_ratio)); n0 = m - n1

            sel0, p0 = _cyclic_take(idx0, p0, n0)
            sel1, p1 = _cyclic_take(idx1, p1, n1)

            epoch_list = [train_data[i] for i in sel0] + [train_data[i] for i in sel1]
            rng.shuffle(epoch_list)
            client_epochs.append(epoch_list)

            label0_counts[cid] += n0
            label1_counts[cid] += n1

        clients.append(Client(client_epochs, test_data))

    if plot:
        x = np.arange(num_clients); w = 0.40
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x - w/2, label0_counts, w, label="Label 0")
        ax.bar(x + w/2, label1_counts, w, label="Label 1")
        ax.set_xlabel("Client"); ax.set_ylabel("Total samples (all epochs)")
        ax.set_title(f"Genome non-IID (ratio={non_iid_ratio:.2f}, qty±={quantity_variation:.2f})")
        ax.set_xticks(x); ax.legend(); fig.tight_layout(); plt.show()

    return clients
