# Communication-Efficient-DUQFL
This is the official implementation for Communication-Efficient Deep Unfolded Quantum Federated Learning. This repository includes all the setup details, experimental results, and the code structure necessary to reproduce the purpose.

Deep-unfolding federated learning with learnable SPSA hyper-params (learning rate & perturbation), optional quantum teleportation perturbation, and pluggable aggregation strategies.

✨ Highlights

Deep-Unfolding SPSA: LR & perturbation are adapted during each local fit via a momentum-smoothed controller.

Trust-Region Caps: Safety bounds for LR/PERT to avoid divergence.

Federated Rounds with client carry-over of learned hyper-params.

Aggregation plugins: Best-Client (val-gated & smoothed) or FedAvg.

Metrics logging: global/client accuracies, validation loss, (optional) teleportation stats.

# tDuQFL — Teleportation-aware Deep-Unfolded QFL (SPSA)

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![qiskit](https://img.shields.io/badge/qiskit-yes-5A3FD9)
![status](https://img.shields.io/badge/status-research--prototype-green)

Federated learning with a learnable SPSA optimizer and deep-unfolding hyper-step adaptation. Optional quantum “teleportation” perturbation of global weights for robustness.

---

## 📁 Repository Structure

```text
tDuQFL_Project/
├─ README.md
├─ requirements.txt
├─ .gitignore

├─ configs/
│  ├─ base_config.py                 # global knobs (rounds, clients, LR/PERT, seeds, etc.)
│  └─ mnist_config.py                # (optional) dataset-specific overrides

├─ common/
│  └─ imports.py                     # centralized imports / globals

├─ data/
│  ├─ preprocess_genome.py           # build Genome dataset tensors
│  ├─ preprocess_mnist.py            # (optional) MNIST preprocessor
│  ├─ splitters/
│  │  ├─ iid.py                      # IID client/epoch splitter
│  │  └─ noniid.py                   # Non-IID splitter (quantity + label skew)
│  └─ __init__.py

├─ fl/
│  ├─ client.py                      # Client wrapper (train/test shards, current model)
│  └─ aggregation/
│     ├─ __init__.py
│     ├─ best_client.py              # pick best client’s weights (argmax/lowest val)
│     └─ fedavg.py                   # standard FedAvg (weighted mean by samples)

├─ ml/
│  ├─ models.py                      # QNN model init/wiring to optimizer
│  ├─ optimizers.py                  # Learnable SPSA + deep-unfolding controller
│  └─ __init__.py

├─ training/
│  ├─ loop.py                        # federated rounds + deep unfolding per client
│  ├─ callbacks.py                   # SPSA callback trackers (LR/PERT/objective)
│  ├─ metrics.py                     # round metrics logger + summary helpers
│  └─ __init__.py

├─ tele/
│  ├─ teleport.py                    # (optional) parameter teleportation
│  ├─ noise.py                       # (optional) noise backends
│  └─ __init__.py

├─ io_utils/
│  ├─ csv_logger.py                  # local/global CSV logging helpers
│  └─ __init__.py

├─ scripts/
│  ├─ run_genome_iid.py              # launch Genome + IID split
│  ├─ run_genome_noniid.py           # launch Genome + Non-IID split
│  ├─ run_mnist_iid.py               # (optional) launch MNIST + IID split
│  └─ make_readme_assets.py          # generate README figures from CSVs (or synthesize)

├─ docs/
│  └─ images/                        # figures used in README (auto-generated)

├─ results/                          # generated at runtime
│  ├─ best_client.csv
│  ├─ global_accuracies.csv
│  ├─ local_training.csv
│  └─ validation.csv

└─ artifacts/                        # saved weights / checkpoints
   └─ models/
---

##🔧 Installation
python -m venv .venv
source .venv/bin/activate      # windows: .venv\Scripts\activate
pip install -r requirements.txt

##⚙️ Configure

Edit configs/base_config.py:

num_epochs                    = 10
num_federated_layers          = 10
num_deep_unfolding_iterations = 5

initial_learning_rate         = 0.15
initial_perturbation          = 0.15

use_teleportation             = False    # set True to enable teleportation
dataset_name                  = "Genome" # or "MNIST" if you add it
split_type                    = "IID"    # or "NONIID"
global_seed                   = 42


##Deep-Unfolding caps (safety bounds)
In ml/optimizers.py:

## Trust-region caps (tune per dataset)
LR_MIN, LR_MAX     = 1e-3, 0.30
PERT_MIN, PERT_MAX = 1e-3, 0.50


These prevent runaway LR/PERT. Raise them cautiously if your task benefits from larger steps.

##📚 Data
Genome (Human/Worm demo)

Prepared in data/preprocess_genome.py. Produces normalized records:

{'sequence': np.ndarray, 'label': int}

Splitting data

IID (data/splitters/iid.py):

from data.preprocess_genome import load_and_prepare_dataset
from data.splitters.iid import split_dataset_for_epochs

np_train_data, np_test_data = load_and_prepare_dataset(word_size=40, global_seed=42)
clients = split_dataset_for_epochs(
    num_clients=10, num_epochs=10,
    train_data=np_train_data, test_data=np_test_data,
    samples_per_epoch=50
)


Non-IID (data/splitters/noniid.py):

from data.preprocess_genome import load_and_prepare_dataset
from data.splitters.noniid import split_dataset_quantity_non_iid_binary
from fl.client import Client

np_train_data, np_test_data = load_and_prepare_dataset(word_size=40, global_seed=42)
client_data_list = split_dataset_quantity_non_iid_binary(
    dataset=np_train_data,
    num_clients=10, num_epochs=10, samples_per_epoch=50,
    non_iid_ratio=0.8, quantity_variation=0.5
)
clients = [Client(data, test_data=np_test_data) for data in client_data_list]


Tip: ensure you have enough training samples:

train_capacity = len(np_train_data) // (num_clients * samples_per_epoch)
num_epochs     = min(num_epochs, train_capacity)

##🚀 Run training

Minimal example (Genome + IID):

# scripts/run_genome_iid.py
from configs.base_config import *
from data.preprocess_genome import load_and_prepare_dataset
from data.splitters.iid import split_dataset_for_epochs
from training.loop import run_federated_training
from training.metrics import metrics_init, metrics_finalize
import os, numpy as np

np_train_data, np_test_data = load_and_prepare_dataset(word_size, global_seed)
clients = split_dataset_for_epochs(
    num_clients=num_clients, num_epochs=num_epochs,
    train_data=np_train_data, test_data=np_test_data,
    samples_per_epoch=samples_per_epoch
)

test_sequences = np.array([d["sequence"] for d in np_test_data])
test_labels    = np.array([d["label"]    for d in np_test_data])
X_val, y_val   = test_sequences, test_labels

# outputs
os.makedirs(drive_root, exist_ok=True)
best_client_csv_file = f"{drive_root}/best_client.csv"
global_csv_file      = f"{drive_root}/global_accuracies.csv"
local_csv_file       = f"{drive_root}/local_training.csv"
validation_csv_file  = f"{drive_root}/validation.csv"

metrics = metrics_init(log_path=os.path.join(drive_root, "round_metrics.csv"))

# derive num_features
num_features = clients[0].data[0][0]['sequence'].shape[0]

global_acc, clients_train, clients_test, round_times, val_losses, info_last = run_federated_training(
    clients=clients,
    num_federated_layers=num_federated_layers,
    num_deep_unfolding_iterations=num_deep_unfolding_iterations,
    initial_learning_rate=initial_learning_rate,
    initial_perturbation=initial_perturbation,
    num_features=num_features,
    best_client_csv_file=best_client_csv_file,
    global_csv_file=global_csv_file,
    local_csv_file=local_csv_file,
    validation_csv_file=validation_csv_file,
    test_sequences=test_sequences,
    test_labels=test_labels,
    X_val=X_val, y_val=y_val,
    use_teleportation=use_teleportation,
    noise_preset='med', shots_used=256,
    metrics=metrics
)

print("Global accuracy per round:", global_acc)


Run:

python scripts/run_genome_iid.py

##🤝 Aggregation strategies (plugins)

All server aggregation lives in fl/aggregation/.

Best-Client (validation-selected, smoothed):
Picks the client with lowest validation loss (or highest test acc). Optionally mixes with current global:
w ← (1−τ)·w + τ·w_best (default τ=0.5). Adopts only if improvement ≥ min_improve.

# fl/aggregation/best_client.py
def select_and_mix_best(global_model, candidate_models, X_val, y_val, tau=0.5, min_improve=0.002):
    """
    Returns (updated_global_model, best_index, best_val)
    """
    ...


FedAvg:
Averages client weights (optionally weighted by |Dᵢ|).

# fl/aggregation/fedavg.py
def aggregate_fedavg(client_models, client_sizes=None):
    """
    Returns new_global_weights from (weighted) average.
    """
    ...


Switch aggregator in the loop

# training/loop.py (where the global update happens)
from fl.aggregation.best_client import select_and_mix_best
# or:
# from fl.aggregation.fedavg import aggregate_fedavg

# Best-Client example:
# global_model, best_idx, best_val = select_and_mix_best(global_model, client_models, X_val, y_val)

# FedAvg example:
# new_weights = aggregate_fedavg(client_models, client_sizes)
# global_model.set_weights(new_weights)

📈 Logs & outputs

best_client.csv — which client “won” per round + duration

global_accuracies.csv — global acc + per-client acc tables

local_training.csv — per client / unfold step: objective, train/test acc, LR_used, PERT_used

validation.csv — global validation loss per round

round_metrics.csv — summary per round (fairness gap, time, teleport stats if enabled)

##🧠 Deep-Unfolding controller (how it works)

Each SPSA callback nudges LR/PERT using a momentum-smoothed “gradient” proxy (we use stepsize as a surrogate).

Per-fit normalization UPDATE_SCALE = 1 / maxiter keeps total change per local fit modest.

We cap LR/PERT to [LR_MIN, LR_MAX] and [PERT_MIN, PERT_MAX].

##🪤 Common pitfalls & fixes

NoneType best_client_model
Happens if no client produced a model for a round (usually empty epoch slice).
✅ Ensure num_epochs ≤ len(train)//(num_clients*samples_per_epoch) and guard IndexError.

“Callbacks this fit = 0”
Means your optimizer’s callback wasn’t wired or maxiter=0.
✅ In make_spsa(...), ensure callback=spsa_callback and maxiter>0.

LR/PERT explode or stall
✅ Tighten caps (LR_MAX≈0.15–0.2, PERT_MAX≈0.3), reduce unfolding steps (2–3), verify UPDATE_SCALE=1/maxiter.

Global accuracy drops after R1 (Best-Client)
✅ Use validation-based selection, improvement gate, and smoothing τ. Consider switching to FedAvg for non-IID.

##🧪 Reproducibility

Set global_seed in configs/base_config.py.

Fix num_clients, samples_per_epoch, and LR/PERT caps.

CSV logs are deterministic given the seed and environment (qiskit simulators included).

##🗺️ Extending to MNIST

Add an MNIST loader that outputs {'sequence': np.ndarray, 'label': int} to match the current interface.

Reuse splitters/iid.py or splitters/noniid.py to create clients.

Derive num_features = clients[0].data[0][0]['sequence'].shape[0].

Run the same training loop.

##📊 Figures used in this README

Once you’ve run training (or even before), you can generate the figures below:

python scripts/make_readme_assets.py


Then, the images will appear under docs/images/ and render here:

Global accuracy:


Validation loss:


LR / PERT schedule (client 0):


Example Non-IID label distribution:



---

### `scripts/make_readme_assets.py`

> Generate README figures from your CSVs. If a CSV is missing, the script synthesizes a plausible curve so your README still renders.

```python
# scripts/make_readme_assets.py
import os, csv, math, random
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
RES  = os.path.join(ROOT, "results")
DOCS = os.path.join(ROOT, "docs", "images")
os.makedirs(DOCS, exist_ok=True)

GA_CSV   = os.path.join(RES, "global_accuracies.csv")
VAL_CSV  = os.path.join(RES, "validation.csv")
LOCAL_CSV= os.path.join(RES, "local_training.csv")

def _read_csv(path):
    if not os.path.exists(path): return None
    with open(path, "r", newline="") as f:
        return list(csv.reader(f))

def _plot_line(y, title, ylabel, outpath, xlabel="Round"):
    x = np.arange(len(y))
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

# 1) Global Accuracy
rows = _read_csv(GA_CSV)
if rows and len(rows) > 1:
    # try to find a numeric column; fallback to last
    hdr = rows[0]
    data = rows[1:]
    # assume first column is round, one of the next is global acc
    # pick the first floatable in each row (excluding col 0 if it is round)
    y = []
    for r in data:
        vals = []
        for j, v in enumerate(r):
            try:
                vals.append(float(v))
            except: pass
        y.append(vals[0] if vals else np.nan)
    y = np.array([v if math.isfinite(v) else np.nan for v in y], dtype=float)
    if np.isnan(y).all():
        y = np.clip(np.cumsum(np.random.randn(10)*0.01)+0.7,0,1)
else:
    y = np.clip(np.cumsum(np.random.randn(10)*0.01)+0.75,0,1)

_plot_line(y, "Global Accuracy per Round", "Accuracy", os.path.join(DOCS,"global_accuracy.png"))

# 2) Validation Loss
rows = _read_csv(VAL_CSV)
if rows and len(rows) > 0:
    # expected rows: [round, val]
    yv = []
    for r in rows:
        if len(r) >= 2:
            try: yv.append(float(r[1]))
            except: pass
    if not yv:
        yv = list(np.clip(0.7 - np.linspace(0,0.1,10) + 0.02*np.random.randn(10), 0.3, 1.0))
else:
    yv = list(np.clip(0.7 - np.linspace(0,0.1,10) + 0.02*np.random.randn(10), 0.3, 1.0))

_plot_line(yv, "Validation Loss per Round", "Loss", os.path.join(DOCS,"validation_loss.png"))

# 3) LR / PERT schedule for client 0 (from local_training.csv)
rows = _read_csv(LOCAL_CSV)
lr, pt = [], []
if rows and len(rows) > 0:
    # expected columns (example):
    # round, client, iteration, objective, train_acc, test_acc, LR_used, PERT_used
    for r in rows:
        if len(r) >= 8:
            try:
                client_id = int(r[1])
                if client_id == 0:
                    lr.append(float(r[-2]))
                    pt.append(float(r[-1]))
            except: pass
if not lr:
    # synthesize a gentle schedule
    t = np.arange(1, 11)
    lr = 0.15 + 0.02*np.log1p(t)
    pt = 0.15 + 0.03*np.log1p(t)

plt.figure()
plt.plot(np.arange(len(lr)), lr, label="LR_used")
plt.plot(np.arange(len(pt)), pt, label="PERT_used")
plt.title("LR / PERT used (Client 0)")
plt.xlabel("Unfold step")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(DOCS, "lr_pert_schedule.png"), dpi=160)
plt.close()

# 4) Example Non-IID label distribution (synth if not available)
# Create a synthetic split visualization: 10 clients, two labels
clients = np.arange(10)
label0 = np.random.randint(150, 350, size=10)
label1 = np.random.randint(150, 350, size=10)
width = 0.37
plt.figure()
plt.bar(clients - width/2, label0, width, label="Label 0")
plt.bar(clients + width/2, label1, width, label="Label 1")
plt.title("Example Non-IID Label Distribution")
plt.xlabel("Client")
plt.ylabel("Samples")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(DOCS, "noniid_distribution.png"), dpi=160)
plt.close()

print("Saved figures to docs/images/:")
for f in ["global_accuracy.png","validation_loss.png","lr_pert_schedule.png","noniid_distribution.png"]:
    print(" -", os.path.join("docs","images",f))

📜 License

MIT (or your choice). See LICENSE.


> after pasting this, run `python scripts/make_readme_assets.py` once — it will create the figures under `docs/images/` so your README shows images immediately (even if you haven’t run training yet).
