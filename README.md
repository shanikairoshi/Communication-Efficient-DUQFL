# Communication-Efficient-DUQFL
# tDuQFL — deep-unfolded quantum federated learning with communication modes
This is the official implementation for Communication-Efficient Deep Unfolded Quantum Federated Learning. This repository includes all the setup details, experimental results, and the code structure necessary to reproduce the purpose.

Deep-unfolding federated learning with learnable SPSA hyper-params (learning rate & perturbation), optional quantum teleportation perturbation, and pluggable aggregation strategies.
Quantum federated learning (QFL) framework supporting deep unfolding (DU), aggregation choices (FedAvg / Best-client), and three communication modes per direction (uplink/downlink): classical_full, seeded (compressed), and quantum (teleportation-simulated). Includes MNIST (binary) and Breast-Lesions datasets, full logging of accuracy, validation loss, and communication bytes.

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![qiskit](https://img.shields.io/badge/qiskit-yes-5A3FD9)
![status](https://img.shields.io/badge/status-research--prototype-green)

Quantum federated learning (QFL) framework supporting deep unfolding (DU), aggregation choices
(FedAvg / Best-client), and three communication modes per direction (uplink/downlink):
`classical_full`, `seeded` (compressed), and `quantum` (teleportation-simulated).
Includes MNIST (binary) and Breast-Lesions datasets, with full logging of accuracy,
validation loss, and communication bytes.

## key features

### quantum models
- QNN over RealAmplitudes ansatz  
- SPSA optimizer for gradient-free updates

### deep unfolding
- Per-round meta-updates to learning rate \( \eta \) and perturbation \( \delta \)
- Toggle DU on or off for ablation analysis

### aggregation
- **fedavg** – standard weighted averaging  
- **best-client** – validation-gated select-and-mix strategy

### communication modes (per direction)
- **classical_full** – send full parameters (baseline)  
- **seeded / seeded_sparse** – deterministic sparse masks and quantized deltas  
- **quantum** – teleportation-simulated channel; accounts for two classical bits per qubit

### metrics and logging
- Global and client accuracies, validation loss, DU traces  
- Fidelity proxy when quantum mode is used  
- Communication bytes per round (down/up and cumulative)

### reproducible runs
- Unified configuration via `base_config.py`
- Consistent seeds and data splits for repeatability

## repo structure



---
python -m venv .venv
source .venv/bin/activate      # windows: .venv\Scripts\activate
pip install -r requirements.txt

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

##datasets

MNIST (binary)
Digits 
digit
𝑎
digit
a
	​

 vs 
digit
𝑏
digit
b
	​

 (default 3 vs 8)
File: data/preprocess_mnist.py → load_and_prepare_dataset(...)

Breast-lesions (USG clinical)
CSV path in preprocess_genome.py → cleans, encodes, PCA → binary labels
{benign = 0, malignant = 1}

configure

All settings are defined in configs/base_config.py.

# Dataset
dataset_name        = "MNIST"
n_features          = 8
digit_a, digit_b    = 3, 8
csv_path_genome     = "/path/to/BrEaST-Lesions-USG-Clinical.csv"

# Split / clients
split_type          = "NonIID"
num_clients         = 5
num_federated_layers= 10
num_du              = 5

# Optimizer seeds
initial_lr          = 0.14
initial_pert        = 0.14
gamma               = 0.5
use_deep_unfolding  = True

# Aggregation & selection
aggregation         = "best"
select_upload       = "winner_only"

# Communication modes
uplink_mode         = "quantum"
downlink_mode       = "quantum"

# Seeded compression knobs
down_seed_base      = 1234
down_scale          = 1e-3
down_mask_ratio     = 0.10
up_bits             = 8
up_k_ratio          = 0.01

# Teleportation / Aer
use_tele_backend    = True
noise_preset        = "med"
shots_used          = 256

# Output root
drive_root          = "./outputs"
random_seed         = 42
