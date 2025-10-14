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

## big-picture overview

You are running quantum federated learning (QFL) over multiple rounds.  
Each round has four phases.

### 1) local training (per client)
- Each client trains its QNN (with or without deep unfolding).
- The client returns an updated local model.

### 2) uplink (client → server)
Depending on the uplink mode, each client transmits one of:
- **classical_full:** full model weights
- **seeded_sparse:** compressed sparse delta (seeded)
- **quantum:** teleportation-simulated quantum payload

### 3) server aggregation
Depending on the aggregation strategy:
- **fedavg:** average all client parameters (or deltas)
- **best:** pick a winner (submodes: `all` or `winner_only`), optionally \( \tau \)-mix

### 4) downlink (server → all clients)
The server broadcasts the global model using one of:
- **classical_full:** full model weights
- **seeded:** seeded compression
- **quantum:** teleportation-simulated quantum downlink

### end-of-round logging
Record the following for analysis and reproducibility:
- Global accuracy, per-client accuracies, validation loss, and wall-clock time
- Communication bytes up/down (including quantum accounting of **2 classical bits per parameter**)
- Communication modes and hyperparameters (for baseline vs quantum comparisons)


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
```

## 🧾 Description of directories

| Directory | Description |
|:-----------|:-------------|
| **configs/** | Global and dataset-specific configuration files containing training and communication parameters |
| **common/** | Shared imports and global constants used across modules |
| **data/** | Dataset preprocessing and client splitting (IID and Non-IID modes) |
| **fl/** | Federated learning components including client logic and aggregation methods (FedAvg, Best-client) |
| **ml/** | Quantum model definitions and optimizers (SPSA, deep unfolding controllers) |
| **training/** | Core training loop, metric logging, and callback handling for each round |
| **tele/** | Quantum-inspired teleportation modules and Aer noise simulators |
| **io_utils/** | CSV logging utilities and result-naming helpers |
| **scripts/** | Entry-point scripts to run predefined experiments and generate plots |
| **docs/** | Documentation and auto-generated figures for the README |
| **results/** | Output CSV files generated during training (accuracy, validation loss, etc.) |
| **artifacts/** | Saved model weights and checkpoints for later evaluation |
