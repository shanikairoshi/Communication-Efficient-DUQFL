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

