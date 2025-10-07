# Communication-Efficient-DUQFL
This is the official implementation for Communication-Efficient Deep Unfolded Quantum Federated Learning. This repository includes all the setup details, experimental results, and the code structure necessary to reproduce the purpose.

Deep-unfolding federated learning with learnable SPSA hyper-params (learning rate & perturbation), optional quantum teleportation perturbation, and pluggable aggregation strategies.

✨ Highlights

Deep-Unfolding SPSA: LR & perturbation are adapted during each local fit via a momentum-smoothed controller.

Trust-Region Caps: Safety bounds for LR/PERT to avoid divergence.

Federated Rounds with client carry-over of learned hyper-params.

Aggregation plugins: Best-Client (val-gated & smoothed) or FedAvg.

Metrics logging: global/client accuracies, validation loss, (optional) teleportation stats.

tDuQFL_Project/
├── configs/
│   └── base_config.py
├── data/
│   ├── preprocess_genome.py        # Genome dataset prep
│   ├── splitters.py                # IID splitter
│   └── noniid.py                   # Non-IID split (quantity + label skew)
├── fl/
│   ├── client.py                   # Client container
│   └── aggregation/
│       ├── best_client.py          # Val-gated & smoothed best-client
│       └── fedavg.py               # Standard FedAvg
├── io_utils/
│   └── csv_logger.py
├── ml/
│   ├── models.py                   # QNN init (qiskit circuits, etc.)
│   └── optimizers.py               # Learnable SPSA + deep-unfold controller
├── tele/
│   ├── noise.py
│   └── teleport.py
├── training/
│   ├── callbacks.py                # trackers for LR/PERT/objective
│   ├── loop.py                     # federated orchestration
│   └── metrics.py                  # round-level metrics logger
└── examples/
    ├── run_genome_iid.py           # minimal runnable examples
    └── run_genome_noniid.py
