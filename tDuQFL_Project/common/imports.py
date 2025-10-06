# --- Future & typing ---------------------------------------------------------
from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

# --- Standard library --------------------------------------------------------
import csv
import os
import sys
import time
from collections import Counter
from datetime import datetime
from functools import partial

# --- Third-party (general) ---------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

# --- Domain-specific: datasets ----------------------------------------------
from genomic_benchmarks.dataset_getters.pytorch_datasets import DemoHumanOrWorm

# --- Qiskit core -------------------------------------------------------------
import qiskit
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.compiler import transpile
from qiskit.visualization import plot_histogram

# --- Qiskit Aer (simulation & noise) -----------------------------------------
from qiskit_aer import Aer, AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# --- Qiskit primitives / ML / algorithms ------------------------------------
from qiskit.primitives import BackendSampler
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_algorithms.utils import algorithm_globals

# --- Optional: Google Colab integration --------------------------------------
# If running in Google Colab and you need Drive, this block will mount it.
try:
    # Comment out if not using Colab.
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive")
except Exception:
    pass

# --- Optional: environment diagnostics ---------------------------------------
print("Python:", sys.version)
print("Qiskit:", getattr(qiskit, "__version__", "unknown"))
print("qiskit_aer available?:", AerSimulator is not None)

# --- Optional: package setup notes -------------------------------------------
# In notebooks (e.g., Colab), install/upgrade Aer if needed:
# %pip install -U qiskit-aer
