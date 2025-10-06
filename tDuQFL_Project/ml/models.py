# ml/models.py
from common.imports import *
from ml.optimizers import make_spsa

def initialize_model(num_features: int, initial_params, initial_learning_rate,initial_perturbation, optimizer):
    """
    Build the QNN classifier and ATTACH the provided optimizer.
    Dependency injection keeps optimizer state (learned LR/perturbation) under trainer control.
    """
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2)
    ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
    qc = feature_map.compose(ansatz)

    def parity(x: int) -> int:
        return "{:b}".format(x).count("1") % 2

    sampler_qnn = SamplerQNN(
        circuit=qc,
        interpret=parity,
        output_shape=2,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
    )
    opt = make_spsa(
    maxiter=25,
    initial_learning_rate=initial_learning_rate,
    initial_perturbation=initial_perturbation,
    lr_alpha=0.01,
)

    clf = NeuralNetworkClassifier(
        neural_network=sampler_qnn,
        optimizer=optimizer,          # injected (stateful)
        loss='squared_error',
        initial_point=initial_params,
    )
    return clf,opt
