# data/preprocess_mnist.py
from common.imports import *
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Option A: torchvision (recommended)
from torchvision import datasets, transforms
import torch

def load_and_prepare_dataset(
    n_features: int,
    digit_a: int = 3,
    digit_b: int = 8,
    global_seed: int = 42,
):
    """
    Create a binary MNIST dataset (digit_a vs digit_b), PCA-reduced to n_features,
    scaled to [0,1], and returned as a list of dicts: {'sequence': vec, 'label': 0/1}.
    """
    rng = np.random.default_rng(global_seed)
    torch.manual_seed(global_seed)

    # Load MNIST
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="./data", train=True,  download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    def to_xy(ds):
        X, y = [], []
        for img, lab in ds:
            if lab == digit_a or lab == digit_b:
                X.append(img.view(-1).numpy())                 # 784 vector
                y.append(0 if lab == digit_a else 1)           # binary label
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=int)

    Xtr, ytr = to_xy(train_ds)
    Xte, yte = to_xy(test_ds)

    # Scale → PCA → MinMax(0..1)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_z = scaler.fit_transform(Xtr)
    Xte_z = scaler.transform(Xte)

    pca = PCA(n_components=n_features, random_state=global_seed)
    Xtr_p = pca.fit_transform(Xtr_z)
    Xte_p = pca.transform(Xte_z)

    mm = MinMaxScaler()
    Xtr_p = mm.fit_transform(Xtr_p)
    Xte_p = mm.transform(Xte_p)

    # Package in your expected format
    np_train_data = [{'sequence': Xtr_p[i], 'label': int(ytr[i])} for i in range(len(ytr))]
    np_test_data  = [{'sequence': Xte_p[i], 'label': int(yte[i])} for i in range(len(yte))]

    return np_train_data, np_test_data
