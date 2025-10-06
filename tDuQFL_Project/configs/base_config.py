# configs/base_config.py
from datetime import datetime

# experiment setup
num_epochs                    = 10
max_train_iterations          = 100
samples_per_epoch             = 50
word_size                     = 40

use_teleportation             = False
use_noise                     = False
noise_preset      = "med"      # add this
shots_used        = 256        # and this

num_clients                   = 10
num_federated_layers          = 10
num_deep_unfolding_iterations = 5

initial_learning_rate         = 0.15
meta_learning_rate            = 1e-4
initial_perturbation          = 0.15
momentum                      = 0.95
gradient_moving_avg           = 0

dataset_name                  = "Genome"
split_type                    = "IID" #"noniid", "non-iid", "non_iid"
drive_root                    = "/content/drive/MyDrive/Teleportation/tDuQFL_Project"

global_seed                   = 42
