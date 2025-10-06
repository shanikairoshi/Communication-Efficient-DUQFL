# io_utils/naming.py
from datetime import datetime
import os

def fmt(x: float) -> str:
    return str(x).replace('.', 'p')

def build_param_str(num_clients, num_layers, num_du, lr, pert):
    return (f"clients{num_clients}_layers{num_layers}_du{num_du}"
            f"_lr{fmt(lr)}_pert{fmt(pert)}")

def stamp_now():
    ts = datetime.now()
    return ts.strftime("%d_%m_%Y_%H%M%S"), ts.strftime("%d_%m_%Y")

def flags(use_teleportation: bool, use_noise: bool):
    return ("Teleport" if use_teleportation else "NoTeleport",
            "WithNoise" if use_noise else "NoNoise")

def make_filenames(drive_root, dataset_name, split_type, date_str, teleport_pl, noise_pl, param_str):
    best_client_csv_file = os.path.join(
        drive_root, f"tDuQFL_{dataset_name}_{split_type}_Best_Client_{date_str}_{teleport_pl}_{noise_pl}_{param_str}.csv"
    )
    global_csv_file = os.path.join(
        drive_root, f"tDuQFL_{dataset_name}_{split_type}_Global_{date_str}_{teleport_pl}_{noise_pl}_{param_str}.csv"
    )
    local_csv_file = os.path.join(
        drive_root, f"tDuQFL_{dataset_name}_{split_type}_Local_{date_str}_{teleport_pl}_{noise_pl}_{param_str}.csv"
    )
    validation_csv_file = os.path.join(
        drive_root, f"tDuQFL_{dataset_name}_{split_type}_Validation_Loss_{date_str}_{teleport_pl}_{noise_pl}_{param_str}.csv"
    )
    return best_client_csv_file, global_csv_file, local_csv_file, validation_csv_file
