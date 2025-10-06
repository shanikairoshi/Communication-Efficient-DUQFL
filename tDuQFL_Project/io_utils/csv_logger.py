# io_utils/csv_logger.py
from common.imports import *
import os
import csv


def _ensure_dir(path: str) -> None:
    """Create parent directory for 'path' if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

# -------------------- INITIALIZERS --------------------
def init_local_csv(csv_file, headers):
    with open(csv_file, mode='w', newline='') as file:
        csv.writer(file).writerow(headers)

def clear_csv_file(csv_file, headers):
    with open(csv_file, mode='w', newline='') as file:
        csv.writer(file).writerow(headers)

def save_results(csv_file, federated_round, client_id, iteration, obj_func_val, train_acc, test_acc, learning_rate, perturbation):
    with open(csv_file, mode='a', newline='') as file:
        csv.writer(file).writerow([federated_round, client_id, iteration, obj_func_val, train_acc, test_acc, learning_rate, perturbation])

def init_validation_csv(validation_csv_file: str):
    """
    Initialize the central validation-loss CSV (per round).
    """
    _ensure_dir(validation_csv_file)
    with open(validation_csv_file, mode='w', newline='') as f:
        csv.writer(f).writerow(["Round", "Central Validation Loss"])
def init_best_csv(best_client_csv_file):
    with open(best_client_csv_file, mode='w', newline='') as file:
        csv.writer(file).writerow(["Federated Round", "Client Number", "Round Duration (s)"])

def save_best_client_results(best_client_csv_file, federated_round, best_client_index, round_duration):
    with open(best_client_csv_file, mode='a', newline='') as file:
        csv.writer(file).writerow([federated_round, best_client_index, f"{round_duration:.4f}"])

def save_accuracies_to_csv(global_accuracies, clients_train_accuracies, clients_test_accuracies, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['Epoch', 'Global Accuracy']
        for i in range(len(clients_train_accuracies[0])):
            header.append(f'Client {i} Train Accuracy')
            header.append(f'Client {i} Test Accuracy')
        writer.writerow(header)
        for epoch in range(len(global_accuracies)):
            row = [epoch, global_accuracies[epoch]]
            for client_index in range(len(clients_train_accuracies[epoch])):
                row.append(clients_train_accuracies[epoch][client_index])
                row.append(clients_test_accuracies[epoch][client_index])
            writer.writerow(row)
