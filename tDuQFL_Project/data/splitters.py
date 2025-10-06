# data/splitters.py
from fl.client import Client

def split_dataset_for_epochs(num_clients, num_epochs, train_data, test_data, samples_per_epoch):
    clients = []
    _ = len(train_data) // num_clients  # kept for parity with your code
    for i in range(num_clients):
        client_data_for_epochs = []
        for epoch in range(num_epochs):
            start_idx = (i * num_epochs * samples_per_epoch) + (epoch * samples_per_epoch)
            end_idx   = (i * num_epochs * samples_per_epoch) + ((epoch + 1) * samples_per_epoch)
            client_data_for_epochs.append(train_data[start_idx:end_idx])

        test_samples_per_client = len(test_data) // num_clients
        test_start_idx = i * test_samples_per_client
        test_end_idx   = (i + 1) * test_samples_per_client
        client_test_data = test_data[test_start_idx:test_end_idx]

        clients.append(Client(client_data_for_epochs, client_test_data))
    return clients
