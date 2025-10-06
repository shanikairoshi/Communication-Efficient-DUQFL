# data/preprocess_genome.py
from common.imports import *
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict

def load_and_prepare_dataset(word_size: int, global_seed: int):
    algorithm_globals.random_seed = global_seed

    test_set  = DemoHumanOrWorm(split='test',  version=0)
    train_set = DemoHumanOrWorm(split='train', version=0)
    data_set  = train_set  # as in your code

    word_combinations = defaultdict(int)
    iteration = 1
    for text, _ in data_set:
        for i in range(len(text)):
            w = text[i:i+word_size]
            if word_combinations.get(w) is None:
                word_combinations[w] = iteration
                iteration += 1

    np_data_set = []
    for i in range(len(data_set)):
        sequence, label = data_set[i]
        sequence = sequence.strip()
        words = [sequence[j:j+word_size] for j in range(0, len(sequence), word_size)]
        int_sequence = np.array([word_combinations[w] for w in words])
        np_data_set.append({'sequence': int_sequence, 'label': label})

    np.random.shuffle(np_data_set)
    sequences = np.vstack([item['sequence'] for item in np_data_set])
    scaler = MinMaxScaler()
    sequences_scaled = scaler.fit_transform(sequences)
    for i, item in enumerate(np_data_set):
        item['sequence'] = sequences_scaled[i]

    np_train_data = np_data_set[:5000]
    np_test_data  = np_data_set[-1000:]
    return np_train_data, np_test_data
