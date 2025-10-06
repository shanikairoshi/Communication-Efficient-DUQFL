# fl/client.py
class Client:
    def __init__(self, data, test_data):
        self.data = data
        self.test_data = test_data
        self.models = []
        self.train_scores = []
        self.test_scores = []
        self.primary_model = None
