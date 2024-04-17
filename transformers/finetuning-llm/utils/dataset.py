from datasets import load_dataset

class Dataset:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.full_training_dataset = None
        self.training_dataset = None
    
    def load_dataset(self):
        return load_dataset(self.dataset_name)
    
    def select_training_dataset(self, dataset, num_samples):
        self.full_training_dataset = dataset["train"]
        self.training_dataset = self.full_training_dataset.shuffle(seed=42).select(range(num_samples))