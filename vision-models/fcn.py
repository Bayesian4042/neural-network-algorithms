# let's build neural network using pytorch to classify images of handwritten digits from MNIST dataset

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

df = pd.read_csv("mnist_train.csv", header=None)
print(df.head())

# simple neural network -> FCN
# input: 28x28 image -> 784 pixels
# output: 10 classes (0-9)
# hidden layer: 200 neurons

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.ReLU(),
            nn.LayerNorm(200),
            nn.Linear(200, 10),
            nn.ReLU()
        )

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    
    def forward(self, inputs):
        return self.model(inputs)

    def train(self, inputs, targets):
        self.optimizer.zero_grad()
        outputs = self.forward(inputs)
        loss = self.loss(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss