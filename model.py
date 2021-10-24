import torch.nn as nn
import torch
import torch.nn.functional as F

train_x = [500]
train_y = [10]

class LanguageModel(nn.Module):
    def __init__(self):
        super(LanguageModel, self).__init__()

        self.fc1 = nn.Linear(in_features=train_x[0], out_features=128)  # FIGURE INPUT SHAPE, add len to train_x[0]
        self.fc2 = nn.Linear(self.fc1.out_features, out_features=64)
        self.fc3 = nn.Linear(self.fc2.out_features, out_features=train_y[0])
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        pass
