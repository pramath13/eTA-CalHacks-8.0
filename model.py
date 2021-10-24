import torch.nn as nn

train_x = [223]
train_y = [23]

class LanguageModel(nn.Module):
    def __init__(self):
        super(LanguageModel, self).__init__()

        self.fc1 = nn.Linear(in_features=train_x[0], out_features=128)
        self.fc2 = nn.Linear(self.fc1.out_features, out_features=64)
        self.fc3 = nn.Linear(self.fc2.out_features, out_features=train_y[0])
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        print("SHAPE", x.shape)
        x = self.relu(self.fc1(x))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

