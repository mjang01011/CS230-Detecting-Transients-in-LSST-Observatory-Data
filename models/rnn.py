import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class LightCurveRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        """
        RNN for binary classification of light curve anomalies

        Args:
            input_size: number of features
            hidden_size: size of RNN hidden state
            num_layers: number of RNN layers
        """
        super(LightCurveRNN, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True) #batch_first=True --> (batch size, sequence_length, input_size) default is false
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid() # for now we'll use sigmoid for binary classification (anomaly or not)

    def forward(self, x):
        out, hidden = self.rnn(x)
        out = out[:, -1, :] # last timestamp (-1)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out