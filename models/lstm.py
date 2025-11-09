import torch
import torch.nn as nn

class LightCurveLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        """
        LSTM for binary classification of light curve anomalies

        Args:
            input_size: number of features
            hidden_size: size of LSTM hidden state
            num_layers: number of LSTM layers
        """
        super(LightCurveLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, (hidden, cell) = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        return out
