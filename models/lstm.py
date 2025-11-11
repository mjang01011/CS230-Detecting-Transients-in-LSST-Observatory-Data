import torch
import torch.nn as nn

class LightCurveLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=14):
        """
        LSTM for classification of light curve transients

        Args:
            input_size: number of features
            hidden_size: size of LSTM hidden state
            num_layers: number of LSTM layers
            num_classes: number of output classes
        """
        super(LightCurveLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, (hidden, cell) = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
