import torch
import torch.nn as nn

class LightCurveRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=14):
        """
        RNN for classification of light curve transients

        Args:
            input_size: number of features
            hidden_size: size of RNN hidden state
            num_layers: number of RNN layers
            num_classes: number of output classes
        """
        super(LightCurveRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, hidden = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out