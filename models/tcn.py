import torch
import torch.nn as nn
from pytorch_tcn import TCN

class LightCurveTCN(nn.Module):
    def __init__(self, input_size=1, num_classes=14, max_length=200, hidden_size=64, num_layers=2):
        """
        TCN for classification of light curve anomalies

        Args:
            input_size: number of features
            hidden_size: size of LSTM hidden state
            num_layers: number of LSTM layers
        """
        super(LightCurveTCN, self).__init__()
        # channels = [8, 16, 32, 64, 128, 256]
        # channels = [64, 128]
        channels = [8, 16, 32, 64]
        # channels = [16, 16, 16, 16, 16]
        # channels = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
        self.tcn = TCN(
            num_inputs=input_size,
            num_channels=channels,
            kernel_size=4,
            dilations=None,
            dropout=0.1,
            causal=False,  # For classification.
            use_norm='batch_norm',
            input_shape='NLC',  # input shape is [batch, sequence length, features]
            # use_skip_connections=True,
        )
        # self.flatten = nn.Flatten()  # flatten -> (batch_size x last channel * seq length)
        # fc_input = input_size * channels[-1] * max_length
        # self.fc = nn.Linear(fc_input, num_classes)
        self.fc = nn.Linear(channels[-1], num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.tcn(x)
        # This step gets the last value from the sequence.
        out = out[:, -1, :]
        # out = self.flatten(out)
        out = self.fc(out)
        out = self.softmax(out)
        return out
