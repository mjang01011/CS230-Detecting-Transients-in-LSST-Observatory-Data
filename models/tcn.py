import torch
import torch.nn as nn
from pytorch_tcn import TCN

class LightCurveTCN(nn.Module):
    def __init__(self, input_size=1, num_classes=14, hidden_size=64, num_layers=2, kernel_size=3):
        """
        TCN for classification of light curve anomalies

        Args:
            input_size: number of features
            num_classes: output classes
            hidden_size: size of channels
            num_layers: number of residual block layers
            kernel_size: kernel size of convolutions
        """
        super(LightCurveTCN, self).__init__()
        channels = [hidden_size] * num_layers
        self.tcn = TCN(
            num_inputs=input_size,
            num_channels=channels,
            kernel_size=kernel_size,
            dilations=None,
            dropout=0,
            causal=False,  # For classification.
            use_norm='weight_norm',
            input_shape='NLC',  # input shape is [batch, sequence length, features]
            use_skip_connections=True,
            kernel_initializer='normal',
        )
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        out = self.tcn(x)
        # This step gets the last value from the sequence.
        out = out[:, -1, :]
        out = self.fc(out)
        return out
