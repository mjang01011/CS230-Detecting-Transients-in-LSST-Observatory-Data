import torch
import torch.nn as nn

class LightCurveGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=14):
        """
        GRU for classification of light curve transients

        Args:
            input_size: number of features
            hidden_size: size of GRU hidden state
            num_layers: number of GRU layers
            num_classes: number of output classes
        """
        super(LightCurveGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            output: predictions of shape (batch_size, num_classes)
        """
        out, hidden = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
