import torch.nn as nn
import torch.nn.functional as F

from .CNNLayerNorm import CNNLayerNorm


class ResidualCNN(nn.Module):
    """ Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm """
    
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_features):
        super(ResidualCNN, self).__init__()

        self.layer_norm1 = CNNLayerNorm(n_features)
        self.dropout1 = nn.Dropout(dropout)
        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)

        self.layer_norm2 = CNNLayerNorm(n_features)
        self.dropout2 = nn.Dropout(dropout)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        out = self.layer_norm1(x)
        out = F.gelu(out)
        out = self.dropout1(out)
        out = self.cnn1(out)
        out = self.layer_norm2(out)
        out = F.gelu(out)
        out = self.dropout2(out)
        out = self.cnn2(out)
        out += residual
        return out # (batch, channel, feature, time)