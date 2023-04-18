import torch.nn as nn


class CNNLayerNorm(nn.Module):
    """Layer Normalization"""
    
    def __init__(self, n_features):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=n_features)
        """About normalized_shape parameter of nn.LayerNorm:
        If a single integer is used, it is treated as a singleton list, and this module will normalize
        over the last dimension which is expected to be of that specific size.
        """

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 