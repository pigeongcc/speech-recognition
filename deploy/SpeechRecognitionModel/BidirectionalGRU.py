import torch.nn as nn
import torch.nn.functional as F

class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_type, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        if rnn_type == "GRU":
            self.rnn_cell = nn.GRU
        elif rnn_type == "LSTM":
            self.rnn_cell = nn.LSTM

        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.BiGRU = self.rnn_cell(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.layer_norm(x)
        out = F.gelu(out)
        out, _ = self.BiGRU(out)
        out = self.dropout(out)
        return out