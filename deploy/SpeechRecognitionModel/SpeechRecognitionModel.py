import torch.nn as nn

from .ResidualCNN import ResidualCNN
from .BidirectionalGRU import BidirectionalGRU

class SpeechRecognitionModel(nn.Module):
    """Speech Recognition Model Inspired by DeepSpeech 2"""

    def __init__(self, rnn_type, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_features, stride=2, dropout=0.1):
        super(SpeechRecognitionModel, self).__init__()
        n_features = n_features // 2

        # TODO: purpose of this conv layer
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3 // 2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_features=n_features) 
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_features*32, rnn_dim)
        """TODO: как я понял, у нас число фичей rnn_dim берётся одиночное для первой GRU, т.к. к ней не перетекает скрытое состояние.
        Каждая следующая GRU получает 2*rnn_dim фичей, т.к. к самим фичам конкатенируется скрытое состояние такой же размерности (hidden_size=rnn_dim)
        """
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_type=rnn_type, 
                             rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x