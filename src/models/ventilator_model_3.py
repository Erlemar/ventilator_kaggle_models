import torch
from torch import nn


class VentilatorNet(nn.Module):
    def __init__(self,
                 input_dim: int = 4,
                 lstm_dim: int = 256,
                 dense_dim: int = 256,
                 logit_dim: int = 256,
                 n_classes: int = 1,
                 ) -> None:
        """
        Model class.

        Args:
            cfg: main config
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4 + input_dim, dense_dim // 2),
            nn.ReLU(),
            nn.Linear(dense_dim // 2, dense_dim),
            nn.ReLU()
        )
        self.r_emb = nn.Embedding(3, 2, padding_idx=0)
        self.c_emb = nn.Embedding(3, 2, padding_idx=0)

        self.lstm1 = nn.LSTM(dense_dim, lstm_dim, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(lstm_dim * 2, lstm_dim // 2, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(lstm_dim, lstm_dim // 4, batch_first=True, bidirectional=True)

        self.logits = nn.Sequential(
            nn.Linear(lstm_dim // 2, logit_dim),
            nn.ReLU(),
            nn.Linear(logit_dim, n_classes),
        )

    def forward(self, x):
        r_emb = self.r_emb(x['r']).view(-1, 80, 2)
        c_emb = self.c_emb(x['c']).view(-1, 80, 2)
        seq_x = torch.cat((r_emb, c_emb, x['input']), 2)
        features = self.mlp(seq_x)

        features, _ = self.lstm1(features)
        features, _ = self.lstm2(features)
        features, _ = self.lstm3(features)
        pred = self.logits(features)
        return pred
