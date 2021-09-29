import torch
from torch import nn

import torch.nn.init as init

class VentilatorNet(nn.Module):
    def __init__(self,
                 input_dim: int = 4,
                 lstm_dim: int = 256,
                 dense_dim: int = 256,
                 logit_dim: int = 256,
                 n_classes: int = 1,
                 initialize: bool = True
                 ) -> None:
        """
        Model class.

        Args:
            cfg: main config
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4 + input_dim, dense_dim // 2),
            nn.PReLU(),
            nn.Linear(dense_dim // 2, dense_dim),
            nn.PReLU()
        )
        self.r_emb = nn.Embedding(3, 2, padding_idx=0)
        self.c_emb = nn.Embedding(3, 2, padding_idx=0)

        self.lstm1 = nn.LSTM(dense_dim, lstm_dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.25)
        self.lstm2 = nn.LSTM(lstm_dim * 2, lstm_dim // 2, batch_first=True, bidirectional=True, num_layers=2, dropout=0.25)
        self.lstm3 = nn.LSTM(lstm_dim, lstm_dim // 4, batch_first=True, bidirectional=True, num_layers=2, dropout=0.25)
        self.gru = nn.GRU(lstm_dim, lstm_dim // 4, batch_first=True, bidirectional=True, num_layers=2, dropout=0.25)

        self.logits = nn.Sequential(
            nn.Linear(lstm_dim, logit_dim // 2),
            nn.PReLU(),
            nn.Linear(logit_dim // 2, logit_dim // 4),
            nn.PReLU(),
            nn.Linear(logit_dim // 4, n_classes),
        )
        if initialize:
            for n, m in self.named_modules():
                if isinstance(m, nn.LSTM):
                    for param in m.parameters():
                        if len(param.shape) >= 2:
                            nn.init.orthogonal_(param.data)
                        else:
                            nn.init.normal_(param.data)
                elif isinstance(m, nn.GRU):
                    for param in m.parameters():
                        if len(param.shape) >= 2:
                            init.orthogonal_(param.data)
                        else:
                            init.normal_(param.data)
                elif isinstance(m, (nn.Linear, nn.Embedding)):
                    m.weight.data.normal_(mean=0.0, std=1.0)
                    if isinstance(m, nn.Linear):
                        if m.bias is not None:
                            m.bias.data.zero_()
                elif isinstance(m, nn.LayerNorm):
                    if m.bias is not None:
                        m.bias.data.zero_()
                    m.weight.data.fill_(1.0)

    def forward(self, x):
        r_emb = self.r_emb(x['r']).view(-1, 80, 2)
        c_emb = self.c_emb(x['c']).view(-1, 80, 2)
        seq_x = torch.cat((r_emb, c_emb, x['input']), 2)
        features = self.mlp(seq_x)

        features, _ = self.lstm1(features)
        features, _ = self.lstm2(features)
        features_gru, _ = self.gru(features)
        features, _ = self.lstm3(features)
        pred = self.logits(torch.cat((features, features_gru), 2))
        return pred
