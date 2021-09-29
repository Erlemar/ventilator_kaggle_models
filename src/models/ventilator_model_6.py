import torch
from torch import nn
import torch.nn.init as init

class VentilatorNet(nn.Module):
    def __init__(self,
                 input_dim: int = 4,
                 lstm_dim: int = 256,
                 dense_dim: int = 256,
                 logit_dim: int = 256,
                 d_model: int = 512,
                 nhead: int = 6,
                 num_layers: int = 6,
                 n_classes: int = 1,
                 norm: bool = False,
                 initialize: bool = False,
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
            nn.PReLU(),
        )
        self.r_emb = nn.Embedding(3, 2, padding_idx=0)
        self.c_emb = nn.Embedding(3, 2, padding_idx=0)

        self.lstm1 = nn.LSTM(dense_dim, lstm_dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.25)
        self.lstm2 = nn.LSTM(lstm_dim * 2, lstm_dim // 2, batch_first=True, bidirectional=True, num_layers=2, dropout=0.25)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        if norm:
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(lstm_dim))
        else:
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.transformer_model = nn.Transformer(nhead=nhead, num_encoder_layers=num_layers)

        self.logits = nn.Sequential(
            nn.Linear(lstm_dim * 3, logit_dim),
            nn.PReLU(),
            nn.Linear(logit_dim, n_classes),
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
        fs = self.transformer_encoder(features)
        fs1 = self.transformer_model(features, features)

        features, _ = self.lstm1(features)
        features, _ = self.lstm2(features)
        seq = torch.cat((features, fs, fs1), 2)
        pred = self.logits(seq)
        return pred
