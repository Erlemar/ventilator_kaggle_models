import torch
from torch import nn

from src.utils.technical_utils import load_obj


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
                 activation: str = 'nn.PReLU',
                 use_mlp: bool = True,
                 use_transformer: bool = True,
                 use_only_transformer_encoder: bool = True,
                 use_lstm: bool = True,
                 ) -> None:
        """
        Model class.

        Args:
            cfg: main config
        """
        super().__init__()
        self.use_mlp = use_mlp
        self.use_transformer = use_transformer
        self.use_only_transformer_encoder = use_only_transformer_encoder
        self.use_lstm = use_lstm
        if self.use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, dense_dim // 2),
                load_obj(activation)(),
                nn.Linear(dense_dim // 2, dense_dim),
                load_obj(activation)(),
            )

        if not self.use_mlp:
            d_model = input_dim
            dense_dim = input_dim

        if self.use_lstm:
            self.lstm1 = nn.LSTM(dense_dim, lstm_dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.25)
            self.lstm2 = nn.LSTM(lstm_dim * 2, lstm_dim // 2, batch_first=True, bidirectional=True, num_layers=2, dropout=0.25)

        if self.use_transformer:
            if not self.use_only_transformer_encoder:
                self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
                self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
                self.transformer_model = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers)
            else:
                self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
                self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        linear_dim = lstm_dim
        if not self.use_mlp and not self.use_transformer and not self.use_mlp and not self.use_lstm:
            linear_dim = input_dim
        if self.use_transformer:
            if self.use_only_transformer_encoder and self.use_lstm:
                linear_dim = lstm_dim + d_model
            elif not self.use_only_transformer_encoder and self.use_lstm:
                linear_dim = lstm_dim + d_model * 2
            elif self.use_only_transformer_encoder and not self.use_lstm:
                linear_dim = d_model
            elif not self.use_only_transformer_encoder and not self.use_lstm:
                linear_dim = d_model * 2

        self.logits = nn.Sequential(
            nn.Linear(linear_dim, logit_dim),
            load_obj(activation)(),
            nn.Linear(logit_dim, n_classes),
        )

    def forward(self, x):
        if self.use_mlp:
            features = self.mlp(x['input'])
        else:
            features = x['input']

        if self.use_transformer:
            fs = self.transformer_encoder(features)
            if not self.use_only_transformer_encoder:
                fs1 = self.transformer_model(features, features)

        if self.use_lstm:
            features, _ = self.lstm1(features)
            features, _ = self.lstm2(features)
        # print('features', features.shape)
        # print('fs', fs.shape)
        # print('fs1', fs1.shape)
        if self.use_transformer:
            if not self.use_only_transformer_encoder and self.use_lstm:
                seq = torch.cat((features, fs, fs1), 2)
            elif self.use_only_transformer_encoder and not self.use_lstm:
                seq = fs
            elif self.use_only_transformer_encoder and self.use_lstm:
                seq = torch.cat((features, fs), 2)
            elif not self.use_only_transformer_encoder and not self.use_lstm:
                seq = torch.cat((fs, fs1), 2)
        else:
            seq = features
        # print('seq', seq.shape)
        pred = self.logits(seq)
        return pred
