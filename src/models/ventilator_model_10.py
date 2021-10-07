from torch import nn
from torch.nn import init

from src.utils.technical_utils import load_obj


class VentilatorNet(nn.Module):
    def __init__(self,
                 input_dim: int = 4,
                 lstm_dim: int = 256,
                 dense_dim: int = 256,
                 logit_dim: int = 256,
                 num_layers: int = 256,
                 n_classes: int = 1,
                 initialize: bool = True,
                 init_style: int = 1,
                 use_mlp: bool = True,
                 simpler_mlp: bool = True,
                 activation: str = 'torch.nn.PReLU',
                 activation2: str = 'torch.nn.PReLU',
                 use_layer_norm: bool = False,
                 layer_norm_logits: bool = False,
                 layer_norm_style: int = 1,
                 lstm_dropout: float = 0.0,
                 dropout: float = 0.0,
                 ) -> None:
        """
        Model class.

        Args:
            cfg: main config
        """
        super().__init__()
        self.use_mlp = use_mlp
        if self.use_mlp:
            if not use_layer_norm:
                if not simpler_mlp:
                    self.mlp = nn.Sequential(
                        nn.Linear(input_dim, dense_dim // 2),
                        load_obj(activation)(),
                        nn.Linear(dense_dim // 2, dense_dim),
                        load_obj(activation)(),
                        nn.Dropout(dropout),
                    )
                elif simpler_mlp and layer_norm_style != 5:
                    self.mlp = nn.Sequential(
                        nn.Linear(input_dim, dense_dim),
                        load_obj(activation)(),
                        nn.Dropout(dropout),
                    )
            else:
                if layer_norm_style == 1:
                    self.mlp = nn.Sequential(
                        nn.Linear(input_dim, dense_dim // 2),
                        load_obj(activation)(),
                        nn.Linear(dense_dim // 2, dense_dim),
                        load_obj(activation)(),
                        nn.LayerNorm(dense_dim),
                        nn.Dropout(dropout),
                    )
                elif layer_norm_style == 2:
                    self.mlp = nn.Sequential(
                        nn.Linear(input_dim, dense_dim // 2),
                        load_obj(activation)(),
                        nn.LayerNorm(dense_dim),
                        nn.Linear(dense_dim // 2, dense_dim),
                        load_obj(activation)(),
                        nn.Dropout(dropout),
                    )
                elif layer_norm_style == 3:
                    self.mlp = nn.Sequential(
                        nn.BatchNorm1d(input_dim),
                        nn.Linear(input_dim, dense_dim // 2),
                        load_obj(activation)(),
                        nn.LayerNorm(dense_dim),
                        nn.Linear(dense_dim // 2, dense_dim),
                        load_obj(activation)(),
                        nn.Dropout(dropout),
                    )
                elif layer_norm_style == 4:
                    self.mlp = nn.Sequential(
                        nn.Linear(input_dim, dense_dim // 2),
                        load_obj(activation)(),
                        nn.LayerNorm(dense_dim),
                        nn.Linear(dense_dim // 2, dense_dim),
                        load_obj(activation)(),
                        nn.LayerNorm(dense_dim),
                        nn.Dropout(dropout),
                    )
                elif simpler_mlp and layer_norm_style == 5:
                    self.mlp = nn.Sequential(
                        nn.Linear(input_dim, dense_dim),
                        nn.LayerNorm(dense_dim),
                        load_obj(activation)(),
                        nn.Dropout(dropout),
                    )
        else:
            dense_dim = input_dim

        self.lstm0 = nn.LSTM(dense_dim, lstm_dim // 2, batch_first=True, bidirectional=True, num_layers=num_layers, dropout=lstm_dropout)
        self.lstm1 = nn.LSTM(lstm_dim, lstm_dim // 4, batch_first=True, bidirectional=True, num_layers=num_layers, dropout=lstm_dropout)
        self.lstm2 = nn.LSTM(lstm_dim // 2, lstm_dim // 8, batch_first=True, bidirectional=True, num_layers=num_layers, dropout=lstm_dropout)
        self.lstm3 = nn.LSTM(lstm_dim // 4, lstm_dim // 16, batch_first=True, bidirectional=True, num_layers=num_layers, dropout=lstm_dropout)

        if not layer_norm_logits:
            self.logits = nn.Sequential(
                nn.Linear(lstm_dim // 8, logit_dim),
                load_obj(activation2)(),
                nn.Linear(logit_dim, n_classes),
            )
        else:
            self.logits = nn.Sequential(
                nn.Linear(lstm_dim // 8, logit_dim),
                nn.LayerNorm(logit_dim),
                load_obj(activation2)(),
                nn.Linear(logit_dim, n_classes),
            )

        if initialize:
            if init_style == 0:
                print(f'{init_style}')
                for n, m in self.named_modules():
                    if isinstance(m, nn.LSTM):
                        for param in m.parameters():
                            if len(param.shape) >= 2:
                                nn.init.orthogonal_(param.data)
                            else:
                                nn.init.normal_(param.data)

            if init_style == 1:
                print(f'{init_style}')
                for n, m in self.named_modules():
                    if isinstance(m, nn.LSTM):
                        for param in m.parameters():
                            if len(param.shape) >= 2:
                                nn.init.orthogonal_(param.data)
                            else:
                                nn.init.normal_(param.data)
                    elif isinstance(m, (nn.Linear, nn.Embedding)):
                        m.weight.data.normal_(mean=0.0, std=1.0)
                        if isinstance(m, nn.Linear):
                            if m.bias is not None:
                                m.bias.data.zero_()

            elif init_style == 2:
                print(f'{init_style}')
                for n, m in self.named_modules():
                    if isinstance(m, nn.LSTM):
                        for param in m.parameters():
                            if len(param.shape) >= 2:
                                nn.init.xavier_uniform_(param.data)
                            else:
                                nn.init.normal_(param.data)
                    elif isinstance(m, (nn.Linear, nn.Embedding)):
                        m.weight.data.normal_(mean=0.0, std=1.0)
                        if isinstance(m, nn.Linear):
                            if m.bias is not None:
                                m.bias.data.zero_()

            elif init_style == 3:
                print(f'{init_style}')
                for n, m in self.named_modules():
                    if isinstance(m, nn.LSTM):
                        for name, param in m.named_parameters():
                            if 'weight_ih' in name:
                                nn.init.xavier_uniform_(param.data)
                            elif 'weight_hh' in name:
                                nn.init.orthogonal_(param.data)
                            elif 'bias' in name:
                                param.data.fill_(0)

                    elif isinstance(m, (nn.Linear, nn.Embedding)):
                        m.weight.data.normal_(mean=0.0, std=1.0)
                        if isinstance(m, nn.Linear):
                            if m.bias is not None:
                                m.bias.data.zero_()

            elif init_style == 4:
                print(f'{init_style}')
                for n, m in self.named_modules():
                    if isinstance(m, nn.LSTM):
                        for name, param in m.named_parameters():
                            if 'weight_ih' in name:
                                nn.init.kaiming_normal_(param.data)
                            elif 'weight_hh' in name:
                                nn.init.orthogonal_(param.data)
                            elif 'bias' in name:
                                nn.init.constant_(param, 0.0)

                    elif isinstance(m, (nn.Linear, nn.Embedding)):
                        m.weight.data.normal_(mean=0.0, std=1.0)
                        if isinstance(m, nn.Linear):
                            if m.bias is not None:
                                m.bias.data.zero_()

            elif init_style == 5:
                print(f'{init_style}')
                for n, m in self.named_modules():
                    if isinstance(m, nn.LSTM):
                        for name, param in m.named_parameters():
                            if 'weight_ih' in name:
                                nn.init.kaiming_normal_(param.data)
                            elif 'weight_hh' in name:
                                nn.init.orthogonal_(param.data)
                            elif 'bias' in name:
                                nn.init.constant_(param, 0.0)

                    elif isinstance(m, (nn.Linear, nn.Embedding)):
                        nn.init.xavier_normal_(m.weight.data)
                        if isinstance(m, nn.Linear):
                            if m.bias is not None:
                                m.bias.data.zero_()

            elif init_style == 6:
                print(f'{init_style}')
                for n, m in self.named_modules():
                    if isinstance(m, nn.LSTM):
                        for name, param in m.named_parameters():
                            if 'weight_ih' in name:
                                nn.init.kaiming_normal_(param.data)
                            elif 'weight_hh' in name:
                                nn.init.orthogonal_(param.data)
                            elif 'bias' in name:
                                nn.init.constant_(param, 0.0)

                    elif isinstance(m, (nn.Linear, nn.Embedding)):
                        nn.init.xavier_uniform_(m.weight.data)
                        if isinstance(m, nn.Linear):
                            if m.bias is not None:
                                m.bias.data.zero_()

            elif init_style == 7:
                print(f'{init_style}')
                for n, m in self.named_modules():
                    if isinstance(m, nn.LSTM):
                        for name, param in m.named_parameters():
                            if 'weight_ih' in name:
                                nn.init.kaiming_normal_(param.data)
                            elif 'weight_hh' in name:
                                nn.init.orthogonal_(param.data)
                            elif 'bias' in name:
                                nn.init.constant_(param, 0.0)

                    elif isinstance(m, (nn.Linear, nn.Embedding)):
                        nn.init.xavier_uniform_(m.weight.data)
                        if isinstance(m, nn.Linear):
                            if m.bias is not None:
                                nn.init.constant_(m.bias.data, 0)

            elif init_style == 8:
                print(f'{init_style}')
                for n, m in self.named_modules():
                    if isinstance(m, nn.Conv2d or nn.Linear or nn.GRU or nn.LSTM):
                        nn.init.xavier_normal_(m.weight)
                        m.bias.data.zero_()
                    elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()

            elif init_style == 9:
                print(f'{init_style}')
                for n, m in self.named_modules():
                    if isinstance(m, nn.LSTM):
                        nn.init.xavier_normal_(m.weight_ih_l0)
                        nn.init.xavier_normal_(m.weight_hh_l0)
                        nn.init.xavier_normal_(m.weight_ih_l0_reverse)
                        nn.init.xavier_normal_(m.weight_hh_l0_reverse)

    def forward(self, x):

        if self.use_mlp:
            features = self.mlp(x['input'])
        else:
            features = x['input']
        # print("x", x['input'].shape)
        features, _ = self.lstm0(features)
        features, _ = self.lstm1(features)
        features, _ = self.lstm2(features)
        features, _ = self.lstm3(features)
        # print('features', features)
        pred = self.logits(features)
        return pred
