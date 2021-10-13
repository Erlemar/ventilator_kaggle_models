import torch
from torch import nn

from src.utils.technical_utils import load_obj


class VentilatorNet(nn.Module):
    def __init__(self,
                 input_dim: int = 4,
                 dense_dim: int = 128,
                 nhead: int = 8,
                 logit_dim: int = 256,
                 transformer_num_layers: int = 6,
                 transformer_dropout: float = 0.1,
                 transformer_activation: str = 'relu',
                 activation: str = 'torch.nn.ReLU',
                 n_classes: int = 1,
                 initialize: bool = True,
                 use_transformer_encoder: bool = True,
                 use_transformer: bool = True,
                 init_style: int = 3,
                 dim_feedforward: int = 3,
                 use_mlp: bool = True,
                 ) -> None:
        """
        Model class.

        Args:
            cfg: main config
        """
        super().__init__()
        self.use_mlp = use_mlp
        if self.use_mlp:

            self.mlp = nn.Sequential(
                nn.Linear(input_dim, dense_dim // 2),
                load_obj(activation)(),
                nn.Linear(dense_dim // 2, dense_dim),
                load_obj(activation)(),
            )
            input_dim = dense_dim

        self.use_transformer_encoder = use_transformer_encoder
        if use_transformer_encoder:
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead,
                                                            dim_feedforward=dim_feedforward,
                                                            dropout=transformer_dropout,
                                                            activation=transformer_activation,
                                                            batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=transformer_num_layers)

        self.use_transformer = use_transformer
        if self.use_transformer:
            self.transformer_model = nn.Transformer(d_model=input_dim, nhead=nhead,
                                                    num_encoder_layers=transformer_num_layers,
                                                    num_decoder_layers=transformer_num_layers,
                                                    dim_feedforward=dim_feedforward,
                                                    dropout=transformer_dropout,
                                                    activation=transformer_activation,
                                                    batch_first=True)
        linear_dim = input_dim
        if self.use_transformer_encoder and self.use_transformer:
            linear_dim *= 2
        self.linear1 = nn.Linear(linear_dim, logit_dim)
        self.act = nn.SELU()
        self.linear2 = nn.Linear(logit_dim, n_classes)

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
                    if isinstance(m, (nn.LSTM, nn.GRU)):
                        nn.init.xavier_normal_(m.weight_ih_l0)
                        nn.init.xavier_normal_(m.weight_hh_l0)
                        nn.init.xavier_normal_(m.weight_ih_l0_reverse)
                        nn.init.xavier_normal_(m.weight_hh_l0_reverse)

            elif init_style == 10:
                for name, p in self.named_parameters():
                    print(name)
                    if 'lstm' in name:
                        print('l')
                        if 'weight_ih' in name:
                            nn.init.xavier_uniform_(p.data)
                        elif 'weight_hh' in name:
                            nn.init.orthogonal_(p.data)
                        elif 'bias_ih' in name:
                            p.data.fill_(0)
                            # Set forget-gate bias to 1
                            n = p.size(0)
                            p.data[(n // 4):(n // 2)].fill_(1)
                        elif 'bias_hh' in name:
                            p.data.fill_(0)
                    elif 'linear' in name:
                        print('lin')
                        if 'weight' in name:
                            nn.init.xavier_uniform_(p.data)
                        elif 'bias' in name:
                            p.data.fill_(0)

            elif init_style == 11:
                print(f'{init_style}')
                for n, m in self.named_modules():
                    if isinstance(m, (nn.LSTM, nn.GRU)):
                        print(m)
                        nn.init.xavier_uniform_(m.weight_ih_l0)
                        nn.init.orthogonal_(m.weight_hh_l0)
                        nn.init.xavier_uniform_(m.weight_ih_l0_reverse)
                        nn.init.orthogonal_(m.weight_hh_l0_reverse)

    def forward(self, x):
        if self.use_mlp:
            features = self.mlp(x['input'])
        else:
            features = x['input']


        if self.use_transformer_encoder and not self.use_transformer:
            features = self.transformer_encoder(features)
        elif not self.use_transformer_encoder and self.use_transformer:
            features = self.transformer_model(features, features)
        elif self.use_transformer_encoder and self.use_transformer:
            features = torch.cat((self.transformer_encoder(features),
                                  self.transformer_model(features, features)),
                                 2)

        features = self.linear1(features)
        features = self.act(features)
        pred = self.linear2(features)
        return pred
