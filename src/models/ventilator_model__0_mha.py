from torch import nn
from torch.nn import init


class VentilatorNet(nn.Module):
    def __init__(self,
                 input_dim: int = 4,
                 lstm_dim: int = 256,
                 lstm_layers: int = 4,
                 logit_dim: int = 256,
                 num_layers: int = 256,
                 dropout: float = 0.0,
                 n_classes: int = 1,
                 num_heads: int = 2,
                 initialize: bool = True,
                 init_style: int = 1,
                 single_atn: bool = True,
                 att_key: str = 'feature',
                 ) -> None:
        """
        Model class.

        Args:
            cfg: main config
        """
        super().__init__()
        self.att_key = att_key
        self.lstm0 = nn.LSTM(input_dim, lstm_dim, batch_first=True, bidirectional=True, num_layers=num_layers, dropout=dropout)
        kdim = None if self.att_key == 'feature' else input_dim
        lstms = []
        for i in range(lstm_layers - 1):
            lstms.append(nn.LSTM(lstm_dim * 4 // (2 ** (i + 1)), lstm_dim // (2 ** (i + 1)), batch_first=True, bidirectional=True, num_layers=num_layers, dropout=dropout))
            if not single_atn:
                lstms.append(nn.MultiheadAttention(lstm_dim * 2 // (2 ** (i + 1)), num_heads=num_heads, batch_first=True, kdim=kdim, vdim=kdim, dropout=dropout))

        self.lstms = nn.ModuleList(lstms)

        self.atn = nn.MultiheadAttention(4 * lstm_dim // (2 ** lstm_layers), num_heads=num_heads, batch_first=True, kdim=kdim, vdim=kdim, dropout=dropout)

        self.linear1 = nn.Linear(4 * lstm_dim // (2 ** lstm_layers), logit_dim)
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
        features, _ = self.lstm0(x['input'])
        for layer in self.lstms:
            if layer._get_name() == 'LSTM':
                features, _ = layer(features)
            elif layer._get_name() == 'MultiheadAttention':
                if self.att_key == 'feature':
                    features, _ = layer(features, features, features)
                else:
                    features, _ = layer(features, x['input'], x['input'])
        if self.att_key == 'feature':
            features, _ = self.atn(features, features, features)
        else:
            features, _ = self.atn(features, x['input'], x['input'])
        features = self.linear1(features)
        features = self.act(features)
        pred = self.linear2(features)
        return pred
