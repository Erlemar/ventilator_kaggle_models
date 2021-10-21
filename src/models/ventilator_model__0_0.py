import torch
from torch import nn
from torch.nn import init


class VentilatorNet(nn.Module):
    def __init__(self,
                 input_dim: int = 4,
                 lstm_dim: int = 256,
                 lstm_layers: int = 4,
                 logit_dim: int = 256,
                 num_layers: int = 256,
                 n_classes: int = 1,
                 initialize: bool = True,
                 init_style: int = 1,
                 single_lstm: bool = True,
                 ) -> None:
        """
        Model class.

        Args:
            cfg: main config
        """
        super().__init__()

        self.lstm1 = nn.LSTM(input_dim, 768, batch_first=True, bidirectional=True, dropout=0.0, num_layers=1)
        self.lstm2 = nn.LSTM(768 * 2, 512, batch_first=True, bidirectional=True, dropout=0.0, num_layers=1)
        self.lstm3 = nn.LSTM(512 * 2, 384, batch_first=True, bidirectional=True, dropout=0.0, num_layers=1)
        self.lstm4 = nn.LSTM(384 * 2, 256, batch_first=True, bidirectional=True, dropout=0.0, num_layers=1)
        self.lstm5 = nn.LSTM(256 * 2, 128, batch_first=True, bidirectional=True, dropout=0.0, num_layers=1)

        self.gru1 = nn.GRU(512 * 2, 384, batch_first=True, bidirectional=True, dropout=0.0, num_layers=1)
        self.gru2 = nn.GRU(384 * 2, 256, batch_first=True, bidirectional=True, dropout=0.0, num_layers=1)
        self.gru3 = nn.GRU(256 * 2, 128, batch_first=True, bidirectional=True, dropout=0.0, num_layers=1)
        self.gru4 = nn.GRU(128 * 2, 64, batch_first=True, bidirectional=True, dropout=0.0, num_layers=1)

        self.bn1 = nn.BatchNorm1d(80)
        self.bn2 = nn.BatchNorm1d(80)
        self.bn3 = nn.BatchNorm1d(80)

        self.lstm0 = nn.LSTM(input_dim, lstm_dim, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.lstms = nn.ModuleList([nn.LSTM(lstm_dim * 4 // (2 ** (i + 1)),
                                            lstm_dim // (2 ** (i + 1)),
                                            batch_first=True, bidirectional=True, num_layers=num_layers)
                                    for i in range(lstm_layers - 1)])

        # self.linear1 = nn.Linear(4 * lstm_dim // (2 ** lstm_layers), logit_dim)
        self.linear1 = nn.Linear(1920, logit_dim)
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

        x1, _ = self.lstm1(x['input'])
        x2, _ = self.lstm2(x1)
        x3, _ = self.lstm3(x2)
        x4, _ = self.lstm4(x3)
        x5, _ = self.lstm5(x4)

        z2, _ = self.gru1(x2)
        z31 = x3 * z2

        z31 = self.bn1(z31)
        z3, _ = self.gru2(z31)

        z41 = x4 * z3
        z41 = self.bn2(z41)
        z4, _ = self.gru3(z41)

        z51 = x5 * z4
        z51 = self.bn3(z51)
        z5, _ = self.gru4(z51)

        features = torch.cat([x5, z2, z3, z4, z5], axis=2)
        features = self.linear1(features)
        features = self.act(features)
        pred = self.linear2(features)
        return pred
