import torch
from torch import nn
from torch.nn import init
# from einops.layers.torch import Rearrange

class VentilatorNet(nn.Module):
    def __init__(self,
                 input_dim: int = 4,
                 lstm_dim: int = 256,
                 lstm_layers: int = 4,
                 logit_dim: int = 256,
                 num_layers: int = 256,
                 conv_out_channels: int = 32,
                 n_classes: int = 1,
                 initialize: bool = True,
                 double_conv: bool = True,
                 init_style: int = 1,
                 conv_style: int = 1,
                 ) -> None:
        """
        Model class.

        Args:
            cfg: main config
        """
        super().__init__()

        self.r_emb = nn.Embedding(4, 2) #, padding_idx=0)
        self.c_emb = nn.Embedding(4, 2) #, padding_idx=0)
        # self.seq_embed = nn.Sequential(
        #     # Rearrange('b l d -> b d l'),
        #     nn.Conv1d(7, 32, kernel_size=5, padding=2, stride=1),
        #     # Rearrange('b d l -> b l d'),
        #     nn.LayerNorm(32),
        #     nn.SiLU(),
        #     nn.Dropout(0.),
        # )
        if conv_style:
            self.conv = nn.Conv1d(7, conv_out_channels, kernel_size=5, padding=2, stride=1)
        elif conv_style == 2:
            self.conv = nn.Conv1d(7, conv_out_channels, kernel_size=3, padding=1, stride=1)
        self.double_conv = double_conv
        if self.double_conv:
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv1d(conv_out_channels, conv_out_channels * 2, kernel_size=5, padding=2, stride=1)
            conv_out_channels *= 2
        self.layer_norm = nn.LayerNorm(conv_out_channels)
        self.act = nn.SiLU()

        self.lstm0 = nn.LSTM(input_dim + conv_out_channels, lstm_dim, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.lstms = nn.ModuleList([nn.LSTM(lstm_dim * 4 // (2 ** (i + 1)),
                                            lstm_dim // (2 ** (i + 1)),
                                            batch_first=True, bidirectional=True, num_layers=num_layers)
                                    for i in range(lstm_layers - 1)])

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
        # print('r', x['r'].shape)
        # print('c', x['c'].shape)
        r_emb = self.r_emb(x['r'].long())#.view(-1, 80, 2)
        c_emb = self.c_emb(x['c'].long())#.view(-1, 80, 2)
        # print('r_emb', r_emb.shape)
        # print('c_emb', c_emb.shape)
        # print(x['feats'].shape)
        seq_x = torch.cat((r_emb, c_emb, x['feats']), 2)
        # seq_x = self.seq_embed(seq_x)
        # seq_x = self.conv(seq_x.transpose(2, 1)).transpose(1, 2)
        if self.double_conv:
            seq_x = self.conv2(self.relu(self.conv(seq_x.transpose(2, 1)))).transpose(1, 2)
        else:
            seq_x = self.conv(seq_x.transpose(2, 1)).transpose(1, 2)

        seq_x = self.act(self.layer_norm(seq_x))

        # print('seq_x', seq_x.shape)
        # print(x['input'].shape)
        features = torch.cat((seq_x, x['input']), 2)

        features, _ = self.lstm0(features)
        for lstm in self.lstms:
            features, _ = lstm(features)
        features = self.linear1(features)
        features = self.act(features)
        pred = self.linear2(features)
        return pred
