from torch import nn


class VentilatorNet(nn.Module):
    def __init__(self,
                 input_dim: int = 4,
                 lstm_dim: int = 256,
                 dense_dim: int = 256,
                 logit_dim: int = 256,
                 num_layers: int = 256,
                 n_classes: int = 1,
                 ) -> None:
        """
        Model class.

        Args:
            cfg: main config
        """
        super().__init__()
        # self.mlp = nn.Sequential(
        #     nn.Linear(input_dim, dense_dim // 2),
        #     nn.ReLU(),
        #     nn.Linear(dense_dim // 2, dense_dim),
        #     nn.ReLU(),
        # )
        self.lstm0 = nn.LSTM(dense_dim, lstm_dim // 2, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.lstm1 = nn.LSTM(lstm_dim, (lstm_dim - 100) // 2, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.lstm2 = nn.LSTM(lstm_dim - 100, (lstm_dim - 200) // 2, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.lstm3 = nn.LSTM(lstm_dim - 200, (lstm_dim - 300) // 2, batch_first=True, bidirectional=True, num_layers=num_layers)

        self.logits = nn.Sequential(
            nn.Linear(lstm_dim - 300, logit_dim),
            nn.SELU(),
            nn.Linear(logit_dim, n_classes),
        )

    def forward(self, x):
        # features = self.mlp(x['input'])
        # print("x", x['input'].shape)
        features, _ = self.lstm0(x['input'])
        features, _ = self.lstm1(features)
        features, _ = self.lstm2(features)
        features, _ = self.lstm3(features)
        # print('features', features)
        pred = self.logits(features)
        return pred
