import numpy as np
import pandas as pd
import math
import time
import pickle
import argparse
import sklearn.preprocessing
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold

debug = False

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_features(df):
    df = df.copy()
    df['area'] = df['time_step'] * df['u_in']
    df['area'] = df.groupby('breath_id')['area'].cumsum()
    
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    
    df['u_in_lag1'] = df.groupby('breath_id')['u_in'].shift(1)
    df['u_out_lag1'] = df.groupby('breath_id')['u_out'].shift(1)
    df['u_in_lag_back1'] = df.groupby('breath_id')['u_in'].shift(-1)
    df['u_out_lag_back1'] = df.groupby('breath_id')['u_out'].shift(-1)
    df['u_in_lag2'] = df.groupby('breath_id')['u_in'].shift(2)
    df['u_out_lag2'] = df.groupby('breath_id')['u_out'].shift(2)
    df['u_in_lag_back2'] = df.groupby('breath_id')['u_in'].shift(-2)
    df['u_out_lag_back2'] = df.groupby('breath_id')['u_out'].shift(-2)
    df['u_in_lag3'] = df.groupby('breath_id')['u_in'].shift(3)
    df['u_out_lag3'] = df.groupby('breath_id')['u_out'].shift(3)
    df['u_in_lag_back3'] = df.groupby('breath_id')['u_in'].shift(-3)
    df['u_out_lag_back3'] = df.groupby('breath_id')['u_out'].shift(-3)
    df['u_in_lag4'] = df.groupby('breath_id')['u_in'].shift(4)
    df['u_out_lag4'] = df.groupby('breath_id')['u_out'].shift(4)
    df['u_in_lag_back4'] = df.groupby('breath_id')['u_in'].shift(-4)
    df['u_out_lag_back4'] = df.groupby('breath_id')['u_out'].shift(-4)
    df = df.fillna(0)
    
    df['breath_id__u_in__max'] = df.groupby(['breath_id'])['u_in'].transform('max')
    df['breath_id__u_out__max'] = df.groupby(['breath_id'])['u_out'].transform('max')
    
    df['u_in_diff1'] = df['u_in'] - df['u_in_lag1']
    df['u_out_diff1'] = df['u_out'] - df['u_out_lag1']
    df['u_in_diff2'] = df['u_in'] - df['u_in_lag2']
    df['u_out_diff2'] = df['u_out'] - df['u_out_lag2']
    
    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']
    
    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']
    
    df['u_in_diff3'] = df['u_in'] - df['u_in_lag3']
    df['u_out_diff3'] = df['u_out'] - df['u_out_lag3']
    df['u_in_diff4'] = df['u_in'] - df['u_in_lag4']
    df['u_out_diff4'] = df['u_out'] - df['u_out_lag4']
    df['cross']= df['u_in']*df['u_out']
    df['cross2']= df['time_step']*df['u_out']
    
    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    df['R__C'] = df["R"].astype(str) + '__' + df["C"].astype(str)
    df = pd.get_dummies(df)
    df.drop(['id', 'breath_id'], axis=1, inplace=True)
    if 'pressure' in df.columns:
        df.drop('pressure', axis=1, inplace=True)

    return df


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y, w):
        if y is None:
            y = np.zeros(len(X), dtype=np.float32)

        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.w = w.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.w[i]

n = 100*1024 if debug else None

di = '/workspace/data/ventilator_pressure_prediction/'
train = pd.read_csv(di + 'train.csv', nrows=n)
test = pd.read_csv(di + 'test.csv', nrows=n)
submit = pd.read_csv(di + 'sample_submission.csv', nrows=n)

features = create_features(train)
rs = sklearn.preprocessing.RobustScaler()
features = rs.fit_transform(features)  # => np.ndarray

X_all = features.reshape(-1, 80, features.shape[-1])
y_all = train.pressure.values.reshape(-1, 80)
w_all = 1 - train.u_out.values.reshape(-1, 80)  # weights for the score, but not used in this notebook

input_size = X_all.shape[2]

print(len(X_all))

class Model(nn.Module):
    def __init__(self, input_size):
        hidden = [400, 300, 200, 100]
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden[0],
                             batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(2 * hidden[0], hidden[1],
                             batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(2 * hidden[1], hidden[2],
                             batch_first=True, bidirectional=True)
        self.lstm4 = nn.LSTM(2 * hidden[2], hidden[3],
                             batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(2 * hidden[3], 50)
        self.selu = torch.nn.SELU()
        self.fc2 = nn.Linear(50, 1)
        self._reinitialize()

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if 'lstm' in name:
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
            elif 'fc' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x, _ = self.lstm4(x)
        x = self.fc1(x)
        x = self.selu(x)
        x = self.fc2(x)

        return x
		
criterion = torch.nn.L1Loss()

def evaluate(model, loader_val):
    tb = time.time()
    was_training = model.training
    model.eval()

    loss_sum = 0
    score_sum = 0
    n_sum = 0
    y_pred_all = []

    for ibatch, (x, y, w) in enumerate(loader_val):
        n = y.size(0)
        x = x.to(device)
        y = y.to(device)
        w = w.to(device)

        with torch.no_grad():
            y_pred = model(x).squeeze()

        loss = criterion(y_pred, y)

        n_sum += n
        loss_sum += n*loss.item()
        
        y_pred_all.append(y_pred.cpu().detach().numpy())

    loss_val = loss_sum / n_sum

    model.train(was_training)

    d = {'loss': loss_val,
         'time': time.time() - tb,
         'y_pred': np.concatenate(y_pred_all, axis=0)}

    return d
	
nfold = 5
kfold = KFold(n_splits=nfold, shuffle=True, random_state=2021)
epochs = 2 if debug else 300
lr = 1e-3
batch_size = 1024
max_grad_norm = 1000
log = {}

for ifold, (idx_train, idx_val) in enumerate(kfold.split(X_all)):
    print('Fold %d' % ifold)
    tb = time.time()
    model = Model(input_size)
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    X_train = X_all[idx_train]
    y_train = y_all[idx_train]
    w_train = w_all[idx_train]
    X_val = X_all[idx_val]
    y_val = y_all[idx_val]
    w_val = w_all[idx_val]

    dataset_train = Dataset(X_train, y_train, w_train)
    dataset_val = Dataset(X_val, y_val, w_val)
    loader_train = torch.utils.data.DataLoader(dataset_train, shuffle=True,
                         batch_size=batch_size, drop_last=True)
    loader_val = torch.utils.data.DataLoader(dataset_val, shuffle=False,
                         batch_size=batch_size, drop_last=False)

    losses_train = []
    losses_val = []
    lrs = []
    time_val = 0
    best_score = np.inf
   
    print('epoch loss_train loss_val lr time')
    for iepoch in range(epochs):
        loss_train = 0
        n_sum = 0
        
        for ibatch, (x, y, w) in enumerate(loader_train):
            n = y.size(0)
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            y_pred = model(x).squeeze()

            loss = criterion(y_pred, y)
            loss_train += n*loss.item()
            n_sum += n

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

        val = evaluate(model, loader_val)
        loss_val = val['loss']
        time_val += val['time']

        losses_train.append(loss_train / n_sum)
        losses_val.append(val['loss'])
        lrs.append(optimizer.param_groups[0]['lr'])

        print('%3d %9.6f %9.6f %7.3e %7.1f %6.1f' %
              (iepoch + 1,
               losses_train[-1], losses_val[-1], 
               lrs[-1], time.time() - tb, time_val))

        scheduler.step(losses_val[-1])


    ofilename = 'model%d.pth' % ifold
    torch.save(model.state_dict(), ofilename)
    print(ofilename, 'written')

    log['fold%d' % ifold] = {
        'loss_train': np.array(losses_train),
        'loss_val': np.array(losses_val),
        'learning_rate': np.array(lrs),
        'y_pred': val['y_pred'],
        'idx': idx_val
    }
	
print('Fold loss_train loss_val best loss_val')
for ifold in range(nfold):
    d = log['fold%d' % ifold]
    print('%4d %9.6f %9.6f %9.6f' % (ifold, d['loss_train'][-1], d['loss_val'][-1], np.min(d['loss_val'])))

features = create_features(test)
features = rs.transform(features)

X_test = features.reshape(-1, 80, features.shape[-1])
y_test = np.zeros(len(features)).reshape(-1, 80)
w_test = 1 - test.u_out.values.reshape(-1, 80)

dataset_test = Dataset(X_test, y_test, w_test)
loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)

y_pred_folds = np.zeros((len(test), nfold), dtype=np.float32)
for ifold in range(nfold):
    model = Model(input_size)
    model.to(device)
    model.load_state_dict(torch.load('model%d.pth' % ifold, map_location=device))
    model.eval()
    
    y_preds = []
    for x, y, _ in loader_test:
        x = x.to(device)
        with torch.no_grad():
            y_pred = model(x).squeeze()

        y_preds.append(y_pred.cpu().numpy())
    
    y_preds = np.concatenate(y_preds, axis=0)
    y_pred_folds[:, ifold] = y_preds.flatten()

submit.pressure = np.mean(y_pred_folds, axis=1)
submit.to_csv('submission_from_notebook.csv', index=False)
print('submission.csv written')

