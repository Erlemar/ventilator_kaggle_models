Tried adding conv from heng code - didn't help




ideas:
* analyse model predictions
* feature selection
* 10 folds
* group by over RC
* read code for molecules again
* google more papers
* use only u_out == 0
* classification 950 classes
* wavenet
* postprocessing 950 unique values
* dual regression head (separate prediction for pressure in and out). HENG
* postprocess Chris: https://www.kaggle.com/cdeotte/ensemble-folds-with-median-0-153/notebook

Better Nearest Neighbor / Clustering with Dynamic Time Warping
https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/279331

Ensembling With Cross Validation
https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/279284
```python
lin_reg = RidgeCV(alphas=np.logspace(-3,10, 20))
lin_reg.fit(X, y)
pred = lin_reg.predict(X)
for sub in zip([sub1, sub2, sub3, sub4], lin_reg.coef_):
    submission.pressure += sub[0].pressure * sub[1]
    
pressure_sorted = np.sort(data['pressure'].unique())
PRESSURE_MIN = pressure_sorted[0]
PRESSURE_MAX = pressure_sorted[-1]
PRESSURE_STEP = pressure_sorted[1] - pressure_sorted[0]

def post_process(pressure):
    pressure = np.round((pressure - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP + PRESSURE_MIN
    pressure = np.clip(pressure, PRESSURE_MIN, PRESSURE_MAX)
    return pressure

```

architecture experiments:
* 1 lstm with 4-6 num_layers
* MLP to higher dimension for transformer
* mlp, transformer and other things
* only transformer
* transformer encoder and decoder?
* gru?
* batchnorm1d after lstm/gru
* bigger logit layer
* skipping layers


```python
def dnn_model():
    
    x_input = Input(shape=(train.shape[-2:]))
    
    x1 = Bidirectional(LSTM(units=768, return_sequences=True))(x_input)
    x2 = Bidirectional(LSTM(units=512, return_sequences=True))(x1)
    x3 = Bidirectional(LSTM(units=256, return_sequences=True))(x2)
    x4 = Bidirectional(LSTM(units=128, return_sequences=True))(x3)
    
    z2 = Bidirectional(GRU(units=256, return_sequences=True))(x2)
    z3 = Bidirectional(GRU(units=128, return_sequences=True))(Add()([x3, z2]))
    z4 = Bidirectional(GRU(units=64, return_sequences=True))(Add()([x4, z3]))
    
    x = Concatenate(axis=2)([x4, z2, z3, z4])
    
    x = Dense(units=128, activation='selu')(x)
    
    x_output = Dense(units=1)(x)

    model = Model(inputs=x_input, outputs=x_output, 
                  name='DNN_Model')
    return model
```


Ventilator Train classification
https://www.kaggle.com/takamichitoda/ventilator-train-classification/notebook
```python
    USE_LAG = 4
    #CATE_FEATURES = ['R_cate', 'C_cate', 'RC_dot', 'RC_sum']
    CONT_FEATURES = ['u_in', 'u_out', 'time_step'] + ['u_in_cumsum', 'u_in_cummean', 'area', 'cross', 'cross2'] + ['R_cate', 'C_cate']
    LAG_FEATURES = ['breath_time']
    LAG_FEATURES += [f'u_in_lag_{i}' for i in range(1, USE_LAG+1)]
    #LAG_FEATURES += [f'u_in_lag_{i}_back' for i in range(1, USE_LAG+1)]
    LAG_FEATURES += [f'u_in_time{i}' for i in range(1, USE_LAG+1)]
    #LAG_FEATURES += [f'u_in_time{i}_back' for i in range(1, USE_LAG+1)]
    LAG_FEATURES += [f'u_out_lag_{i}' for i in range(1, USE_LAG+1)]
    #LAG_FEATURES += [f'u_out_lag_{i}_back' for i in range(1, USE_LAG+1)]
    #ALL_FEATURES = CATE_FEATURES + CONT_FEATURES + LAG_FEATURES
    ALL_FEATURES = CONT_FEATURES + LAG_FEATURES

    df['time_delta'] = df.groupby('breath_id')['time_step'].diff().fillna(0)
    df['delta'] = df['time_delta'] * df['u_in']
    df['area'] = df.groupby('breath_id')['delta'].cumsum()

    df['cross']= df['u_in']*df['u_out']
    df['cross2']= df['time_step']*df['u_out']
    
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    df['one'] = 1
    df['count'] = (df['one']).groupby(df['breath_id']).cumsum()
    df['u_in_cummean'] =df['u_in_cumsum'] / df['count']
    
    df = df.drop(['count','one'], axis=1)
    
    for lag in range(1, config.USE_LAG+1):
        df[f'breath_id_lag{lag}']=df['breath_id'].shift(lag).fillna(0)
        df[f'breath_id_lag{lag}same']=np.select([df[f'breath_id_lag{lag}']==df['breath_id']], [1], 0)

        # u_in 
        df[f'u_in_lag_{lag}'] = df['u_in'].shift(lag).fillna(0) * df[f'breath_id_lag{lag}same']
        #df[f'u_in_lag_{lag}_back'] = df['u_in'].shift(-lag).fillna(0) * df[f'breath_id_lag{lag}same']
        df[f'u_in_time{lag}'] = df['u_in'] - df[f'u_in_lag_{lag}']
        #df[f'u_in_time{lag}_back'] = df['u_in'] - df[f'u_in_lag_{lag}_back']
        df[f'u_out_lag_{lag}'] = df['u_out'].shift(lag).fillna(0) * df[f'breath_id_lag{lag}same']
        #df[f'u_out_lag_{lag}_back'] = df['u_out'].shift(-lag).fillna(0) * df[f'breath_id_lag{lag}same']

    # breath_time
    df['time_step_lag'] = df['time_step'].shift(1).fillna(0) * df[f'breath_id_lag{lag}same']
    df['breath_time'] = df['time_step'] - df['time_step_lag']

    drop_columns = ['time_step_lag']
    drop_columns += [f'breath_id_lag{i}' for i in range(1, config.USE_LAG+1)]
    drop_columns += [f'breath_id_lag{i}same' for i in range(1, config.USE_LAG+1)]
    df = df.drop(drop_columns, axis=1)

    # fill na by zero
    df = df.fillna(0)
    
    
    c_dic = {10: 0, 20: 1, 50:2}
    r_dic = {5: 0, 20: 1, 50:2}
    rc_sum_dic = {v: i for i, v in enumerate([15, 25, 30, 40, 55, 60, 70, 100])}
    rc_dot_dic = {v: i for i, v in enumerate([50, 100, 200, 250, 400, 500, 2500, 1000])}    
    
    def add_category_features(df):
        df['C_cate'] = df['C'].map(c_dic)
        df['R_cate'] = df['R'].map(r_dic)
        df['RC_sum'] = (df['R'] + df['C']).map(rc_sum_dic)
        df['RC_dot'] = (df['R'] * df['C']).map(rc_dot_dic)
        return df
    
    norm_features = config.CONT_FEATURES + config.LAG_FEATURES
    def norm_scale(train_df, test_df):
        scaler = RobustScaler()
        all_u_in = np.vstack([train_df[norm_features].values, test_df[norm_features].values])
        scaler.fit(all_u_in)
        train_df[norm_features] = scaler.transform(train_df[norm_features].values)
        test_df[norm_features] = scaler.transform(test_df[norm_features].values)
        return train_df, test_df
    

    self.seq_emb = nn.Sequential(
        #nn.Linear(12+len(config.CONT_FEATURES)+len(config.LAG_FEATURES), config.EMBED_SIZE),
        nn.Linear(len(config.CONT_FEATURES)+len(config.LAG_FEATURES), config.EMBED_SIZE),
        nn.LayerNorm(config.EMBED_SIZE),
    )
    
    self.lstm = nn.LSTM(config.EMBED_SIZE, config.HIDDEN_SIZE, batch_first=True, bidirectional=True, dropout=0.0, num_layers=4)

    self.head = nn.Sequential(
        nn.Linear(config.HIDDEN_SIZE * 2, config.HIDDEN_SIZE * 2),
        nn.LayerNorm(config.HIDDEN_SIZE * 2),
        nn.ReLU(),
        nn.Linear(config.HIDDEN_SIZE * 2, 950),
    )



    unique_pressures = train_df["pressure"].unique()
    sorted_pressures = np.sort(unique_pressures)
    total_pressures_len = len(sorted_pressures)

    def find_nearest(prediction):
        insert_idx = np.searchsorted(sorted_pressures, prediction)
        if insert_idx == total_pressures_len:
            # If the predicted value is bigger than the highest pressure in the train dataset,
            # return the max value.
            return sorted_pressures[-1]
        elif insert_idx == 0:
            # Same control but for the lower bound.
            return sorted_pressures[0]
        lower_val = sorted_pressures[insert_idx - 1]
        upper_val = sorted_pressures[insert_idx]
        return lower_val if abs(lower_val - prediction) < abs(upper_val - prediction) else upper_val
    
    
    sub_df["pressure"] = sub_df["pressure"].apply(find_nearest)
    
    # https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/278362
    def loss_fn(y_pred, y_true):
        criterion = nn.CrossEntropyLoss()
    
        loss = criterion(y_pred.reshape(-1, 950), y_true.reshape(-1, 950))
    
        for lag, w in [(1, 0.4), (2, 0.2), (3, 0.1), (4, 0.1)]:
            # negative lag loss
            # if target < 0, target = 0
            neg_lag_target = F.relu(y_true.reshape(-1) - lag)
            neg_lag_target = neg_lag_target.long()
            neg_lag_loss = criterion(y_pred.reshape(-1, 950), neg_lag_target)
    
            # positive lag loss
            # if target > 949, target = 949
            pos_lag_target = 949 - F.relu((949 - (y_true.reshape(-1) + lag)))
            pos_lag_target = pos_lag_target.long()
            pos_lag_loss = criterion(y_pred.reshape(-1, 950), pos_lag_target)
    
            loss += (neg_lag_loss + pos_lag_loss) * w
    
        return loss

```
