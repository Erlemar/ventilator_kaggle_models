1. targets:
train['pressure_diff']=train.groupby(['breath_id'])['pressure'].diff().fillna(0)
train['pressure_diff2']=train.groupby(['breath_id'])['pressure_diff'].shift(-1).fillna(0)


df['pressure_diff1'] = df['pressure'].diff(periods=1)
df['pressure_diff2'] = df['pressure'].diff(periods=2)
df['pressure_diff3'] = df['pressure'].diff(periods=3)
df['pressure_diff4'] = df['pressure'].diff(periods=4)
df.loc[df['step']<1, 'pressure_diff1'] = 0
df.loc[df['step']<2, 'pressure_diff2'] = 0
df.loc[df['step']<3, 'pressure_diff3'] = 0
df.loc[df['step']<4, 'pressure_diff4'] = 0    


2. AdamW, lr 3e-4, epoch 280 with CosineAnnealingWarmRestarts

3. 
u_in and np.log1p(u_in)
raw_features, u_in_cumsum, u_in_log, diff1-4 of [time_step, u_in, u_in_log]
df['inhaled_air'] = df['time_diff']*df['u_in']

4. batch_size 128





====================

classification task

multi-target: pressure-diff, pressure-cumsum, its derivative and its integral.

StratifiedKFold using R&C

MixUp by third team

features:
* raw
* made
* learned

models:
* lstm
* hybrid
* transformer
At least one team (Patrick & Wong, 11th position) reported the use of a Conformer. Another team (CR7 is the GOAT) used a Conformer-like solution.

ReZero is All You Need

The Upstage team (3rd position) made a very interesting use of 3 data augmentations simultaneously: Mask R or C, Shuffle input and target values within a short window and MixUp. They claim this last one in particular considerably improved their performance. To use MixUp the used one-hot embeddings instead of categorical embeddings.

Cosine annealing
Cosine annealing with warm restarts
ReduceLROnPlateau

solutions:
1
https://www.kaggle.com/shujun717/1-solution-lstm-cnn-transformer-1-fold
https://github.com/Shujun-He/Google-Brain-Ventilator
https://github.com/whoknowsB/google-brain-ventilator-pressure-prediction


2

3
https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/285330
Architecture: Conv1d + Stacked LSTM
Optimizer: AdamW
Scheduler: Cosine

Original: u_in, u_out, R(one-hot), C(one-hot)
Engineered: u_in_min, u_in_diff, inhaled_air, time_diff

Masking: Randomly mask the R or C
Shuffling: Randomly shuffle our sequences
Mixup: Select two sequences and mix up them

MAE of pressure
Difference of pressure

```python
for name, p in self.named_parameters():
    if 'lstm' in name:
        if 'weight_ih' in name:
            nn.init.xavier_normal_(p.data)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(p.data)
        elif 'bias_ih' in name:
            p.data.fill_(0)
            # Set forget-gate bias to 1
            n = p.size(0)
            p.data[(n // 4):(n // 2)].fill_(1)
        elif 'bias_hh' in name:
            p.data.fill_(0)

    if 'conv' in name:
        if 'weight' in name:
            nn.init.xavier_normal_(p.data)
        elif 'bias' in name:
            p.data.fill_(0)

    elif 'reg' in name:
        if 'weight' in name:
            nn.init.xavier_normal_(p.data)
        elif 'bias' in name:
            p.data.fill_(0)
```

`df['inhaled_air'] = df['time_diff']*df['u_in']`

```python
one_sample = get_sample(idx)

# Shuffling sequence
if random.random() < .2:
    ws = np.random.choice([2, 4, 5])
    num = max_seq_len // ws

    idx = np.arange(0, max_seq_len)
    for i in range(num):
        np.random.shuffle(idx[i * ws:(i + 1) * ws])
        one_sample = one_sample[idx]
```

```python
one_sample = get_sample(idx)

# Mixup two sequences
if random.random() < 0.4:
    # Get new index
    idx = np.random.randint(len(indices_by_breath_id))
    sampled_sample = get_sample(idx)
    mix_cols = ['R', 'C', 'numeric_features', 'target']
    k = 0.5
    for col in mix_cols:
        one_sample[col] = (one_sample[col]*k + (1-k)*sampled_sample[col])

```


```python
df['step'] = list(range(80))*df['breath_id'].nunique()

df['pressure_diff1'] = df['pressure'].diff(periods=1)
df['pressure_diff2'] = df['pressure'].diff(periods=2)
df['pressure_diff3'] = df['pressure'].diff(periods=3)
df['pressure_diff4'] = df['pressure'].diff(periods=4)
df.loc[df['step']<1, 'pressure_diff1'] = 0
df.loc[df['step']<2, 'pressure_diff2'] = 0
df.loc[df['step']<3, 'pressure_diff3'] = 0
df.loc[df['step']<4, 'pressure_diff4'] = 0    
```


```python
if 'target' in batch:
    # pressure, diff1, diff2, diff3, diff4
    loss = F.l1_loss(pred, batch['target'], reduction='none')
    res['loss'] = loss[batch['u_out']==0].mean()
    with torch.no_grad():
        res['mae'] = loss[:, :, 0][batch['u_out']==0].mean()
```

13
Chris. keras
https://www.kaggle.com/cdeotte/tensorflow-transformer-0-112?scriptVersionId=79039122

If you take the top public notebooks and train with batch size 32 instead of 512 and change the learning rate to 2.5e-4 and use 70 epochs cosine learning schedule (with 0 warm up), then most CV boost about +0.010! and then score around LB 0.130 with lots of folds.


14
Robust Scaler qunatile-range=(10,90)
We tried out all sorts of scalers but found that the robust scaler worked the best for our model as in most of the popular notebooks. However, we also ascertained that some of our features ranged from -200 ~ +200 even after scaling with the default quantile_range of (25,75). Therefore we expanded the range of our scalers quantile_range and found out that (10,90) worked the best for us. This improved one of our member's models by approximately 0.010.

```python
train['pressure_diff']=train.groupby(['breath_id'])['pressure'].diff().fillna(0)
train['pressure_diff2']=train.groupby(['breath_id'])['pressure_diff'].shift(-1).fillna(0)
```
CosineAnnealingWarmupRestarts


===
`df['delta_delta_pressure'] = (df['delta_pressure'] - df.groupby('breath_id')['delta_pressure'].shift(1)).fillna(0).values`


