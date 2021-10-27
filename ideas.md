Простое:

Ideas to try:
I. General architecture/training
* gaussian noise:
x = torch.zeros(5, 10, 20, dtype=torch.float64)
x = x + (0.1**0.5)*torch.randn(5, 10, 20)
def gaussian(ins, is_training, mean, stddev):
    if is_training:
        noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
        return ins + noise
    return ins
* use only u_out == 0 for training
* postprocess before loss calculation
* dual regression head
* batchnorm1d after lstm/gru
* resnet-like
* densenet-like
* activation=torch.nn.PReLU
activation=torch.nn.Tanh
activation=torch.nn.SELU 
SiLU
* 1 lstm with 4-6 num_layers
* dropout
* mlp for lstm

===
* head for 950 classes?
* head for next pressure prediction?
* 

II. Research
* read code for molecules again
* google more papers

III. Architecture
architecture experiments:

* MLP to higher dimension for transformer
* mlp, transformer and other things
* only transformer
* transformer encoder and decoder?
* multihead attention?!!!!!!!!!!!!!!!!!!!
positional embedding?!!!!!!!!!!!!!!!!!!!
===
NO EARLY STOPPING


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

def dnn_model():
    
    x_input = Input(shape=(train.shape[-2:]))
    
    x1 = Bidirectional(LSTM(units=768, return_sequences=True))(x_input)
    x2 = Bidirectional(LSTM(units=512, return_sequences=True))(x1)
    x3 = Bidirectional(LSTM(units=256, return_sequences=True))(x2)
    
    z2 = Bidirectional(GRU(units=256, return_sequences=True))(x2)
    z3 = Bidirectional(GRU(units=128, return_sequences=True))(Add()([x3, z2]))
    
    x = Concatenate(axis=2)([x3, z2, z3])
    x = Bidirectional(LSTM(units=192, return_sequences=True))(x)
    
    x = Dense(units=128, activation='selu')(x)
    
    x_output = Dense(units=1)(x)

    model = Model(inputs=x_input, outputs=x_output, 
                  name='DNN_Model')
    return model

def get_model():
    inputs = keras.layers.Input(shape=train.shape[-2:])
    
#     x = keras.layers.Bidirectional(keras.layers.LSTM(2048, return_sequences=True))(inputs)
    x = keras.layers.Bidirectional(keras.layers.LSTM(1024, return_sequences=True))(inputs)
    x1 = keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True))(x)
    x2 = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True))(x1)
    
#     z2 = keras.layers.Bidirectional(keras.layers.GRU(units=256, return_sequences=True))(x2)
#     z3 = keras.layers.Bidirectional(keras.layers.GRU(units=128, return_sequences=True))(keras.layers.Add()([x2, z2]))
    x3 = tf.keras.layers.Concatenate(axis=2)([x1,x2])
    x4 = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True))(x3)
    
    x5 = keras.layers.Dense(100, activation='selu')(x4)
    x6 = keras.layers.Dense(100, activation='selu')(x5)
    x7 = keras.layers.Dense(100, activation='selu')(x6)
    x7 = tf.keras.layers.Concatenate(axis=2)([x7,x5])
    outputs = keras.layers.Dense(1)(x7)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="lstm_model")
    return model


def dnn_model():
    
    x_input = Input(shape=(train.shape[-2:]))
    
    x1 = Bidirectional(LSTM(units=768, return_sequences=True))(x_input)
    x2 = Bidirectional(LSTM(units=512, return_sequences=True))(x1)
    x3 = Bidirectional(LSTM(units=384, return_sequences=True))(x2)
    x4 = Bidirectional(LSTM(units=256, return_sequences=True))(x3)
    x5 = Bidirectional(LSTM(units=128, return_sequences=True))(x4)
    
    z2 = Bidirectional(GRU(units=384, return_sequences=True))(x2)
    
    z31 = Multiply()([x3, z2])
    z31 = BatchNormalization()(z31)
    z3 = Bidirectional(GRU(units=256, return_sequences=True))(z31)
    
    z41 = Multiply()([x4, z3])
    z41 = BatchNormalization()(z41)
    z4 = Bidirectional(GRU(units=128, return_sequences=True))(z41)
    
    z51 = Multiply()([x5, z4])
    z51 = BatchNormalization()(z51)
    z5 = Bidirectional(GRU(units=64, return_sequences=True))(z51)
    
    x = Concatenate(axis=2)([x5, z2, z3, z4, z5])
    
    x = Dense(units=128, activation='selu')(x)
    
    x_output = Dense(units=1)(x)

    model = Model(inputs=x_input, outputs=x_output, 
                  name='DNN_Model')
    return model


