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