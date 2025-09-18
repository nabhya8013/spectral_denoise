import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Add
from tensorflow.keras.models import Model

def res_block(x, filters, kernel_size=3):
    shortcut = x
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    return x

def build_resunet(input_shape=(1024, 1)):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    for filters in [64, 128, 256]:
        x = res_block(x, filters)
    x = res_block(x, 512)
    for filters in [256, 128, 64]:
        x = res_block(x, filters)
    outputs = Conv1D(1, 1, activation='linear')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

if __name__ == "__main__":
    model = build_resunet()
    model.summary()
