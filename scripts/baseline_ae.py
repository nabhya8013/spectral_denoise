import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def build_baseline_ae(input_dim=1024, latent_dim=32):
    inputs = Input(shape=(input_dim,))
    x = Dense(256, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    latent = Dense(latent_dim, activation='relu')(x)
    x = Dense(64, activation='relu')(latent)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(input_dim, activation='linear')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

if __name__ == "__main__":
    model = build_baseline_ae()
    model.summary()
