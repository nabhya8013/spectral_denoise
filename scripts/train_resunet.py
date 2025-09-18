import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from resunet_model import build_resunet
from augment_data import add_noise, add_baseline, add_spikes
from load_qc import FTIRDataset

def prepare_training_data(dataset):
    X = []
    Y = []
    for spec in dataset.process_all():
        if spec['valid']:
            clean = spec['interpolated']
            noisy = add_noise(clean, noise_level=0.02)
            noisy = add_baseline(noisy, coeff=0.0001)
            noisy = add_spikes(noisy, num_spikes=10, spike_height=0.5)
            X.append(noisy)
            Y.append(clean)
    return np.array(X), np.array(Y)

if __name__ == "__main__":
    dataset = FTIRDataset('data/raw')
    X, Y = prepare_training_data(dataset)
    X = X[..., np.newaxis]
    Y = Y[..., np.newaxis]
    
    model = build_resunet(input_shape=X.shape[1:])
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
    
    model.fit(X, Y, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop, reduce_lr])
    model.save('models/resunet_model.h5')
    print("Model saved to models/resunet_model.h5")
