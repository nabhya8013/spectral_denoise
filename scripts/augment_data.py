import numpy as np

def add_noise(intensity, noise_level=0.01):
    noise = np.random.normal(0, noise_level, intensity.shape)
    return intensity + noise

def add_baseline(intensity, coeff=0.0001):
    trend = coeff * np.linspace(-1, 1, len(intensity))**2
    return intensity + trend

def add_spikes(intensity, num_spikes=5, spike_height=0.5):
    corrupted = intensity.copy()
    indices = np.random.choice(len(intensity), num_spikes)
    for i in indices:
        corrupted[i] += spike_height * np.random.uniform(0.5, 1.0)
    return corrupted

def augment_spectrum(intensity, apply_noise=True, apply_baseline=True, apply_spikes=True):
    """Apply augmentation pipeline to one spectrum."""
    augmented = intensity.copy()
    
    if apply_noise:
        augmented = add_noise(augmented)
    if apply_baseline:
        augmented = add_baseline(augmented)
    if apply_spikes:
        augmented = add_spikes(augmented)
    
    return augmented
