import numpy as np

def add_noise(intensity, noise_level=0.01):
    # --- MODIFICATION: Use a random noise level for each sample ---
    random_noise_level = np.random.uniform(0, noise_level)
    noise = np.random.normal(0, random_noise_level, intensity.shape)
    return intensity + noise

def add_baseline(intensity, coeff=0.0001):
    trend = coeff * np.linspace(-1, 1, len(intensity))**2
    return intensity + trend

def add_spikes(intensity, num_spikes=5, spike_height=0.5):
    corrupted = intensity.copy()
    # --- MODIFICATION: Use a random number of spikes for each sample ---
    random_num_spikes = np.random.randint(0, num_spikes + 1)
    if random_num_spikes > 0:
        indices = np.random.choice(len(intensity), random_num_spikes, replace=False)
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