import os
import numpy as np
from augment_data import augment_spectrum

# Paths
clean_dir = "data/train/clean"
noisy_dir = "data/train/noisy"

os.makedirs(noisy_dir, exist_ok=True)

clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith(".txt")])

for fname in clean_files:
    clean_path = os.path.join(clean_dir, fname)
    noisy_path = os.path.join(noisy_dir, fname)

    data = np.loadtxt(clean_path, skiprows=1)
    # The script assumes your data has two columns: wavenumber and intensity.
    # It adds noise to the intensity column (index 1).
    wavenumber, intensity = data[:, 0], data[:, 1]
    
    # Apply a combination of noise, baseline drift, and spikes
    noisy_intensity = augment_spectrum(intensity, apply_noise=True, apply_baseline=True, apply_spikes=True)
    
    noisy_data = np.column_stack((wavenumber, noisy_intensity))
    np.savetxt(noisy_path, noisy_data, header="Wavenumber Intensity", comments="")

print(f"✅ Generated {len(clean_files)} noisy files in {noisy_dir}")