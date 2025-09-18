import os
import numpy as np

# Paths
clean_dir = "data/train/clean"
noisy_dir = "data/train/noisy"
pairs_dir = "data/pairs"

os.makedirs(pairs_dir, exist_ok=True)

# Get file lists
clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.txt')])
noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.txt')])

# Make sure both folders have same number of files
assert len(clean_files) == len(noisy_files), "Mismatch between clean and noisy files!"

for fname_c, fname_n in zip(clean_files, noisy_files):
    clean_path = os.path.join(clean_dir, fname_c)
    noisy_path = os.path.join(noisy_dir, fname_n)

    # Skip header row (assumes first row has text like 'Wavenumber Intensity')
    clean = np.loadtxt(clean_path, skiprows=1)
    noisy = np.loadtxt(noisy_path, skiprows=1)

    # Use only intensity column (2nd column, index 1)
    clean_y = clean[:, 1]
    noisy_y = noisy[:, 1]

    # Save as .npy pair
    base = os.path.splitext(fname_c)[0]
    np.save(os.path.join(pairs_dir, f"{base}_clean.npy"), clean_y)
    np.save(os.path.join(pairs_dir, f"{base}_noisy.npy"), noisy_y)

print(f"✅ Saved {len(clean_files)} pairs to {pairs_dir}")
