import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Paths
pairs_dir = "data/pairs"
dataset_dir = "data/dataset"
os.makedirs(dataset_dir, exist_ok=True)

def normalize_spectrum(spectrum, method="zscore"):
    """Normalize spectrum to help model training."""
    if method == "zscore":
        mean = spectrum.mean()
        std = spectrum.std()
        return (spectrum - mean) / (std + 1e-8)
    elif method == "minmax":
        min_val = spectrum.min()
        max_val = spectrum.max()
        return (spectrum - min_val) / (max_val - min_val + 1e-8)
    else:
        return spectrum

# Get list of clean files
clean_files = sorted([f for f in os.listdir(pairs_dir) if f.endswith("_clean.npy")])

X_noisy = []
Y_clean = []

print(f"Found {len(clean_files)} clean/noisy pairs.")

for fname_c in clean_files:
    base = fname_c.replace("_clean.npy", "")
    fname_n = f"{base}_noisy.npy"

    clean_path = os.path.join(pairs_dir, fname_c)
    noisy_path = os.path.join(pairs_dir, fname_n)

    clean_y = np.load(clean_path)
    noisy_y = np.load(noisy_path)

    if clean_y.shape != noisy_y.shape:
        print(f"❌ Length mismatch in {base}: clean={clean_y.shape}, noisy={noisy_y.shape}")
        continue

    # Normalize spectra (z-score by default)
    clean_y = normalize_spectrum(clean_y, method="zscore")
    noisy_y = normalize_spectrum(noisy_y, method="zscore")

    Y_clean.append(clean_y)
    X_noisy.append(noisy_y)

    # Quick QC plot
    plt.figure(figsize=(10, 4))
    plt.plot(clean_y, label="Clean (norm)", linewidth=2)
    plt.plot(noisy_y, label="Noisy (norm)", alpha=0.7)
    plt.title(f"QC Pair (Normalized): {base}")
    plt.xlabel("Index")
    plt.ylabel("Normalized Intensity")
    plt.legend()
    plt.show()

    stop = input("Press Enter for next, or 'q' to quit QC early: ")
    if stop.lower() == 'q':
        break

# Convert to arrays
X_noisy = np.array(X_noisy, dtype=np.float32)
Y_clean = np.array(Y_clean, dtype=np.float32)

print(f"\nFinal dataset shapes: X_noisy={X_noisy.shape}, Y_clean={Y_clean.shape}")

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X_noisy, Y_clean, test_size=0.2, random_state=42
)

print(f"Train: X={X_train.shape}, Y={Y_train.shape}")
print(f"Test:  X={X_test.shape}, Y={Y_test.shape}")

# Save
np.save(os.path.join(dataset_dir, "X_noisy.npy"), X_noisy)
np.save(os.path.join(dataset_dir, "Y_clean.npy"), Y_clean)
np.save(os.path.join(dataset_dir, "X_train.npy"), X_train)
np.save(os.path.join(dataset_dir, "Y_train.npy"), Y_train)
np.save(os.path.join(dataset_dir, "X_test.npy"), X_test)
np.save(os.path.join(dataset_dir, "Y_test.npy"), Y_test)

print(f"✅ Saved normalized + split dataset to {dataset_dir}")
