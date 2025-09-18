# scripts/make_pairs.py
import os
import numpy as np

def add_noise(y, noise_level=0.02):
    noise = np.random.normal(0, noise_level * np.std(y), size=y.shape)
    return y + noise

def prepare_pairs(input_dir="data/processed", output_dir="data/train", val_split=0.2):
    files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    os.makedirs(os.path.join(output_dir, "noisy"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "clean"), exist_ok=True)

    n_val = int(len(files) * val_split)
    val_dir = "data/val"
    os.makedirs(os.path.join(val_dir, "noisy"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "clean"), exist_ok=True)

    for i, fname in enumerate(files):
        data = np.loadtxt(os.path.join(input_dir, fname), skiprows=1)  # skip header
        x, y = data[:, 0], data[:, 1]

        noisy_y = add_noise(y, noise_level=0.05)

        # save clean
        pair = np.column_stack((x, y))
        np.savetxt(
            os.path.join(output_dir if i >= n_val else val_dir, "clean", fname),
            pair, header="Wavenumber Intensity", comments=""
        )

        # save noisy
        noisy_pair = np.column_stack((x, noisy_y))
        np.savetxt(
            os.path.join(output_dir if i >= n_val else val_dir, "noisy", fname),
            noisy_pair, header="Wavenumber Intensity(noisy)", comments=""
        )

    print(f"Generated {len(files) - n_val} train pairs and {n_val} validation pairs")

if __name__ == "__main__":
    prepare_pairs()
