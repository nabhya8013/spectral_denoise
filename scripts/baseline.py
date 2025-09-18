import os
import numpy as np
from pybaselines import als
import matplotlib.pyplot as plt

def process_file(file_path, output_dir):
    data = np.loadtxt(file_path)  # assuming txt with 2 columns: x, intensity
    x, y = data[:, 0], data[:, 1]

    baseline, _ = als.als(y, lam=1e6, p=0.01)
    corrected = y - baseline

    # Save corrected spectrum
    out_file = os.path.join(output_dir, os.path.basename(file_path))
    np.savetxt(out_file, np.column_stack((x, corrected)))

    # Optional plot
    plt.plot(x, y, label='Original')
    plt.plot(x, baseline, label='Baseline')
    plt.plot(x, corrected, label='Corrected')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    input_dir = "data/raw"
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if fname.endswith(".txt"):
            process_file(os.path.join(input_dir, fname), output_dir)
