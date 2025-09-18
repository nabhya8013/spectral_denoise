import numpy as np
import matplotlib.pyplot as plt

def plot_spectra(raw, baseline, denoised, wavenumber_grid, title="Spectra Comparison"):
    plt.plot(wavenumber_grid, raw, label='Raw')
    plt.plot(wavenumber_grid, baseline, label='Baseline Corrected')
    plt.plot(wavenumber_grid, denoised, label='Denoised')
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.title(title)
    plt.legend()
    plt.show()
