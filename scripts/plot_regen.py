import numpy as np
import matplotlib.pyplot as plt

def regenerate_and_save_plot(wavenumber_grid, intensity, filename):
    plt.plot(wavenumber_grid, intensity)
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.title(f'Spectrum: {os.path.basename(filename)}')
    plt.savefig(filename)
    plt.close()


