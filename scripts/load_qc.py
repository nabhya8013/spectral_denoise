import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class FTIRDataset:
    def __init__(self, data_dir, grid_min=400, grid_max=4000, grid_points=1024):
        self.data_dir = data_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('.txt')])
        self.wavenumber_grid = np.linspace(grid_max, grid_min, grid_points)
        
    def load_spectrum(self, filepath):
        try:
            data = np.loadtxt(filepath)
            wn = data[:, 0]
            intensity = data[:, 1]
            return wn, intensity
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None, None
        
    def interpolate_spectrum(self, wn, intensity):
        f = interp1d(wn, intensity, kind='cubic', bounds_error=False, fill_value="extrapolate")
        interpolated = f(self.wavenumber_grid)
        return interpolated
    
    def is_valid(self, intensity):
        if np.any(np.isnan(intensity)) or np.any(np.isinf(intensity)):
            return False
        if np.max(np.abs(np.diff(intensity))) > 1e3:
            return False
        return True
    
    def process_all(self):
        processed = []
        for file in self.files:
            filepath = os.path.join(self.data_dir, file)
            wn, intensity = self.load_spectrum(filepath)
            if wn is None or intensity is None:
                continue
            interpolated = self.interpolate_spectrum(wn, intensity)
            valid = self.is_valid(interpolated)
            processed.append({
                'filename': file,
                'interpolated': interpolated,
                'valid': valid
            })
        return processed

if __name__ == "__main__":
    dataset = FTIRDataset('data/raw')
    spectra = dataset.process_all()
    for spec in spectra:
        if spec['valid']:
            plt.plot(dataset.wavenumber_grid, spec['interpolated'], label='Interpolated')
            plt.xlabel('Wavenumber (cm⁻¹)')
            plt.ylabel('Intensity')
            plt.title(f"Spectra from {spec['filename']}")
            plt.legend()
            plt.show()
            break
