import numpy as np

def find_peaks(intensity, threshold=0.1):
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(intensity, height=threshold * np.max(intensity))
    return peaks

def peak_shift_metric(raw, denoised, wavenumber_grid):
    raw_peaks = find_peaks(raw)
    denoised_peaks = find_peaks(denoised)
    # Returns array of shifts (in cm⁻¹), matched by index
    return wavenumber_grid[denoised_peaks] - wavenumber_grid[raw_peaks]

def fwhm(intensity, wavenumber_grid):
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(intensity)
    results = []
    for p in peaks:
        half_max = intensity[p] / 2.0
        left = np.where(intensity[:p] <= half_max)[0]
        right = np.where(intensity[p:] <= half_max)[0]
        l_idx = left[-1] if left.size else 0
        r_idx = p + right[0] if right.size else len(intensity)-1
        width = np.abs(wavenumber_grid[r_idx] - wavenumber_grid[l_idx])
        results.append(width)
    return results  # widths for all detected peaks

def band_area(intensity, wavenumber_grid, start, end):
    idx = (wavenumber_grid >= start) & (wavenumber_grid <= end)
    area = np.trapz(intensity[idx], wavenumber_grid[idx])
    return area
