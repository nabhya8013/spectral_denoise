from sklearn.metrics import mean_squared_error
import numpy as np
import cv2

def compute_psnr(target, output):
    max_pixel = np.max(target)
    mse = mean_squared_error(target, output)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def compute_ssim(target, output):
    target_uint8 = np.uint8((target - np.min(target)) / (np.max(target) - np.min(target)) * 255)
    output_uint8 = np.uint8((output - np.min(output)) / (np.max(output) - np.min(output)) * 255)
    ssim_value = cv2.quality.QualitySSIM_compute(target_uint8, output_uint8)[0][0]
    return ssim_value

def compute_nrmse(target, output):
    return np.sqrt(mean_squared_error(target, output)) / (np.max(target) - np.min(target))
