import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.signal import resample
from skimage.metrics import structural_similarity as ssim
from train_resunet import SpectraDataset, ResUNet1D  # your model & dataset

# ------------------------
# Settings
# ------------------------
pairs_dir = "data/pairs"
target_len = 1024
batch_size = 16

# Device selection (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
dataset = SpectraDataset(pairs_dir, target_len=target_len)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Load model
model_path = "models/resunet1d.pth"
model = ResUNet1D(input_len=target_len).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"✅ Model loaded from {model_path}")

# ------------------------
# Evaluation
# ------------------------
mse_list, psnr_list, ssim_list = [], [], []

with torch.no_grad():
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        output = model(noisy)

        # Convert to numpy
        out_np = output.cpu().numpy()
        clean_np = clean.cpu().numpy()

        for o, c in zip(out_np, clean_np):
            mse_val = np.mean((o - c)**2)
            mse_list.append(mse_val)
            
            psnr_val = 20 * np.log10(np.max(c) / (np.sqrt(mse_val)+1e-8))
            psnr_list.append(psnr_val)
            
            ssim_val = ssim(o, c, data_range=c.max()-c.min())
            ssim_list.append(ssim_val)

# Compute mean metrics
mean_mse = np.mean(mse_list)
mean_psnr = np.mean(psnr_list)
mean_ssim = np.mean(ssim_list)

# Convert metrics to quality scores
mse_quality = 100 / (1 + mean_mse)                # smaller MSE → higher quality
psnr_quality = mean_psnr / 50 * 100              # scale assuming 50 dB max
ssim_quality = mean_ssim * 100                   # 0-1 → 0-100

# Weighted overall quality (adjust weights if needed)
overall_quality = 0.2*mse_quality + 0.3*psnr_quality + 0.5*ssim_quality

# ------------------------
# Print results
# ------------------------
print("\n✅ Evaluation Results on Validation Set:")
print(f"Mean MSE: {mean_mse:.6f}")
print(f"Mean PSNR: {mean_psnr:.2f} dB")
print(f"Mean SSIM: {mean_ssim:.4f}")
print(f"Overall Quality Score: {overall_quality:.2f}%")
