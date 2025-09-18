import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.signal import resample
from skimage.metrics import structural_similarity as ssim
from train_resunet_cpu import SpectraDataset, ResUNet1D

# ------------------------
# Settings
# ------------------------
pairs_dir = "data/pairs"
target_len = 1024
batch_size = 16

# Load dataset
dataset = SpectraDataset(pairs_dir, target_len=target_len)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Load model
device = torch.device("cpu")
model = ResUNet1D(input_len=target_len).to(device)
model.load_state_dict(torch.load("models/resunet1d_cpu.pth", map_location=device))
model.eval()

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
            
            ssim_val = ssim(o, c)
            ssim_list.append(ssim_val)

print(f"✅ Evaluation Results on Validation Set:")
print(f"Mean MSE: {np.mean(mse_list):.6f}")
print(f"Mean PSNR: {np.mean(psnr_list):.2f} dB")
print(f"Mean SSIM: {np.mean(ssim_list):.4f}")
