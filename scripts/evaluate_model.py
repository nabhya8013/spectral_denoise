import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
from scipy.signal import resample
from train_resunet import ResUNet1D, FTIRPairsDataset

# ------------------------
# Settings
# ------------------------
pairs_dir = "data/pairs"
target_len = 1024
batch_size = 8
model_path = "models/resunet1d.pth"
results_path = "results/eval_metrics.json"

# Device selection (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ------------------------
# Helper functions
# ------------------------
def normalize(arr):
    """Normalize array to [0,1] range."""
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

def compute_metrics(o, c):
    """Compute MSE, PSNR, SSIM, Pearson correlation between two 1D arrays."""
    o_norm, c_norm = normalize(o), normalize(c)

    # MSE
    mse_val = np.mean((o_norm - c_norm) ** 2)

    # PSNR (normalized, so max=1)
    psnr_val = 20 * np.log10(1.0 / (np.sqrt(mse_val) + 1e-8))

    # SSIM (for 1D signals, flatten and treat as single channel)
    ssim_val = ssim(o_norm, c_norm, data_range=1.0)

    # Pearson correlation
    corr_val, _ = pearsonr(o_norm.flatten(), c_norm.flatten())

    return mse_val, psnr_val, ssim_val, corr_val

# ------------------------
# Evaluation
# ------------------------
def evaluate_model():
    print("\nStarting evaluation...")
    
    # Load Model
    model = ResUNet1D().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded from {model_path} ({num_params/1e6:.2f}M parameters)")

    # Load data for evaluation
    clean_files = sorted([os.path.join(pairs_dir, f) for f in os.listdir(pairs_dir) if f.endswith("clean.npy")])
    noisy_files = sorted([os.path.join(pairs_dir, f) for f in os.listdir(pairs_dir) if f.endswith("noisy.npy")])
    
    # The evaluation data is a subset of the total data as defined by the split
    train_c, val_c, train_n, val_n = train_test_split(clean_files, noisy_files, test_size=0.2, random_state=42)
    val_dataset = FTIRPairsDataset(val_c, val_n, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    mse_list, psnr_list, ssim_list, corr_list = [], [], [], []

    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)

            out_np = output.cpu().numpy()
            clean_np = clean.cpu().numpy()

            for o, c in zip(out_np, clean_np):
                mse_val, psnr_val, ssim_val, corr_val = compute_metrics(o.squeeze(), c.squeeze())
                mse_list.append(mse_val)
                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)
                corr_list.append(corr_val)

    mean_mse = np.mean(mse_list)
    mean_psnr = np.mean(psnr_list)
    mean_ssim = np.mean(ssim_list)
    mean_corr = np.mean(corr_list)

    mse_quality = 100 / (1 + mean_mse)
    psnr_quality = min(mean_psnr / 50 * 100, 100)
    ssim_quality = mean_ssim * 100
    corr_quality = mean_corr * 100

    weights = {"mse": 0.2, "psnr": 0.3, "ssim": 0.3, "corr": 0.2}
    overall_quality = (
        weights["mse"] * mse_quality +
        weights["psnr"] * psnr_quality +
        weights["ssim"] * ssim_quality +
        weights["corr"] * corr_quality
    )

    print("\n✅ Evaluation Results on Validation Set:")
    print(f"Mean MSE:   {mean_mse:.6f}")
    print(f"Mean PSNR:  {mean_psnr:.2f} dB")
    print(f"Mean SSIM:  {mean_ssim:.4f}")
    print(f"Mean Corr:  {mean_corr:.4f}")
    print(f"Overall Quality Score: {overall_quality:.2f}%")

    metrics = {
        "mean_mse": float(mean_mse),
        "mean_psnr": float(mean_psnr),
        "mean_ssim": float(mean_ssim),
        "mean_corr": float(mean_corr),
        "mse_quality": float(mse_quality),
        "psnr_quality": float(psnr_quality),
        "ssim_quality": float(ssim_quality),
        "corr_quality": float(corr_quality),
        "overall_quality": float(overall_quality),
        "weights": weights
    }
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\n📂 Metrics saved to {results_path}")

if __name__ == "__main__":
    evaluate_model()