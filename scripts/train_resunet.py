import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from augment_data import add_noise, add_baseline, add_spikes

# -------------------
# Device Selection
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# -------------------
# Dataset
# -------------------
class FTIRPairsDataset(Dataset):
    def __init__(self, clean_files, noisy_files, augment=True):
        self.clean_files = clean_files
        self.noisy_files = noisy_files
        self.augment = augment

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean = np.load(self.clean_files[idx])
        noisy = np.load(self.noisy_files[idx])

        # Apply augmentations only on training data
        if self.augment:
            if np.random.rand() < 0.5:
                noisy = add_noise(noisy, noise_level=0.02)
            if np.random.rand() < 0.3:
                noisy = add_baseline(noisy, coeff=0.0005)
            if np.random.rand() < 0.2:
                noisy = add_spikes(noisy, num_spikes=3)

        clean = torch.tensor(clean, dtype=torch.float32).unsqueeze(0)
        noisy = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0)
        return noisy, clean

# -------------------
# ResUNet1D Model
# -------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.shortcut = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

class ResUNet1D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch*2)
        self.enc3 = ConvBlock(base_ch*2, base_ch*4)

        self.pool = nn.MaxPool1d(2)

        self.bottleneck = ConvBlock(base_ch*4, base_ch*8)

        self.up2 = nn.ConvTranspose1d(base_ch*8, base_ch*4, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_ch*8, base_ch*4)

        self.up1 = nn.ConvTranspose1d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_ch*4, base_ch*2)

        self.final = nn.Conv1d(base_ch*2, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d2 = self.up2(b)
        d2 = torch.cat([d2, e3], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return out

# -------------------
# Loss Function (MSE + Structural Similarity Approx)
# -------------------
class HybridLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        # Approximate SSIM with cosine similarity (1D spectra)
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        cos_sim = torch.nn.functional.cosine_similarity(pred_flat, target_flat, dim=1).mean()
        ssim_loss = 1 - cos_sim
        return self.alpha * mse_loss + self.beta * ssim_loss

# -------------------
# Training Function
# -------------------
def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    criterion = HybridLoss()

    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                output = model(noisy)
                loss = criterion(output, clean)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")

# -------------------
# Main
# -------------------
if __name__ == "__main__":
    pairs_dir = "data/pairs"
    clean_files = sorted([os.path.join(pairs_dir, f) for f in os.listdir(pairs_dir) if f.endswith("clean.npy")])
    noisy_files = sorted([os.path.join(pairs_dir, f) for f in os.listdir(pairs_dir) if f.endswith("noisy.npy")])

    train_c, val_c, train_n, val_n = train_test_split(clean_files, noisy_files, test_size=0.2, random_state=42)

    train_dataset = FTIRPairsDataset(train_c, train_n, augment=True)
    val_dataset = FTIRPairsDataset(val_c, val_n, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    model = ResUNet1D().to(device)
    train_model(model, train_loader, val_loader, epochs=50)
