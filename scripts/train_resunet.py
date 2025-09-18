import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# -------------------------------
# Dataset loader
# -------------------------------
class SpectraDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir):
        self.noisy_files = sorted([os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir) if f.endswith('.txt')])
        self.clean_files = sorted([os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith('.txt')])

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy = np.loadtxt(self.noisy_files[idx], skiprows=1)[:, 1]
        clean = np.loadtxt(self.clean_files[idx], skiprows=1)[:, 1]

        # normalize (mean 0, std 1)
        noisy = (noisy - np.mean(noisy)) / (np.std(noisy) + 1e-8)
        clean = (clean - np.mean(clean)) / (np.std(clean) + 1e-8)

        noisy = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0)  # (1, L)
        clean = torch.tensor(clean, dtype=torch.float32).unsqueeze(0)  # (1, L)

        return noisy, clean

# -------------------------------
# ResUNet Model
# -------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class ResUNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(1, 16)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool1d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(32, 64)

        # Decoder
        self.up2 = nn.ConvTranspose1d(64, 32, 2, stride=2)
        self.dec2 = ConvBlock(64, 32)
        self.up1 = nn.ConvTranspose1d(32, 16, 2, stride=2)
        self.dec1 = ConvBlock(32, 16)

        # Output
        self.out_conv = nn.Conv1d(16, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)
        return out

# -------------------------------
# Training Loop
# -------------------------------
def train_model(train_loader, val_loader, device, epochs=20, lr=1e-3, save_path="models/resunet_ftir.pth"):
    model = ResUNet1D().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            outputs = model(noisy)
            loss = criterion(outputs, clean)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                outputs = model(noisy)
                val_loss += criterion(outputs, clean).item()

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to {save_path}")

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = SpectraDataset("data/train/noisy", "data/train/clean")
    val_dataset = SpectraDataset("data/val/noisy", "data/val/clean")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    train_model(train_loader, val_loader, device, epochs=50, lr=1e-3)
