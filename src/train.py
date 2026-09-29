import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import list_pairs, split_names, UnderwaterDataset
from model import UnderwaterRestorationModel
from losses import CombinedLoss, psnr, ssim


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
RAW_DIR = os.path.join("data", "raw-890")
REFERENCE_DIR = os.path.join("data", "reference-890")

MODEL_PATH = os.path.join("models", "underwater_restoration.pth")
LAST_MODEL_PATH = os.path.join("models", "underwater_restoration_last.pth")

IMAGE_SIZE = (256, 256)
EPOCHS = 200
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
VAL_RATIO = 0.1
SEED = 42


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------------------------------------------------
# Data (train/validation split + augmentation on training only)
# ------------------------------------------------------------
names = list_pairs(RAW_DIR, REFERENCE_DIR)
train_names, val_names = split_names(names, VAL_RATIO, SEED)

train_set = UnderwaterDataset(
    RAW_DIR, REFERENCE_DIR, train_names, IMAGE_SIZE, augment=True
)
val_set = UnderwaterDataset(
    RAW_DIR, REFERENCE_DIR, val_names, IMAGE_SIZE, augment=False
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)


# ------------------------------------------------------------
# Model, loss, optimizer, scheduler
# ------------------------------------------------------------
model = UnderwaterRestorationModel().to(device)

# The VGG perceptual loss is slow on CPU, so only use it with a GPU
criterion = CombinedLoss(
    ssim_weight=0.2,
    perceptual_weight=0.05,
    use_perceptual=torch.cuda.is_available()
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------
def evaluate(use_model=True):
    """
    Average PSNR / SSIM against the reference images.
    use_model=False measures the RAW image itself (the baseline),
    so you can see how much the model improves on the original.
    """

    if use_model:
        model.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    with torch.no_grad():
        for raw, reference in val_loader:

            raw = raw.to(device)
            reference = reference.to(device)

            output = model(raw).clamp(0, 1) if use_model else raw

            batch = raw.size(0)
            total_psnr += psnr(output, reference).item() * batch
            total_ssim += ssim(output, reference).item() * batch
            count += batch

    return total_psnr / count, total_ssim / count


os.makedirs("models", exist_ok=True)

base_psnr, base_ssim = evaluate(use_model=False)
print(f"Baseline (raw image vs reference): "
      f"PSNR {base_psnr:.2f} dB, SSIM {base_ssim:.4f}\n")


# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------
best_psnr = 0.0

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0.0

    for raw, reference in train_loader:

        raw = raw.to(device)
        reference = reference.to(device)

        optimizer.zero_grad()

        output = model(raw)
        loss = criterion(output, reference)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    train_loss = total_loss / len(train_loader)
    val_psnr, val_ssim = evaluate()

    marker = ""
    if val_psnr > best_psnr:
        best_psnr = val_psnr
        torch.save(model.state_dict(), MODEL_PATH)
        marker = "  <- best model saved"

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {train_loss:.4f} | "
        f"Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f}"
        f"{marker}"
    )

torch.save(model.state_dict(), LAST_MODEL_PATH)

print("\nTraining completed.")
print(f"Baseline PSNR {base_psnr:.2f} dB -> best model PSNR {best_psnr:.2f} dB")
print("Best model:", MODEL_PATH)