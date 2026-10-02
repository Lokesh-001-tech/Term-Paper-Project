import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import list_pairs, UnderwaterDataset
from model import UnderwaterRestorationModel
from losses import CombinedLoss


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
RAW_DIR = os.path.join("data", "raw-890")
REFERENCE_DIR = os.path.join("data", "reference-890")

MODEL_PATH = os.path.join("models", "underwater_restoration.pth")
LAST_MODEL_PATH = os.path.join("models", "underwater_restoration_last.pth")

# Optional: start from an already trained model instead of from scratch.
# Set to MODEL_PATH to fine-tune your current model with the new losses
# and blur/noise augmentation (needs fewer epochs, e.g. 80-100).
# Leave as None to train from scratch.
INIT_FROM = None

# If you change this, change MODEL_SIZE in inference.py to the same value.
IMAGE_SIZE = (256, 256)

EPOCHS = 200
BATCH_SIZE = 8
LEARNING_RATE = 2e-4

# Random blur / noise added to the RAW training image only
BLUR_PROB = 0.5
MAX_BLUR_SIGMA = 3.0
NOISE_PROB = 0.5
MAX_NOISE = 5.0

SEED = 42


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
# IMPORTANT:
# All 890 paired images are used for training.
# There is NO train/validation split.
# challenging-60 is kept completely separate and is used
# only during inference.
# ------------------------------------------------------------

names = list_pairs(
    RAW_DIR,
    REFERENCE_DIR
)

print("Total paired images:", len(names))


train_set = UnderwaterDataset(
    RAW_DIR,
    REFERENCE_DIR,
    names,
    IMAGE_SIZE,
    augment=True,
    degrade=True,
    blur_prob=BLUR_PROB,
    max_blur_sigma=MAX_BLUR_SIGMA,
    noise_prob=NOISE_PROB,
    max_noise=MAX_NOISE
)


train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
model = UnderwaterRestorationModel().to(device)

if INIT_FROM is not None:
    model.load_state_dict(torch.load(INIT_FROM, map_location=device))
    print("Starting from trained model:", INIT_FROM)

print(
    "Total model parameters:",
    sum(p.numel() for p in model.parameters())
)


# ------------------------------------------------------------
# Loss
# ------------------------------------------------------------
criterion = CombinedLoss(
    ssim_weight=0.2,
    perceptual_weight=0.05,
    edge_weight=0.3,
    color_weight=0.2,
    use_perceptual=torch.cuda.is_available()
).to(device)


# ------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ------------------------------------------------------------
# Learning-rate scheduler
# ------------------------------------------------------------
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6
)


# ------------------------------------------------------------
# Create model directory
# ------------------------------------------------------------
os.makedirs("models", exist_ok=True)


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
best_loss = float("inf")


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for raw, reference in train_loader:

        raw = raw.to(
            device,
            non_blocking=True
        )

        reference = reference.to(
            device,
            non_blocking=True
        )

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(raw)

        # Calculate loss
        loss = criterion(
            output,
            reference
        )

        # Backpropagation
        loss.backward()

        # Prevent very large gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        # Update model weights
        optimizer.step()

        total_loss += loss.item()


    # Update learning rate
    scheduler.step()


    # Average training loss
    train_loss = (
        total_loss / len(train_loader)
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------
    marker = ""

    if train_loss < best_loss:

        best_loss = train_loss

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        marker = " <- best model saved"


    # Current learning rate
    current_lr = optimizer.param_groups[0]["lr"]


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {train_loss:.4f} | "
        f"LR: {current_lr:.7f}"
        f"{marker}"
    )


    # --------------------------------------------------------
    # Save latest model every epoch
    # --------------------------------------------------------
    torch.save(
        model.state_dict(),
        LAST_MODEL_PATH
    )


# ------------------------------------------------------------
# Training completed
# ------------------------------------------------------------
print("\nTraining completed.")

print(
    f"Best training loss: {best_loss:.4f}"
)

print(
    "Best model:",
    MODEL_PATH
)

print(
    "Last model:",
    LAST_MODEL_PATH
)