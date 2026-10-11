import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import list_pairs, UnderwaterDataset
from model import UnderwaterRestorationModel
from losses import CombinedLoss
RAW_DIR = os.path.join(
    "data",
    "raw-890"
)
REFERENCE_DIR = os.path.join(
    "data",
    "reference-890"
)
MODEL_PATH = os.path.join(
    "models",
    "underwater_restoration.pth"
)
LAST_MODEL_PATH = os.path.join(
    "models",
    "underwater_restoration_last.pth"
)
INIT_FROM = None
IMAGE_SIZE = (256, 256)
EPOCHS = 200
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
BLUR_PROB = 0.5
MAX_BLUR_SIGMA = 3.0
NOISE_PROB = 0.5
MAX_NOISE = 5.0
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(
    "Using device:",
    device
)
if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )
names = list_pairs(
    RAW_DIR,
    REFERENCE_DIR
)
print(
    "Total paired images:",
    len(names)
)
if len(names) == 0:
    raise RuntimeError(
        "No paired images were found. "
        "Check data/raw-890 and data/reference-890."
    )
if len(names) != 890:
    print(
        "WARNING: Expected 890 pairs, "
        f"but found {len(names)} pairs."
    )
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
print(
    "Training batches:",
    len(train_loader)
)
model = UnderwaterRestorationModel().to(
    device
)
if INIT_FROM is not None:
    print(
        "Loading initial model:",
        INIT_FROM
    )
    model.load_state_dict(
        torch.load(
            INIT_FROM,
            map_location=device
        )
    )
print(
    "Total model parameters:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)
criterion = CombinedLoss(
    ssim_weight=0.2,
    perceptual_weight=0.05,
    edge_weight=0.4,
    color_weight=0.2,
    use_perceptual=torch.cuda.is_available()
).to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6
)
os.makedirs(
    "models",
    exist_ok=True
)
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
        optimizer.zero_grad()
        output = model(raw)
        loss = criterion(
            output,
            reference
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    train_loss = (
        total_loss /
        len(train_loader)
    )
    current_lr = (
        optimizer.param_groups[0]["lr"]
    )
    marker = ""
    if train_loss < best_loss:
        best_loss = train_loss
        torch.save(
            model.state_dict(),
            MODEL_PATH
        )
        marker = " <- best model saved"
    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {train_loss:.4f} | "
        f"LR: {current_lr:.7f}"
        f"{marker}"
    )
    torch.save(
        model.state_dict(),
        LAST_MODEL_PATH
    )
print(
    "\nTraining completed."
)
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
