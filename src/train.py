import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import UnderwaterDataset
from model import UnderwaterRestorationModel


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# Dataset paths
raw_dir = os.path.join("data", "raw-890")
reference_dir = os.path.join("data", "reference-890")


# Load dataset
dataset = UnderwaterDataset(
    raw_dir,
    reference_dir
)


# DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)


# Model
model = UnderwaterRestorationModel().to(device)


# Loss function
criterion = nn.MSELoss()


# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# Number of epochs
epochs = 50


# Training
for epoch in range(epochs):

    model.train()

    total_loss = 0.0

    for raw_images, reference_images in dataloader:

        raw_images = raw_images.to(device)
        reference_images = reference_images.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(raw_images)

        # Calculate loss
        loss = criterion(
            output,
            reference_images
        )

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)

    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Loss: {average_loss:.6f}"
    )


# Save trained model
os.makedirs("models", exist_ok=True)

torch.save(
    model.state_dict(),
    "models/underwater_restoration.pth"
)

print("Training completed.")
print("Model saved successfully.")