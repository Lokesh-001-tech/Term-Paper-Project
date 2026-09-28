import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Dataset paths
# -----------------------------
RAW_DIR = "data/raw-890"
REFERENCE_DIR = "data/reference-890"


# -----------------------------
# Image size
# -----------------------------
IMAGE_SIZE = (256, 256)


# -----------------------------
# Underwater Image Dataset
# -----------------------------
class UnderwaterDataset(Dataset):

    def __init__(self, raw_dir, reference_dir, image_size=(256, 256)):

        self.raw_dir = raw_dir
        self.reference_dir = reference_dir
        self.image_size = image_size

        # Get image filenames
        self.raw_images = sorted([
            file for file in os.listdir(raw_dir)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        self.reference_images = sorted([
            file for file in os.listdir(reference_dir)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        # Check number of images
        print("Number of raw images:", len(self.raw_images))
        print("Number of reference images:", len(self.reference_images))

        if len(self.raw_images) != len(self.reference_images):
            raise ValueError(
                "Raw and reference image counts are different."
            )

    def __len__(self):
        return len(self.raw_images)

    def __getitem__(self, index):

        # Get filenames
        raw_name = self.raw_images[index]
        reference_name = self.reference_images[index]

        # Complete paths
        raw_path = os.path.join(self.raw_dir, raw_name)
        reference_path = os.path.join(
            self.reference_dir, reference_name
        )

        # Read images
        raw_image = cv2.imread(raw_path)
        reference_image = cv2.imread(reference_path)

        # Check image loading
        if raw_image is None:
            raise ValueError(
                f"Could not read raw image: {raw_path}"
            )

        if reference_image is None:
            raise ValueError(
                f"Could not read reference image: {reference_path}"
            )

        # OpenCV uses BGR.
        # Convert BGR to RGB.
        raw_image = cv2.cvtColor(
            raw_image,
            cv2.COLOR_BGR2RGB
        )

        reference_image = cv2.cvtColor(
            reference_image,
            cv2.COLOR_BGR2RGB
        )

        # Resize images
        raw_image = cv2.resize(
            raw_image,
            self.image_size
        )

        reference_image = cv2.resize(
            reference_image,
            self.image_size
        )

        # Convert pixel values from 0-255 to 0-1
        raw_image = raw_image.astype("float32") / 255.0
        reference_image = reference_image.astype("float32") / 255.0

        # Convert from:
        # Height x Width x Channels
        #
        # to:
        # Channels x Height x Width
        raw_image = torch.from_numpy(
            raw_image
        ).permute(2, 0, 1)

        reference_image = torch.from_numpy(
            reference_image
        ).permute(2, 0, 1)

        return raw_image, reference_image


# -----------------------------
# Create dataset
# -----------------------------
dataset = UnderwaterDataset(
    RAW_DIR,
    REFERENCE_DIR,
    IMAGE_SIZE
)


# -----------------------------
# Create DataLoader
# -----------------------------
dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)


# -----------------------------
# Test the preprocessing
# -----------------------------
if __name__ == "__main__":

    print("\nTesting preprocessing...\n")

    raw_images, reference_images = next(
        iter(dataloader)
    )

    print("Raw batch shape:", raw_images.shape)
    print("Reference batch shape:", reference_images.shape)

    print(
        "Raw pixel range:",
        raw_images.min().item(),
        "to",
        raw_images.max().item()
    )

    print(
        "Reference pixel range:",
        reference_images.min().item(),
        "to",
        reference_images.max().item()
    )

    print("\nPreprocessing completed successfully.")