import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class UnderwaterDataset(Dataset):

    def __init__(self, raw_dir, reference_dir):

        self.raw_dir = raw_dir
        self.reference_dir = reference_dir

        # Get image filenames from raw folder
        self.image_names = [
            file for file in os.listdir(raw_dir)
            if file.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):

        image_name = self.image_names[index]

        raw_path = os.path.join(
            self.raw_dir,
            image_name
        )

        reference_path = os.path.join(
            self.reference_dir,
            image_name
        )

        # Check whether reference image exists
        if not os.path.exists(reference_path):
            raise FileNotFoundError(
                f"Reference image not found: {image_name}"
            )

        # Open images
        raw_image = Image.open(raw_path).convert("RGB")
        reference_image = Image.open(reference_path).convert("RGB")

        # Apply preprocessing
        raw_image = self.transform(raw_image)
        reference_image = self.transform(reference_image)

        return raw_image, reference_image


if __name__ == "__main__":

    # Paths relative to project root
    raw_dir = os.path.join("data", "raw-890")
    reference_dir = os.path.join("data", "reference-890")

    dataset = UnderwaterDataset(
        raw_dir,
        reference_dir
    )

    print("Number of image pairs:", len(dataset))

    raw_image, reference_image = dataset[0]

    print("Raw image shape:", raw_image.shape)
    print("Reference image shape:", reference_image.shape)