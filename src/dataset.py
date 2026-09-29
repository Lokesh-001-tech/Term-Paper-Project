import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def list_pairs(raw_dir, reference_dir):
    """Returns filenames that exist in both folders (checks every pair)."""

    names = sorted(
        file for file in os.listdir(raw_dir)
        if file.lower().endswith(IMAGE_EXTENSIONS)
    )

    missing = [
        name for name in names
        if not os.path.exists(os.path.join(reference_dir, name))
    ]

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} raw images have no reference image, "
            f"for example: {missing[:5]}"
        )

    return names


def split_names(names, val_ratio=0.1, seed=42):
    """Fixed random split so validation images are never trained on."""

    names = list(names)
    random.Random(seed).shuffle(names)

    val_count = max(1, int(len(names) * val_ratio))

    return names[val_count:], names[:val_count]


def load_image(path, image_size):
    image = Image.open(path).convert("RGB")
    image = image.resize(image_size, Image.Resampling.LANCZOS)
    return np.array(image, dtype=np.uint8)


class UnderwaterDataset(Dataset):
    """
    Loads all images once into memory (as uint8) so training is fast.
    augment=True applies the SAME random flip / rotation to raw and
    reference, which effectively multiplies the small dataset.
    """

    def __init__(self, raw_dir, reference_dir, names,
                 image_size=(256, 256), augment=False):

        self.augment = augment

        self.raw_images = []
        self.reference_images = []

        for name in names:
            self.raw_images.append(
                load_image(os.path.join(raw_dir, name), image_size)
            )
            self.reference_images.append(
                load_image(os.path.join(reference_dir, name), image_size)
            )

        print(f"Loaded {len(names)} image pairs (augment={augment})")

    def __len__(self):
        return len(self.raw_images)

    @staticmethod
    def _to_tensor(array):
        return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, index):

        raw = self._to_tensor(self.raw_images[index])
        reference = self._to_tensor(self.reference_images[index])

        if self.augment:

            if random.random() < 0.5:
                raw = torch.flip(raw, dims=[2])
                reference = torch.flip(reference, dims=[2])

            if random.random() < 0.5:
                raw = torch.flip(raw, dims=[1])
                reference = torch.flip(reference, dims=[1])

            k = random.randint(0, 3)
            if k:
                raw = torch.rot90(raw, k, dims=[1, 2])
                reference = torch.rot90(reference, k, dims=[1, 2])

        return raw, reference


if __name__ == "__main__":

    raw_dir = os.path.join("data", "raw-890")
    reference_dir = os.path.join("data", "reference-890")

    names = list_pairs(raw_dir, reference_dir)
    print("Number of image pairs:", len(names))

    train_names, val_names = split_names(names)
    print("Train:", len(train_names), "Validation:", len(val_names))