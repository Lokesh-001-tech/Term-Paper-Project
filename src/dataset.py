import os
import random

import cv2
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


def load_image(path, image_size):
    image = Image.open(path).convert("RGB")
    image = image.resize(image_size, Image.Resampling.LANCZOS)
    return np.array(image, dtype=np.uint8)


class UnderwaterDataset(Dataset):
    """
    Loads all images once into memory (as uint8) so training is fast.

    augment=True  -> the SAME random flip / rotation is applied to the
                     raw and the reference image.

    degrade=True  -> the RAW image only is randomly blurred and/or made
                     noisy, while the reference (target) is left clean.
                     The model therefore learns to remove blur and noise
                     as well as fix colour. Use it for training only.
    """

    def __init__(self, raw_dir, reference_dir, names,
                 image_size=(256, 256), augment=False,
                 degrade=False, blur_prob=0.5, max_blur_sigma=3.0,
                 noise_prob=0.5, max_noise=5.0):

        self.augment = augment
        self.degrade = degrade
        self.blur_prob = blur_prob
        self.max_blur_sigma = max_blur_sigma
        self.noise_prob = noise_prob
        self.max_noise = max_noise

        self.raw_images = []
        self.reference_images = []

        for name in names:
            self.raw_images.append(
                load_image(os.path.join(raw_dir, name), image_size)
            )
            self.reference_images.append(
                load_image(os.path.join(reference_dir, name), image_size)
            )

        print(
            f"Loaded {len(names)} image pairs "
            f"(augment={augment}, degrade={degrade})"
        )

    def __len__(self):
        return len(self.raw_images)

    @staticmethod
    def _to_tensor(array):
        return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0

    def _degrade(self, array):
        """Random blur and noise on the raw image (a fresh random draw
        every time the image is loaded)."""

        if random.random() < self.blur_prob:
            sigma = random.uniform(0.5, self.max_blur_sigma)
            array = cv2.GaussianBlur(array, (0, 0), sigma)

        if random.random() < self.noise_prob:
            sigma = random.uniform(1.0, self.max_noise)
            noise = np.random.normal(0, sigma, array.shape)
            array = np.clip(
                array.astype(np.float32) + noise, 0, 255
            ).astype(np.uint8)

        return array

    def __getitem__(self, index):

        raw_array = self.raw_images[index]

        if self.degrade:
            raw_array = self._degrade(raw_array)

        raw = self._to_tensor(raw_array)
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
    print("Number of image pairs used for training:", len(names))