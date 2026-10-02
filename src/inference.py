import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

from model import UnderwaterRestorationModel


# --------------------------------------------------
# Settings
# --------------------------------------------------
INPUT_DIR = os.path.join("test_data", "challenging-60")
MODEL_PATH = os.path.join("models", "underwater_restoration.pth")
OUTPUT_DIR = "outputs"

MODEL_SIZE = 256   # must match the size used in training


# --------------------------------------------------
# Command line: which images to restore
#   python src/inference.py 34.png
#   python src/inference.py 2.png 5.png 34.png
#   python src/inference.py            (all images in the folder)
# --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "images", nargs="*",
    help="image file names inside test_data/challenging-60"
)
parser.add_argument(
    "--post", action="store_true",
    help="apply a light CLAHE contrast boost after restoration"
)
args = parser.parse_args()

if not args.images:
    parser.error("Give at least one image name, e.g. python src\\inference.py 34.png")

image_names = args.images

# --------------------------------------------------
# Load model
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UnderwaterRestorationModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

print("Trained model loaded successfully.")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Restoration
# --------------------------------------------------
def restore(image):
    """
    The network runs at 256x256 (the size it was trained on). Instead of
    enlarging the blurry 256x256 result, we take the CORRECTION it
    predicted, enlarge only that, and add it to the ORIGINAL full-size
    image. This keeps all the original sharpness and resolution.
    """

    width, height = image.size

    small = image.resize((MODEL_SIZE, MODEL_SIZE), Image.Resampling.LANCZOS)
    small_tensor = to_tensor(small).unsqueeze(0).to(device)

    with torch.no_grad():
        restored_small = model(small_tensor).clamp(0, 1)

    correction = restored_small - small_tensor

    correction = F.interpolate(
        correction, size=(height, width),
        mode="bicubic", align_corners=False
    )

    full_tensor = to_tensor(image).unsqueeze(0).to(device)

    result = (full_tensor + correction).clamp(0, 1)

    return to_pil_image(result.squeeze(0).cpu())


def clahe_boost(image):
    """Optional: local contrast enhancement on the lightness channel."""

    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    lab = cv2.merge([l_channel, a_channel, b_channel])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


# --------------------------------------------------
# Process every requested image
# --------------------------------------------------
for name in image_names:

    input_path = os.path.join(INPUT_DIR, name)

    if not os.path.exists(input_path):
        print("Not found, skipping:", input_path)
        continue

    image = Image.open(input_path).convert("RGB")

    restored = restore(image)

    if args.post:
        restored = clahe_boost(restored)

    stem = os.path.splitext(name)[0]

    restored_path = os.path.join(OUTPUT_DIR, f"{stem}_restored.png")
    compare_path = os.path.join(OUTPUT_DIR, f"{stem}_compare.png")

    restored.save(restored_path)

    # Side-by-side picture: original | restored
    comparison = Image.new("RGB", (image.width * 2, image.height))
    comparison.paste(image, (0, 0))
    comparison.paste(restored, (image.width, 0))
    comparison.save(compare_path)

    print(f"{name}: saved {restored_path} and {compare_path}")

print("Done.")