import os
import torch
from PIL import Image
from torchvision import transforms

from model import UnderwaterRestorationModel


# --------------------------------------------------
# 1. Select device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# --------------------------------------------------
# 2. Define paths
# --------------------------------------------------

input_path = os.path.join(
    "test_data",
    "challenging-60",
    "2.png"
)

model_path = os.path.join(
    "models",
    "underwater_restoration.pth"
)

output_dir = "outputs"

os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(
    output_dir,
    "2_restored.png"
)


# --------------------------------------------------
# 3. Load the trained model
# --------------------------------------------------

model = UnderwaterRestorationModel()

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Trained model loaded successfully.")


# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# --------------------------------------------------
# 5. Load input image
# --------------------------------------------------

image = Image.open(input_path).convert("RGB")

print("Original image size:", image.size)


# --------------------------------------------------
# 6. Apply preprocessing
# --------------------------------------------------

input_tensor = transform(image)

input_tensor = input_tensor.unsqueeze(0)

input_tensor = input_tensor.to(device)

print("Input tensor shape:", input_tensor.shape)


# --------------------------------------------------
# 7. Perform inference
# --------------------------------------------------

with torch.no_grad():

    restored_image = model(input_tensor)


# --------------------------------------------------
# 8. Convert output tensor to image
# --------------------------------------------------

restored_image = restored_image.squeeze(0)

restored_image = restored_image.cpu()

restored_image = transforms.ToPILImage()(restored_image)


# --------------------------------------------------
# 9. Save restored image
# --------------------------------------------------

restored_image.save(output_path)

print("Restored image saved successfully.")
print("Output path:", output_path)