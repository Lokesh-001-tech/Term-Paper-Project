import os
import sys
import io
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from torchvision.transforms.functional import to_tensor, to_pil_image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import Response, HTMLResponse
import uvicorn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
from model import UnderwaterRestorationModel

MODEL_PATH = os.path.join(BASE_DIR, "models", "underwater_restoration.pth")
MODEL_SIZE = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UnderwaterRestorationModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

app = FastAPI()

def clahe_boost(image):
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge([l_channel, a_channel, b_channel])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def restore(image):
    width, height = image.size
    small = image.resize((MODEL_SIZE, MODEL_SIZE), Image.Resampling.LANCZOS)
    small_tensor = to_tensor(small).unsqueeze(0).to(device)

    with torch.no_grad():
        restored_small = model(small_tensor).clamp(0, 1)

    correction = restored_small - small_tensor
    correction = F.interpolate(correction, size=(height, width), mode="bicubic", align_corners=False)

    full_tensor = to_tensor(image).unsqueeze(0).to(device)
    result = (full_tensor + correction).clamp(0, 1)

    return to_pil_image(result.squeeze(0).cpu())

@app.post("/api/restore")
async def process_image(file: UploadFile = File(...), apply_clahe: bool = Form(False)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    restored_img = restore(image)
    if apply_clahe:
        restored_img = clahe_boost(restored_img)
        
    img_byte_arr = io.BytesIO()
    restored_img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    return Response(content=img_byte_arr, media_type="image/png")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    print("\n" + "="*50)
    print("FastAPI Server is starting!")
    print("Open this link in your browser: http://127.0.0.1:8000")
    print("="*50 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
