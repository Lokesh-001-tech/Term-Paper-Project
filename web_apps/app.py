import streamlit as st
import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from torchvision.transforms.functional import to_tensor, to_pil_image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
from model import UnderwaterRestorationModel

MODEL_PATH = os.path.join(BASE_DIR, "models", "underwater_restoration.pth")
MODEL_SIZE = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="Underwater Image Restoration", layout="wide", page_icon="🌊")

@st.cache_resource
def load_model():
    model = UnderwaterRestorationModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def clahe_boost(image):
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge([l_channel, a_channel, b_channel])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def restore(image, model):
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

st.title("🌊 Underwater Image Restoration")
st.write("Upload a raw underwater image to enhance its quality, colors, and contrast using our deep learning model.")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.sidebar.header("Settings")
apply_clahe = st.sidebar.checkbox("Apply CLAHE Contrast Boost", value=False)

uploaded_file = st.file_uploader("Choose an underwater image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    upload_dir = os.path.join(BASE_DIR, "test_data", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    save_path = os.path.join(upload_dir, uploaded_file.name)
    
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = Image.open(save_path).convert("RGB")
    
    with st.spinner("Restoring image..."):
        restored_img = restore(image, model)
        if apply_clahe:
            restored_img = clahe_boost(restored_img)
            
        output_dir = os.path.join(BASE_DIR, "outputs", "uploads")
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(uploaded_file.name)[0]
        restored_save_path = os.path.join(output_dir, f"{base_name}_restored.png")
        restored_img.save(restored_save_path)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Raw Image")
        st.image(image, use_container_width=True)
        st.success(f"Saved to: `{save_path}`")
    with col2:
        st.subheader("Restored Image")
        st.image(restored_img, use_container_width=True)
        st.success(f"Saved to: `{restored_save_path}`")
