import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -----------------------------
# Page Title
# -----------------------------
st.title("CIFAR-10 Image Classification")
st.write("ResNet-18 Fine-Tuned Model (Transfer Learning)")

# -----------------------------
# CIFAR-10 Classes
# -----------------------------
classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load("resnet18_cifar10.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -----------------------------
# Image Preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize(96),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -----------------------------
# Image Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image_tensor = transform(image).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    st.success(f"Predicted Class: **{classes[predicted.item()]}**")
