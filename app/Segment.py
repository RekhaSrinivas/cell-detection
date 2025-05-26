import streamlit as st
import os
import torch
from PIL import Image
from ultralytics import YOLO
import tempfile
import io  # for BytesIO

# Ensure that the KMP_DUPLICATE_LIB_OK environment variable is set to avoid issues with OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

st.set_page_config(page_title="Cell Segmentation", page_icon="🧬", layout="centered")
st.title("🧬 Cell Segmentation with YOLOv8")

uploaded_files = st.file_uploader("Upload image(s)", type=["jpg", "png"], accept_multiple_files=True)
confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
model_path = st.selectbox("Select Model", ["models/yolov8n-seg.pt", "runs/detect/train/weights/best.pt"])

if st.button("Run Segmentation") and uploaded_files:
    model = YOLO(model_path)
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file).convert("RGB")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp.name)
            results = model.predict(source=tmp.name, conf=confidence, save=False)
            annotated_np = results[0].plot()  # NumPy array
            annotated_img = Image.fromarray(annotated_np)

            # Convert to bytes
            buf = io.BytesIO()
            annotated_img.save(buf, format="JPEG")
            byte_img = buf.getvalue()

            # Display and Download
            st.image([img, annotated_img], caption=["Original", "Segmented"], use_container_width=True)
            st.download_button("Download Mask", byte_img, file_name="mask.jpg", mime="image/jpeg")