import streamlit as st
from tensorflow.keras.models import load_model
from utils import load_and_preprocess_image, make_gradcam_heatmap, overlay_heatmap
import numpy as np
import os
from PIL import Image
import cv2

st.title("Cat vs Dog Classifier with Grad-CAM")

model = load_model("model/cat_dog_model.h5")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.image(file_path, caption="Uploaded Image", use_column_width=True)

    img_array = load_and_preprocess_image(file_path)
    pred = model.predict(img_array)[0][0]
    label = "Dog " if pred > 0.5 else "Cat "
    st.markdown(f"### Prediction: **{label}** ({pred*100:.2f}%)")

    heatmap = make_gradcam_heatmap(img_array, model)
    overlayed = overlay_heatmap(file_path, heatmap)
    st.image(overlayed, caption="Grad-CAM", use_column_width=True)

    os.remove(file_path)
