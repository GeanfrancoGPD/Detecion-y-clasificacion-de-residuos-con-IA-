# src/app_streamlit.py
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json

st.title("Clasificador de Residuos")
model = tf.keras.models.load_model('../models/best_model.h5')
with open('../models/class_indices.json','r') as f:
    class_indices = json.load(f)
# invert mapping
idx2class = {v:k for k,v in class_indices.items()}

uploaded_file = st.file_uploader("Sube una imagen", type=['jpg','jpeg','png'])
if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Imagen subida', use_container_width =True)
    img_resized = img.resize((224,224))
    x = np.array(img_resized)/255.0
    x = np.expand_dims(x,0)
    preds = model.predict(x)[0]
    top_idx = preds.argmax()
    st.write(f"Predicción: **{idx2class[top_idx]}** con confianza {preds[top_idx]:.2f}")
    # mostrar top-3
    top3 = preds.argsort()[-3:][::-1]
    st.write("Top 3:")
    for i in top3:
        st.write(f"- {idx2class[i]}: {preds[i]:.3f}")
