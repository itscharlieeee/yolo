import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import platform

# --- CONFIGURACIÓN GENERAL ---
st.set_page_config(
    page_title="Reconocimiento de Imágenes IA",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- ENCABEZADO PRINCIPAL ---
st.markdown(
    """
    <h1 style='text-align:center; color:#4A90E2;'>Reconocimiento de Imágenes con IA</h1>
    <p style='text-align:center; font-size:17px;'>
    Usa un modelo entrenado en <b>Teachable Machine</b> para identificar objetos o gestos desde tu cámara.
    </p>
    """,
    unsafe_allow_html=True
)

# --- INFO DEL SISTEMA ---
st.caption(f"🧩 Versión de Python: {platform.python_version()}")

# --- CARGA DEL MODELO ---
with st.spinner("Cargando modelo IA..."):
    model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# --- IMAGEN ILUSTRATIVA ---
image = Image.open('2a9847d15807e6bc8037c57afa472967.jpg')
st.image(image, width=300, caption="Modelo entrenado en Teachable Machine")

# --- SIDEBAR ---
with st.sidebar:
    st.header("📋 Instrucciones")
    st.write("""
    1️⃣ Presiona el botón para tomar una foto.  
    2️⃣ La IA procesará la imagen.  
    3️⃣ Observa la predicción con su probabilidad.
    """)
    st.info("Asegúrate de tener buena iluminación para mejores resultados ✨")

# --- ENTRADA DE CÁMARA ---
st.markdown("<h3 style='text-align:center;'>📸 Toma una foto para analizar</h3>", unsafe_allow_html=True)
img_file_buffer = st.camera_input("Haz clic para capturar")

# --- PROCESAMIENTO DE LA IMAGEN ---
if img_file_buffer is not None:
    with st.spinner("🔍 Analizando imagen..."):
        img = Image.open(img_file_buffer)

        # Redimensionar imagen
        newsize = (224, 224)
        img = img.resize(newsize)

        # Convertir a array y normalizar
        img_array = np.array(img)
        normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array

        # --- PREDICCIÓN ---
        prediction = model.predict(data)

    # --- RESULTADOS ---
    st.success("✅ Análisis completado")
    st.markdown("### 📊 Resultado del análisis:")

    if prediction[0][0] > 0.5:
        st.markdown(
            f"""
            <div style='background-color:#E8F6EF; padding:12px; border-radius:12px;'>
            <b>Predicción:</b> Movimiento a la izquierda 🫲  
            <b>Probabilidad:</b> {prediction[0][0]:.2f}
            </div>
            """,
            unsafe_allow_html=True
        )

    elif prediction[0][1] > 0.5:
        st.markdown(
            f"""
            <div style='background-color:#E3F2FD; padding:12px; border-radius:12px;'>
            <b>Predicción:</b> Movimiento hacia arriba 👆  
            <b>Probabilidad:</b> {prediction[0][1]:.2f}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.warning("🤔 No se detectó una categoría clara. Intenta otra posición o iluminación.")

    st.balloons()

else:
    st.info("📷 Captura una foto para comenzar el reconocimiento.")
