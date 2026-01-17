import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os
from dotenv import load_dotenv
from google import genai


st.set_page_config(page_title="AI Crop Disease Detection", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #dedbd2;
    }
    </style>
    """,
    unsafe_allow_html=True
)


load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

SYSTEM_PROMPT = """
You are an agricultural plant disease expert.

Explain the predicted crop disease shortly.
Mention disease type, symptoms, prevention, and treatment in simple language.
Keep answers short and practical.
Do not claim full certainty.
Give short answers only.
If the crop condition is is not control so suggest them to consult physically to plant pathologists.
"""

def gemini_chat(crop, disease, confidence):
    prompt = f"""
Crop: {crop}
Disease: {disease}
Confidence: {confidence:.2f}%

Give short farmer-friendly advice.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            {"role": "system", "parts": [{"text": SYSTEM_PROMPT}]},
            {"role": "user", "parts": [{"text": prompt}]}
        ],
        generation_config=types.GenerationConfig(
            max_output_tokens=700,
            temperature=0.4,
            top_p=0.9
        )
    )

    return response.text


tabs1, tabs2, tabs3, tabs4, tabs5, tabs6 = st.tabs([
    "Cotton disease detection",
    "Maize disease detection",
    "Potato disease detection",
    "Rice disease detection",
    "Sugarcane disease detection",
    "Tomato disease detection"
])

# ----------------- Rice Disease Tab -----------------
with tabs4:
    st.title("🌾 Rice Leaf Disease Detection")
    st.write("Upload a rice leaf image to detect the disease.")

    model_rice = tf.keras.models.load_model("rice_disease_model", compile=False)
    class_names_rice = ["BacterialBlight", "BrownSpot", "LeafSmut", "Blast", "Tungro"]

    uploaded_file_rice = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"], key="rice")
    if uploaded_file_rice is not None:
        img = Image.open(uploaded_file_rice)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        img = img.resize((128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model_rice.predict(img_array)
        index = np.argmax(predictions)
        confidence = predictions[0][index] * 100
        disease = class_names_rice[index]

        st.success(f"🌱 **Prediction: {disease}**")
        st.info(f"🔍 Confidence: {confidence:.2f}%")

        st.markdown("### 🌾 AI Advisory")
        st.markdown(gemini_chat("Rice", disease, confidence))

# ----------------- Cotton Disease Tab -----------------
with tabs1:
    st.title("🌿 Cotton Leaf Disease Detection")
    st.write("Upload a cotton leaf image to detect the disease.")

    model_cotton = tf.keras.models.load_model("cotton_disease_model", compile=False)
    class_names_cotton = ["Alternaria Leaf Spot", "Bacterial Blight", "Fusarium Wilt", "Verticillium Wilt"]

    uploaded_file_cotton = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"], key="cotton")
    if uploaded_file_cotton is not None:
        img = Image.open(uploaded_file_cotton)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        img = img.resize((128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model_cotton.predict(img_array)
        index = np.argmax(predictions)
        confidence = predictions[0][index] * 100
        disease = class_names_cotton[index]

        st.success(f"🌱 **Prediction: {disease}**")
        st.info(f"🔍 Confidence: {confidence:.2f}%")

        st.markdown("### 🌾 AI Advisory")
        st.markdown(gemini_chat("Cotton", disease, confidence))

# ----------------- Maize Disease Tab -----------------
with tabs2:
    st.title("🌿 Maize Leaf Disease Detection")
    st.write("Upload a maize leaf image to detect the disease.")

    model_maize = tf.keras.models.load_model("maize_disease_model", compile=False)
    class_names_maize = ["Blight", "Common_Rust", "Gray_Leaf_Spot"]

    uploaded_file_maize = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"], key="maize")
    if uploaded_file_maize is not None:
        img = Image.open(uploaded_file_maize)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        img = img.resize((128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model_maize.predict(img_array)
        index = np.argmax(predictions)
        confidence = predictions[0][index] * 100
        disease = class_names_maize[index]

        st.success(f"🌱 **Prediction: {disease}**")
        st.info(f"🔍 Confidence: {confidence:.2f}%")

        st.markdown("### 🌾 AI Advisory")
        st.markdown(gemini_chat("Maize", disease, confidence))
