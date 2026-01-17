import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os
from dotenv import load_dotenv
from google import genai

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
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

# --------------------------------------------------
# LOAD ENV
# --------------------------------------------------
load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# --------------------------------------------------
# SYSTEM PROMPT
# --------------------------------------------------
SYSTEM_PROMPT = """
You are an agricultural plant disease expert.

Explain the predicted crop disease briefly.
Mention disease type, symptoms, prevention, and treatment.
Use simple farmer-friendly language.
Do not claim full certainty.
If disease is severe or uncontrollable, advise consulting a plant pathologist.
Keep answers short and practical.
"""

# --------------------------------------------------
# GEMINI CHAT FUNCTION (FIXED)
# --------------------------------------------------
def gemini_chat(crop, disease, confidence):
    prompt = f"""
{SYSTEM_PROMPT}

Crop: {crop}
Disease: {disease}
Confidence: {confidence:.2f}%

Give short farmer-friendly advice.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={                     # ✅ CORRECT KEY
            "max_output_tokens": 700,
            "temperature": 0.4,
            "top_p": 0.9
        }
    )

    return response.text

# --------------------------------------------------
# TABS
# --------------------------------------------------
tabs1, tabs2, tabs3, tabs4, tabs5, tabs6 = st.tabs([
    "Cotton disease detection",
    "Maize disease detection",
    "Potato disease detection",
    "Rice disease detection",
    "Sugarcane disease detection",
    "Tomato disease detection"
])

# --------------------------------------------------
# RICE
# --------------------------------------------------
with tabs4:
    st.title("🌾 Rice Leaf Disease Detection")

    model_rice = tf.keras.models.load_model("rice_disease_model", compile=False)
    class_names_rice = ["BacterialBlight", "BrownSpot", "LeafSmut", "Blast", "Tungro"]

    uploaded_file = st.file_uploader(
        "Upload a rice leaf image",
        type=["jpg", "jpeg", "png"],
        key="rice"
    )

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, use_column_width=True)

        img = img.resize((128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model_rice.predict(img_array)
        idx = np.argmax(preds)
        confidence = preds[0][idx] * 100
        disease = class_names_rice[idx]

        st.success(f"🌱 Prediction: {disease}")
        st.info(f"🔍 Confidence: {confidence:.2f}%")

        st.markdown("### 🌾 AI Advisory")
        st.markdown(gemini_chat("Rice", disease, confidence))

# --------------------------------------------------
# COTTON
# --------------------------------------------------
with tabs1:
    st.title("🌿 Cotton Leaf Disease Detection")

    model = tf.keras.models.load_model("cotton_disease_model", compile=False)
    classes = ["Alternaria Leaf Spot", "Bacterial Blight", "Fusarium Wilt", "Verticillium Wilt"]

    uploaded_file = st.file_uploader(
        "Upload a cotton leaf image",
        type=["jpg", "jpeg", "png"],
        key="cotton"
    )

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, use_column_width=True)

        img = img.resize((128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        idx = np.argmax(preds)
        confidence = preds[0][idx] * 100
        disease = classes[idx]

        st.success(f"🌱 Prediction: {disease}")
        st.info(f"🔍 Confidence: {confidence:.2f}%")

        st.markdown("### 🌾 AI Advisory")
        st.markdown(gemini_chat("Cotton", disease, confidence))
