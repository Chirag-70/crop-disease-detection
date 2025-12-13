import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Create tabs
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

    # Load model
    model_rice = tf.keras.models.load_model("rice_disease_model", compile=False)

    # Classes
    class_names_rice = ["BacterialBlight", "BrownSpot", "LeafSmut", "Blast", "Tungro"]

    # File uploader
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

        st.success(f"🌱 **Prediction: {class_names_rice[index]}**")
        st.info(f"🔍 Confidence: {confidence:.2f}%")

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

        st.success(f"🌱 **Prediction: {class_names_cotton[index]}**")
        st.info(f"🔍 Confidence: {confidence:.2f}%")

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

        st.success(f"🌱 **Prediction: {class_names_maize[index]}**")
        st.info(f"🔍 Confidence: {confidence:.2f}%")

# ----------------- Potato Disease Tab -----------------
with tabs3:
    st.title("🌿 Potato Leaf Disease Detection")
    st.write("Upload a potato leaf image to detect the disease.")

    model_potato = tf.keras.models.load_model("potato_disease_model", compile=False)
    class_names_potato = ["Early_Blight", "Late_Blight"]

    uploaded_file_potato = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"], key="potato")
    if uploaded_file_potato is not None:
        img = Image.open(uploaded_file_potato)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        img = img.resize((128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model_potato.predict(img_array)
        index = np.argmax(predictions)
        confidence = predictions[0][index] * 100

        st.success(f"🌱 **Prediction: {class_names_potato[index]}**")
        st.info(f"🔍 Confidence: {confidence:.2f}%")

# ----------------- Sugarcane Disease Tab -----------------
with tabs5:
    st.title("🌿 Sugarcane Leaf Disease Detection")
    st.write("Upload a sugarcane leaf image to detect the disease.")

    model_sugarcane = tf.keras.models.load_model("sugarcane_disease_model", compile=False)
    class_names_sugarcane = ["Brown Spot", "BrownRust", "Grassy Shoot", "Viral Disease", "Yellow Leaf"]

    uploaded_file_sugarcane = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"], key="sugarcane")
    if uploaded_file_sugarcane is not None:
        img = Image.open(uploaded_file_sugarcane)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        img = img.resize((128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model_sugarcane.predict(img_array)
        index = np.argmax(predictions)
        confidence = predictions[0][index] * 100

        st.success(f"🌱 **Prediction: {class_names_sugarcane[index]}**")
        st.info(f"🔍 Confidence: {confidence:.2f}%")

# ----------------- Tomato Disease Tab -----------------
with tabs6:
    st.title("🌿 Tomato Leaf Disease Detection")
    st.write("Upload a tomato leaf image to detect the disease.")

    model_tomato = tf.keras.models.load_model("tomato_disease_model", compile=False)
    class_names_tomato = ["Bacterial_Spot", "Early_Blight", "Late_Blight", "Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus"]

    uploaded_file_tomato = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"], key="tomato")
    if uploaded_file_tomato is not None:
        img = Image.open(uploaded_file_tomato)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        img = img.resize((128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model_tomato.predict(img_array)
        index = np.argmax(predictions)
        confidence = predictions[0][index] * 100

        st.success(f"🌱 **Prediction: {class_names_tomato[index]}**")
        st.info(f"🔍 Confidence: {confidence:.2f}%")
