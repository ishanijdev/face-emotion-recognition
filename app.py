import streamlit as st
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# --- Page Configuration ---
st.set_page_config(
    page_title="EmotionAI - Real-Time Emotion Recognition",
    page_icon="🙂",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- Model and Cascade Loading ---
@st.cache_resource
def load_resources():
    """Loads the pre-trained model and face cascade classifier."""
    model = load_model('5_class_model.h5')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return model, face_cascade

model, face_cascade = load_resources()
CLASS_LABELS = ['angry', 'happy', 'neutral', 'sad', 'surprise']

# --- Core Prediction Function ---
def predict_emotion(image):
    """
    Detects a face, preprocesses it, and predicts the emotion.
    Returns the processed image with annotations and the predicted emotion.
    """
    img_copy = image.copy()
    gray_image = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return None, None

    (x, y, w, h) = faces[0]
    face_roi = gray_image[y:y+h, x:x+w]
    roi_resized = cv2.resize(face_roi, (48, 48))
    roi_normalized = roi_resized / 255.0
    roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))

    prediction = model.predict(roi_reshaped)
    label_index = np.argmax(prediction)
    emotion = CLASS_LABELS[label_index]
    
    cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img_copy, emotion.capitalize(), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return emotion, img_copy

# --- Custom CSS for Styling ---
# This injects custom CSS to match your original design's feel.
st.markdown("""
<style>
    .stApp {
        background-color: #F0F2F6; /* Light gray background */
    }
    .st-emotion-cache-18ni7ap { /* Main container */
        background-color: #FFFFFF;
        border-radius: 1rem;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .st-emotion-cache-16txtl3 { /* Headers */
        font-family: 'Inter', sans-serif;
        color: #333;
    }
    .stButton>button {
        background-color: #4B8BBE;
        color: white;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #3A6A94;
    }
</style>
""", unsafe_allow_html=True)

# --- Header Section ---
with st.container():
    st.markdown("<h1 style='text-align: center; font-weight: 700; color: #3A6A94;'>🙂 EmotionAI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #555;'>Real-Time Face Emotion Recognition</h3>", unsafe_allow_html=True)
    st.write(
        "Experience real-time emotion recognition using deep learning. "
        "Interact with your webcam or upload an image and explore how AI interprets human emotions."
    )
st.divider()

# --- Main Interaction Section ---
# Use columns to create the two main feature areas
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.header("🖼️ Upload Image")
    uploaded_file = st.file_uploader("Choose an image to analyze its emotion.", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        pil_image = Image.open(uploaded_file)
        opencv_image = np.array(pil_image.convert('RGB'))[:, :, ::-1] # Convert to BGR
        
        st.image(pil_image, caption='Uploaded Image', use_container_width=True)
        
        if st.button("Analyze Uploaded Image"):
            with st.spinner("Analyzing..."):
                emotion, result_img = predict_emotion(opencv_image)
                if emotion:
                    st.success(f"**Predicted Emotion: {emotion.capitalize()}**")
                    st.image(result_img, caption="Analysis Result", use_container_width=True)
                else:
                    st.error("No face detected. Please try another image.")

with col2:
    st.header("📷 Use Webcam")
    st.write("Click the button below to capture an image and detect emotion in real-time.")
    img_file_buffer = st.camera_input("Take a picture")
    
    if img_file_buffer:
        pil_image = Image.open(img_file_buffer)
        opencv_image = np.array(pil_image.convert('RGB'))[:, :, ::-1] # Convert to BGR
        
        with st.spinner("Analyzing..."):
            emotion, result_img = predict_emotion(opencv_image)
            if emotion:
                st.success(f"**Predicted Emotion: {emotion.capitalize()}**")
                st.image(result_img, caption="Analysis Result", use_container_width=True)
            else:
                st.error("No face detected. Please try again.")

st.divider()

# --- About the Project Section ---
with st.expander("💡 About the Project"):
    st.markdown("""
    This project is a smart system that detects and classifies human emotions from facial expressions. It was trained on the **FER2013 dataset** to recognize 5 emotions: **Happy, Sad, Angry, Surprise**, and **Neutral**.

    - **📷 Real-Time:** Works instantly with your webcam.
    - **📈 Accuracy:** Achieves 73.67% on the test dataset.
    - **🖼️ Upload Photo:** Detects emotion from any uploaded image containing a face.
    """)

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center;'>Created by Ishani Jindal & Mehar Bhanwra | Powered by FER2013</p>", unsafe_allow_html=True)