import streamlit as st
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# --- Page Configuration ---
# This should be the first Streamlit command in your script
st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="🙂",
    layout="wide", # Use "wide" layout for a more app-like feel
    initial_sidebar_state="auto",
)

# --- Model Loading ---
# Using st.cache_resource ensures the model is loaded only once
@st.cache_resource
def load_emotion_model():
    """Loads the pre-trained model and face cascade classifier."""
    model = load_model('5_class_model.h5') # Ensure this path is correct
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return model, face_cascade

model, face_cascade = load_emotion_model()
CLASS_LABELS = ['angry', 'happy', 'neutral', 'sad', 'surprise']

# --- Core Prediction Function ---
def predict_emotion(image):
    """
    Detects a face in an image, preprocesses it, and predicts the emotion.
    Returns the processed image with annotations and the predicted emotion.
    """
    # Create a copy to avoid modifying the original image
    img_copy = image.copy()
    gray_image = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return None, None # No face was detected

    # Process the first detected face
    (x, y, w, h) = faces[0]
    face_roi = gray_image[y:y+h, x:x+w]
    
    # Preprocess for the model
    roi_resized = cv2.resize(face_roi, (48, 48))
    roi_normalized = roi_resized / 255.0
    roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))

    # Make prediction
    prediction = model.predict(roi_reshaped)
    label_index = np.argmax(prediction)
    emotion = CLASS_LABELS[label_index]
    
    # Annotate the image
    cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img_copy, emotion.capitalize(), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return emotion, img_copy

# --- Streamlit App Interface ---
# Main title of the application
st.title("🙂 Facial Emotion Recognition Web App")
st.write("This application leverages a deep learning model to detect emotions from faces in real-time or from an uploaded image.")

st.divider()

# Create two columns for the layout
col1, col2 = st.columns(2)

# --- Column 1: Image Upload Feature ---
with col1:
    st.header("🖼️ Analyze an Image")
    uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        opencv_image = np.array(pil_image.convert('RGB'))
        # Convert RGB to BGR for OpenCV
        opencv_image = opencv_image[:, :, ::-1].copy()

        st.image(pil_image, caption='Your Uploaded Image', use_container_width=True)
        
        if st.button('Analyze Image'):
            with st.spinner('Processing...'):
                emotion, result_image = predict_emotion(opencv_image)
                if emotion:
                    st.success(f"**Predicted Emotion: {emotion.capitalize()}**")
                    st.image(result_image, caption='Analysis Result', use_container_width=True)
                else:
                    st.error("No face was detected. Please try a different image.")

# --- Column 2: Webcam Feature ---
with col2:
    st.header("📷 Use Live Webcam")
    img_file_buffer = st.camera_input("Take a picture to analyze")

    if img_file_buffer is not None:
        pil_image = Image.open(img_file_buffer)
        opencv_image = np.array(pil_image.convert('RGB'))
        # Convert RGB to BGR for OpenCV
        opencv_image = opencv_image[:, :, ::-1].copy()

        with st.spinner('Processing...'):
            emotion, result_image = predict_emotion(opencv_image)
            if emotion:
                st.success(f"**Predicted Emotion: {emotion.capitalize()}**")
                st.image(result_image, caption='Analysis Result', use_container_width=True)
            else:
                st.error("No face was detected. Please try again.")