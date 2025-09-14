import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Use Streamlit's caching to load the model only once.
@st.cache_resource
def load_emotion_model():
    """Loads the pre-trained emotion detection model and face cascade."""
    model = load_model('5_class_model.h5')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return model, face_cascade

# Define the labels for the emotions
CLASS_LABELS = ['angry', 'happy', 'neutral', 'sad', 'surprise']

# Load the model and cascade classifier
model, face_cascade = load_emotion_model()


# --- Streamlit User Interface ---
st.set_page_config(page_title="Face Emotion Recognition", layout="centered")
st.title("🙂 Face Emotion Recognition")
st.write("Upload an image to see the predicted emotion. The model will detect the first face found.")

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    pil_image = Image.open(uploaded_file)
    # Convert the PIL image to an OpenCV image (NumPy array)
    opencv_image = np.array(pil_image.convert('RGB'))
    # Convert RGB to BGR for OpenCV
    opencv_image = opencv_image[:, :, ::-1].copy()

    # Display the uploaded image
    st.image(pil_image, caption='Uploaded Image.', use_column_width=True)
    
    # Add a button to trigger prediction
    if st.button('Detect Emotion'):
        # Convert the image to grayscale for face detection
        gray_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            st.error("No face was detected in the image. Please try another one.")
        else:
            # Process the first detected face
            (x, y, w, h) = faces[0]
            
            # Extract the region of interest (the face)
            roi_gray = gray_image[y:y+h, x:x+w]
            
            # Preprocess the face for the model
            roi_resized = cv2.resize(roi_gray, (48, 48))
            roi_normalized = roi_resized / 255.0
            roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))

            # Make a prediction
            with st.spinner('Analyzing emotion...'):
                prediction = model.predict(roi_reshaped)
                label_index = np.argmax(prediction)
                emotion = CLASS_LABELS[label_index]

                st.success(f"**Predicted Emotion: {emotion.capitalize()}**")