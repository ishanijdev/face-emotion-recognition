import streamlit as st
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from gtts import gTTS
import base64
from io import BytesIO

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EmotionAI",
    page_icon="🙂",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. CUSTOM STYLING (from your screenshots) ---
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

body {
    font-family: 'Inter', sans-serif;
}

/* Main container and styling */
.stApp {
    background-color: #F0F4F8; /* Light blue-gray background */
}

/* Hide Streamlit's default header and footer */
header, footer {
    visibility: hidden;
}

/* Custom header */
.custom-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 2rem;
    background-color: white;
    border-bottom: 1px solid #E6EAF1;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
}
.logo-text {
    font-weight: 700;
    font-size: 1.7rem;
    color: #3A6A94;
}
.nav-links a {
    margin: 0 15px;
    color: #555;
    text-decoration: none;
    font-weight: 600;
}

/* Card-like containers for sections */
.section-container {
    background-color: white;
    border-radius: 1rem;
    padding: 2.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

/* Feature boxes in the "About" section */
.feature-box {
    background-color: #E6F0F8;
    border-radius: 0.75rem;
    padding: 1rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.feature-box span {
    font-size: 1.5rem;
}

/* Main interaction buttons */
.stButton>button {
    background: linear-gradient(45deg, #4B8BBE, #3A6A94);
    color: white;
    border-radius: 0.5rem;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    border: none;
    transition: transform 0.2s, box-shadow 0.2s;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}
.stButton>button:hover {
    transform: scale(1.03);
    box-shadow: 0 6px 15px rgba(0,0,0,0.15);
}
"""
st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)


# --- 3. MODEL AND RESOURCE LOADING ---
@st.cache_resource
def load_resources():
    model = load_model('5_class_model.h5')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return model, face_cascade

model, face_cascade = load_resources()
CLASS_LABELS = ['angry', 'happy', 'neutral', 'sad', 'surprise']


# --- 4. HELPER FUNCTIONS ---
def text_to_audio_autoplay(text):
    try:
        tts = gTTS(text=text, lang='en')
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        b64 = base64.b64encode(mp3_fp.read()).decode()
        audio_html = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        print(f"Error generating audio: {e}")

def predict_emotion(image):
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
    cv2.rectangle(img_copy, (x, y), (x+w, y+h), (75, 139, 190), 3)
    cv2.putText(img_copy, emotion.capitalize(), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (75, 139, 190), 2)
    return emotion, img_copy

# --- 5. UI RENDERING ---

# Custom Header
st.markdown("""
<div class="custom-header">
    <div class="logo-text">🙂 EmotionAI</div>
    <div class="nav-links">
        <a href="#about-the-project">About</a>
        <a href="#emotions-recognized-by-emotionai">Emotions</a>
        <a href="#contact-us">Contact</a>
    </div>
</div>
""", unsafe_allow_html=True)


# Main Hero Section
st.title("Real-Time Face Emotion Recognition")
st.markdown("Experience real-time emotion recognition using deep learning. Interact with your webcam or upload an image and explore how AI interprets human emotions.")
audio_feedback = st.toggle("🔈 Audio Feedback", value=True)
st.divider()

# Main Interaction Tabs
tab1, tab2 = st.tabs(["🖼️ Upload Image", "🎥 Use Webcam"])

with tab1:
    st.header("Upload an Image to Detect Emotion")
    uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded_file:
        pil_image = Image.open(uploaded_file)
        opencv_image = np.array(pil_image.convert('RGB'))[:, :, ::-1]
        st.image(pil_image, caption='Your Uploaded Image', use_container_width=True)
        if st.button("Analyze Image", use_container_width=True):
            with st.spinner("Analyzing..."):
                emotion, result_img = predict_emotion(opencv_image)
                if emotion:
                    st.success(f"**Predicted Emotion: {emotion.capitalize()}**")
                    st.image(result_img, use_container_width=True)
                    if audio_feedback: text_to_audio_autoplay(f"The detected emotion is {emotion}")
                else:
                    st.error("No face detected.")

with tab2:
    st.header("Use Your Webcam for Real-time Prediction")
    img_file_buffer = st.camera_input("Take a picture", label_visibility="collapsed")
    if img_file_buffer:
        pil_image = Image.open(img_file_buffer)
        opencv_image = np.array(pil_image.convert('RGB'))[:, :, ::-1]
        with st.spinner("Analyzing..."):
            emotion, result_img = predict_emotion(opencv_image)
            if emotion:
                st.success(f"**Predicted Emotion: {emotion.capitalize()}**")
                st.image(result_img, use_container_width=True)
                if audio_feedback: text_to_audio_autoplay(f"The detected emotion is {emotion}")
            else:
                st.error("No face detected.")

st.divider()

# About Section
with st.container():
    st.markdown("<div class='section-container'>", unsafe_allow_html=True)
    st.markdown("<a id='about-the-project'></a>", unsafe_allow_html=True) # Anchor for navigation
    st.header("💡 About the Project")
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.write("""
        This project is a smart system that detects and classifies human emotions from facial expressions in real-time. It uses the **FER2013 dataset** to detect 5 emotions: **Happy, Sad, Angry, Surprise**, and **Neutral**.
        Whether for mental health, user experience, or just curiosity, this tool makes emotion-aware interaction simple and insightful!
        """)
    with col2:
        st.markdown("""
        <div class="feature-box"><span>📷</span> <div><strong>Real-Time</strong> — Works instantly with your webcam.</div></div>
        <div class="feature-box"><span>📈</span> <div><strong>Accuracy</strong> — Achieves 73.67% on FER2013.</div></div>
        <div class="feature-box"><span>🔊</span> <div><strong>Audio Feedback</strong> — Spoken output of your emotion.</div></div>
        <div class="feature-box"><span>🖼️</span> <div><strong>Upload Photo</strong> — Detect emotion from an uploaded image.</div></div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# Emotions Gallery
with st.container():
    st.markdown("<div class='section-container'>", unsafe_allow_html=True)
    st.markdown("<a id='emotions-recognized-by-emotionai'></a>", unsafe_allow_html=True)
    st.header("🔍 Emotions Recognized by EmotionAI")
    
    # IMPORTANT: Create a 'static' folder in your repo and place these images inside.
    image_files = {
        "Angry": "static/angry.jpg",
        "Happy": "static/happy.jpg",
        "Sad": "static/sad.jpg",
        "Neutral": "static/neutral.jpg",
        "Surprise": "static/surprise.jpg"
    }
    
    cols = st.columns(5)
    for i, (emotion, path) in enumerate(image_files.items()):
        try:
            with cols[i]:
                st.image(path, caption=emotion)
        except FileNotFoundError:
            with cols[i]:
                st.warning(f"Image not found: {path}")

    st.markdown("</div>", unsafe_allow_html=True)


# Contact Section
with st.container():
    st.markdown("<div class='section-container'>", unsafe_allow_html=True)
    st.markdown("<a id='contact-us'></a>", unsafe_allow_html=True)
    st.header("📬 Contact Us")
    st.write("Email: [ijindal2005@gmail.com](mailto:ijindal2005@gmail.com) | [meharbhanwra1004@gmail.com](mailto:meharbhanwra1004@gmail.com)")
    st.write("GitHub: [github.com/ishanijdev](https://github.com/ishanijdev) | [github.com/meharbhanwra](https://github.com/meharbhanwra)")
    st.markdown("</div>", unsafe_allow_html=True)


# Footer
st.divider()
st.markdown("<p style='text-align: center; color: #555;'>Created by Ishani Jindal & Mehar Bhanwra | Powered by FER2013</p>", unsafe_allow_html=True)